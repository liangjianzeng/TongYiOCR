"""
代理层统一版面后处理：阅读顺序 + 逻辑块（选项组）识别。

三引擎（GLM-OCR / Unlimited-OCR / PaddleOCR-VL）返回的 elements[] 都只有
bbox + content，缺少「阅读顺序(reading order)」和「逻辑块(logical block)」信息，
导致前端只能用 position:absolute 按 bbox% 死定位，任意 bbox 偏差都会错位 / 堆叠。

本模块在三个 client 解析出 elements 后统一调用，给每个 element 原地补充：
  - reading_order : int   全局阅读顺序（按中心 y 聚类成视觉行 → 行内按 x 排序）
  - block_id      : int   逻辑块编号（同一视觉行 / 选项组共享，前端结构化分组用）
  - is_option     : bool  是否多选题选项（A./B./C./D.），前端选项块用 grid 成组

这样前端既能保留「排版还原（绝对定位对照）」，又能新增「结构化视图（阅读流）」，
对所有引擎统一受益，从根本解决「图文堆叠」「ABCD 答案错乱」。
"""
import re

# 选项标记：A./B./C./D.（或中文顿号/括号），以及 ①②③④
_OPTION_RE = re.compile(r"^\s*([A-Da-d①-④])[.、)）]\s*")

# 逻辑块起始：题号 / 中文序号 / 括号序号（与合并逻辑一致的意图）
_BLOCK_STARTER_RE = re.compile(
    r"^\s*(?:"
    r"\d+[.、)）]|"
    r"[A-Za-z][.、)）]|"
    r"[（(][0-9一二三四五六七八九十百千]+[)）]|"
    r"[一二三四五六七八九十百千]+[.、、]"
    r")"
)


def _is_option(text: str) -> bool:
    if not text:
        return False
    return bool(_OPTION_RE.match(text))


def _is_block_starter(text: str) -> bool:
    if not text:
        return False
    return bool(_BLOCK_STARTER_RE.match(text))


def _rows_by_overlap(elements):
    """按中心 y 聚类成视觉行：两元素 y 区间接近（重叠或间距 < 行高容差）即同属一行。

    返回 list[list[element]]，每行内元素仍保持原顺序（后续调用方自行按 x 排序）。
    仅处理带合法 bbox 的元素。
    """
    valid = [e for e in elements if e.get("bbox") and len(e["bbox"]) == 4]
    if not valid:
        return []
    page_h = max(e["bbox"][3] for e in valid) or 1000
    valid.sort(key=lambda e: (e["bbox"][1] + e["bbox"][3]) / 2)  # 按中心 y 升序
    rows = []
    for e in valid:
        b = e["bbox"]
        placed = False
        for row in rows:
            ry1 = min(x["bbox"][1] for x in row)
            ry2 = max(x["bbox"][3] for x in row)
            # 容差用「固定页面比例」而非「行高比例」：若用行高*0.5，遇到高图
            # （如几何图 bbox 很高）会把下一行也吸入同一行，破坏阅读顺序。
            tol = max(8, page_h * 0.02)
            if not (b[3] < ry1 - tol or b[1] > ry2 + tol):
                row.append(e)
                placed = True
                break
        if not placed:
            rows.append([e])
    return rows


def assign_reading_order(elements):
    """给每个 element 加 reading_order（全局阅读顺序）。"""
    rows = _rows_by_overlap(elements)
    order = 0
    for row in rows:
        row.sort(key=lambda e: (e["bbox"][0] + e["bbox"][2]) / 2)  # 行内按中心 x
        for e in row:
            e["reading_order"] = order
            order += 1
    # 无 bbox 元素排最后
    for e in elements:
        if "reading_order" not in e:
            e["reading_order"] = order
            order += 1
    return elements


def detect_option_blocks(elements):
    """给 element 加 block_id（逻辑块）+ is_option。

    规则：
      - 普通视觉行：整行共享一个 block_id（块内元素按阅读顺序渲染）。
      - 选项行（同行 >=2 个 A./B./C./D. 选项）：这些选项共享一个 block_id 且
        is_option=True，前端据此用 grid 横排成组；同行非选项残片单独成块。
    """
    if not elements:
        return elements
    rows = _rows_by_overlap(elements)
    block_id = 0
    for row in rows:
        row_sorted = sorted(row, key=lambda e: (e["bbox"][0] + e["bbox"][2]) / 2)
        opt_els = [e for e in row_sorted if _is_option(e.get("content", "") or "")]
        if len(opt_els) >= 2:
            # 选项行：选项们成组（is_option=True，共享 block）
            block_id += 1
            for e in opt_els:
                e["block_id"] = block_id
                e["is_option"] = True
            # 同行的非选项元素（如题干残片、说明）单独成块
            non_opt = [e for e in row_sorted if e not in opt_els]
            for e in non_opt:
                block_id += 1
                e["block_id"] = block_id
                e["is_option"] = False
        else:
            # 普通行：整行共享一个 block（无选项标记）
            block_id += 1
            for e in row_sorted:
                e["block_id"] = block_id
                e["is_option"] = bool(_is_option(e.get("content", "") or ""))
    # 兜底无 bbox 元素
    for e in elements:
        if "block_id" not in e:
            block_id += 1
            e["block_id"] = block_id
            e["is_option"] = False
    return elements


def postprocess_layout(elements):
    """统一入口：阅读顺序 + 逻辑块，原地补充字段并返回。"""
    if not elements:
        return elements
    assign_reading_order(elements)
    detect_option_blocks(elements)
    return elements
