"""
Unlimited-OCR 客户端（转发到 Docker 容器）

链路：FastAPI /unlimited-ocr/* → Docker 容器（默认 localhost:8090）

真实端点 /v1/ocr/multi：
- images 支持 路径 / URL / base64；实测需用 data: URI（裸 base64 会被当成文件路径）。
- enable_grounding=true 返回 markdown，版面与文字都编码进 <|det|> 标注：
      <|det|>label [x1,y1,x2,y2]<|/det|>文字/表格/公式内容
  bbox 为像素坐标（与原图同分辨率），文字/表格HTML/公式须从标注后的整段内容里取。
- crops：容器返回的是容器内路径，代理不可达 → 统一按 bbox 从原图裁剪（base64 内联）。
容器未启动 / 不可达时，返回明确错误，不抛异常中断主服务。
"""
import base64
import io
import json
import re
import threading
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from PIL import Image

from . import config
from .layout_postprocess import postprocess_layout

_label_map = {
    "text": "paragraph",
    "title": "title",
    "paragraph": "paragraph",
    "table": "table",
    "formula": "formula",
    "image": "figure",
    "chart": "figure",
    "figure": "figure",
    "header": "header",
    "footer": "footer",
    "equation": "formula",
    "equation_block": "formula",
}

# 仅匹配 det 开标注，内容（可能跨多行、含 <table>）留待按块截取
_DET_RE = re.compile(r"<\|det\|>(\w+)\s+\[([\d.,\s]+)\]<\|/det\|>")
_TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL)
# 图注识别：普通文字若疑似图片说明，则归并为 figure 占位（不裁真实图）。
# 必须非常保守：
#   - 不含"如图所示"等题目正文常见短语（它们不是图注，误判会把整段题目变成 🖼 图片占位）
#   - 仅匹配"图N：/图N："、"见下图"、"上图/下图"、"插图/图片/图示"等明确图注标记
#   - 且文字较短（图注通常是一句话，不会是长段题目正文）
_CAPTION_RE = re.compile(r"(图\s*[\d一二三四五六七八九十]+\s*[:：]|见[上下]图|上图|下图|插图|图片|图示)")
_CAPTION_MAX_LEN = 40  # 超过此长度的文本视为题目正文，不归并为图注
# 控制字符（去 \t \r 等模型偶发噪声，保留换行）
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _normalize_text(s: str) -> str:
    """清洗识别文本：LaTeX 定界符转 $、去除控制字符。"""
    if not s:
        return s
    s = s.replace("\\(", "$").replace("\\)", "$").replace("\\[", "$$").replace("\\]", "$$")
    s = _CTRL_RE.sub("", s)
    return s.strip()


def _dedup_elements(elements: list) -> list:
    """去除引擎偶发的重复块（同 type + 近同 bbox + 近似内容），避免叠加乱层。

    - bbox 按 5px 网格量化，容忍引擎把同一块输出成差 1~2px 的重复；
    - 内容取「仅保留中英文字符与数字」的签名，容忍同区域两次转录的细微差异
      （如多一个撇号 / 标点），避免精确比对漏掉近乎重复的块。
    """
    seen = set()
    out = []
    for e in elements:
        b = e.get("bbox") or []
        qb = tuple(round(v / 5) * 5 for v in b)
        sig = re.sub(r"[^\w\u4e00-\u9fff]", "", e.get("content") or "")
        key = (e.get("type"), qb, sig)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# 题号前缀提取：从内容开头匹配 "4." / "4、" / "4）" 等题号标记
# 兼容 VLM 常见变体：前导 markdown（**4.*/# 4.）、前导空白、中文括号
_QNUM_PREFIX_RE = re.compile(
    r'^\s*(?:'
    r'\*{1,3}\s*'        # markdown 加粗/斜体 **4. *4.
    r'|#{1,6}\s*'        # markdown 标题 # 4.
    r')?'
    r'(\d+[.、)\）])'     # 实际题号：数字 + 中文/英文标点
)


def _extract_qnum_prefix(content: str) -> str | None:
    """提取题号前缀（如 "4."），无则返回 None。"""
    if not content:
        return None
    m = _QNUM_PREFIX_RE.match(content)
    return m.group(1) if m else None


def _normalize_content_head(content: str, max_chars: int = 50) -> str:
    """提取内容「有效首行」用于相似度比对：去空白/markdown/标点，取前 max_chars。

    无论 VLM 输出 "4." / "**4.**" / "\\n4." / "4、" 都能归一化为相同签名。
    """
    if not content:
        return ''
    # 去前导空白、markdown、LaTeX
    s = content.strip()
    s = re.sub(r'^[\s\*#>`\-\=]+', '', s)   # 去前导 markdown
    # 取第一行（遇到 \\n 或选项行就截断）
    first_line = s.split('\n')[0]
    # 去除纯标点/空格，保留中英文数字
    sig = re.sub(r'[^\w\u4e00-\u9fff]', '', first_line)
    return sig[:max_chars].lower()


def _is_page_footer(content: str) -> bool:
    """检测是否为 PDF 页脚/页码文字（如 "第1页（共14页）" / "- 3 -" / "Page 5"）。
    
    这些不应作为正文渲染，或至少应 nowrap 不换行。
    """
    if not content or len(content.strip()) > 60:
        return False
    s = content.strip()
    return bool(re.match(
        r'^(?:'
        r'第\s*\d+\s*页(?:[（(]\s*共\s*\d+\s*页[)）])?'  # 第1页（共14页）
        r'|-\s*\d+\s*-|'                                   # - 3 -
        r'|Page\s*\d+'                                     # Page 5
        r'|^\d+\s*/\s*\d+$'                                # 1 / 14
        r')$',
        s, re.IGNORECASE
    ))


def _norm_line_for_dedup(line: str) -> str:
    """单行归一化（用于合并时去重比对）：去前导 markdown/空白、去标点，保留中英文数字。"""
    s = line.strip()
    s = re.sub(r'^[\s\*#>`\-\=]+', '', s)
    s = re.sub(r'[^\w\u4e00-\u9fff]', '', s)
    return s.lower()


def _merge_block_contents(parts: list) -> str:
    """拼接多块内容，去除重复行。

    同一道题被 VLM 拆成多块时，题干（首行）在每块里都重复出现——
    若直接拼接会导致「题目文字重复」。这里按归一化行做去重，
    只保留首次出现的行，同时保留 A/B/C/D 等互不相同的选项。
    """
    seen: set = set()
    out_lines: list = []
    for p in parts:
        for line in (p or '').split('\n'):
            n = _norm_line_for_dedup(line)
            if not n:
                # 保留单个空行作为分隔，避免连续空行
                if out_lines and out_lines[-1] != '':
                    out_lines.append('')
                continue
            if n in seen:
                continue
            seen.add(n)
            out_lines.append(line.rstrip())
    while out_lines and out_lines[-1] == '':
        out_lines.pop()
    return '\n'.join(out_lines)


def _dedup_question_duplicates(elements: list) -> tuple:
    """去除「同一道题被引擎拆成多块」的重复 + 标记页脚。

    返回 (merged_list, merged_y_ranges)：
      - merged_list: 去重后的元素列表
      - merged_y_ranges: list of (y_min, y_max)，每个合并组覆盖的 y 范围。
        调用方可用此信息校正夹在合并组之间的 figure 元素 bbox，
        避免 VLM 拆块时下块的 figure bbox 偏低导致裁剪顶部被截。
    """
    # 第一步：标记页脚
    for e in elements:
        if _is_page_footer(e.get('content') or ''):
            e['_is_footer'] = True

    # 第二步：按内容首行签名分组（模糊匹配同题）
    groups: dict[str, list] = {}
    order: list = []
    for e in elements:
        head = _normalize_content_head(e.get('content') or '')
        if len(head) >= 6:  # 足够长的有意义内容才参与分组
            key = head
        else:
            key = f'__short_{id(e)}'
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(e)

    merged = []
    merged_y_ranges = []  # 新增：记录每个合并组的 y 范围

    for key in order:
        grp = groups[key]

        # 短内容/唯一元素直接保留
        if len(grp) == 1 or key.startswith('__short__'):
            merged.append(grp[0])
            continue

        # 多个相似元素 → 合并为一个（同一题被 VLM 拆成多块）
        first = grp[0]
        bboxes = [e.get('bbox', [0, 0, 0, 0])
                  for e in grp if e.get('bbox') and len(e['bbox']) == 4]
        merged_bbox = [
            min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes),
        ] if bboxes else (first.get('bbox') or [0, 0, 0, 0])

        parts = [e.get('content', '') for e in grp]
        me = dict(first)
        me['bbox'] = merged_bbox
        me['content'] = _merge_block_contents(parts)
        merged.append(me)

        # 记录合并组 y 范围（用于校正夹在中间的 figure bbox）
        if bboxes:
            merged_y_ranges.append((merged_bbox[1], merged_bbox[3]))

    return merged, merged_y_ranges


def _parse_det_blocks(md: str) -> List[tuple]:
    """返回 (label, [x1,y1,x2,y2], content) 列表。

    content 为从 <|/det|> 到下一个 <|det|>（或文末）的整段文本，
    含换行与 <table> 等结构化内容——这对表格/公式/多行段落至关重要。
    """
    out = []
    matches = list(_DET_RE.finditer(md or ""))
    for i, m in enumerate(matches):
        label = m.group(1)
        coords = [float(x) for x in re.findall(r"[\d.]+", m.group(2))]
        if len(coords) != 4:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        content = md[start:end].strip()
        out.append((label, coords, content))
    return out


def _clean_latex(s: str) -> str:
    """剥离 LaTeX 的 \[ \] / \( \) / $ 包裹，保留内部公式文本。"""
    s = s.strip()
    s = s.strip("$").strip()
    s = s.strip("\\").strip()
    s = s.strip("[]()").strip()
    s = s.strip("\\").strip()
    return s.strip()


# 行首判定"新逻辑块"：题目序号(1. / 2、/ （三）)、选项(A. / B、)、中文序号(一、/ 二。)
# 这些行即便与上一行 y 间距很小、水平重叠，也应独立成块，避免"题目+答案被并成一行"。
_BLOCK_STARTER_RE = re.compile(
    r"^\s*(?:"
    r"\d+[.、)）]|"                      # 1. 2、 3)
    r"[A-Za-z][.、)）]|"                  # A. B、 C)
    r"[（(][0-9一二三四五六七八九十百千]+[)）]|"  # （1）（三）
    r"[一二三四五六七八九十百千]+[.、、]"         # 一、 二、
    r")"
)


def _is_block_starter(text: str) -> bool:
    """该行是否开启一个新的逻辑块（题号 / 选项 / 中文序号）。"""
    if not text:
        return False
    return bool(_BLOCK_STARTER_RE.match(text))


def _merge_adjacent_elements(elements: list, page_h: float = 1200) -> list:
    """将相邻同类的行级 bbox 合并为块级段落。

    Unlimited-OCR 基于 Qwen2-VL grounding 返回的是逐行检测框。
    合并条件（全部满足才合并）：
      1. 类型完全相同且属于可合并集（paragraph/header/footer，title 不并入）
      2. 与上一组的 y 间距 <= 页面高度 * 2%
      3. 水平方向重叠 >= 25%
      4. 当前行不是"新逻辑块"起始行（题号/选项/中文序号），否则独立成块
    """
    if not elements or len(elements) <= 1:
        return elements

    _OK_TYPES = {"paragraph", "header", "footer"}
    _Y_GAP_RATIO = 0.02
    _X_OVERLAP = 0.25

    groups = []
    cur = [0, 0]

    for j in range(1, len(elements)):
        prev_e = elements[cur[1]]
        curr_e = elements[j]

        pb = prev_e.get("bbox", [])
        cb = curr_e.get("bbox", [])
        if len(pb) != 4 or len(cb) != 4:
            groups.append(list(cur)); cur = [j, j]; continue

        pt, ct = prev_e.get("type", ""), curr_e.get("type", "")

        # 仅同类型才合并（防止 title 被并入其后的段落）
        type_ok = (pt == ct) and (pt in _OK_TYPES)

        # 垂直间距
        y_gap = cb[1] - pb[3]
        y_ok = (-page_h * 0.5) <= y_gap <= (page_h * _Y_GAP_RATIO)

        # 水平重叠
        ix1 = max(pb[0], cb[0])
        ix2 = min(pb[2], cb[2])
        pw = pb[2] - pb[0]
        cw = cb[2] - cb[0]
        overlap_w = max(0, ix2 - ix1)
        x_ok = (overlap_w / max(pw, cw)) >= _X_OVERLAP if max(pw, cw) > 0 else False

        # 当前行是题号/选项等"新块"起始 → 不并入上一组，独立成块
        block_ok = not _is_block_starter(curr_e.get("content", "") or "")

        if type_ok and y_ok and x_ok and block_ok:
            cur[1] = j
        else:
            groups.append(list(cur))
            cur = [j, j]
    groups.append(list(cur))

    merged = []
    for g_start, g_end in groups:
        first = elements[g_start]
        last = elements[g_end]
        fb = first.get("bbox", [0, 0, 0, 0])
        lb = last.get("bbox", [0, 0, 0, 0])
        merged_bbox = [
            min(fb[0], lb[0]),
            min(fb[1], lb[1]),
            max(fb[2], lb[2]),
            max(fb[3], lb[3]),
        ]
        parts = []
        for k in range(g_start, g_end + 1):
            c = elements[k].get("content", "")
            if c:
                parts.append(c)
        me = dict(first)
        me["bbox"] = merged_bbox
        me["content"] = "\n".join(parts)
        if me.get("type") == "formula":
            me["latex"] = _clean_latex(me["content"])
        merged.append(me)
    return merged


_lock = threading.Lock()


def _build_url(path: str) -> str:
    base = config.UNLIMITED_OCR_URL.rstrip("/")
    return f"{base}{path}"


def _to_data_uri(b64: str) -> str:
    if b64.startswith("data:"):
        return b64
    return f"data:image/png;base64,{b64}"


def _decode_dims(b64: str) -> tuple:
    """返回 (PIL.Image, width, height)；失败返回 (None,0,0)。"""
    try:
        _, data = (b64.split(",", 1) + [""])[:2] if "," in b64 else ("", b64)
        pil = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        return pil, pil.size[0], pil.size[1]
    except Exception:  # noqa: BLE001
        return None, 0, 0


def _detect_and_scale_coords(elements: list, width: float, height: float) -> None:
    """检测引擎坐标是否为归一化坐标（0~1000 空间），若是则逐元素缩放到像素坐标。

    Unlimited-OCR（baidu/Unlimited-OCR）基于 Qwen2-VL grounding，
    返回的 bbox 坐标为归一化值（0~1000），不匹配输入图片的实际像素尺寸。

    鲁棒判据（三条件全部满足才判定为归一化）：
      1. 全局最大坐标值聚集在 ~1000 附近（900~1100，容忍模型输出噪声）；
      2. 最大坐标显著小于对应图片尺寸（< 80%），排除「已是像素坐标」的情况；
      3. 至少有半数元素的 bbox 跨度合理（宽高均 > 图片的 1%），排除空框干扰。
    """
    if not elements or width <= 0 or height <= 0:
        return
    valid = [e for e in elements if e.get("bbox") and len(e["bbox"]) == 4]
    if not valid:
        return

    max_cx = max(b[2] for b in (e["bbox"] for e in valid))
    max_cy = max(b[3] for b in (e["bbox"] for e in valid))

    # 条件 1 & 2：坐标聚集在 ~1000 且显著小于图片尺寸
    near_1000_x = 900 <= max_cx <= 1100 and max_cx < width * 0.8
    near_1000_y = 900 <= max_cy <= 1100 and max_cy < height * 0.8
    is_normalized = near_1000_x or near_1000_y

    if not is_normalized:
        return

    sx = width / 1000.0
    sy = height / 1000.0
    for e in valid:
        b = e["bbox"]
        e["bbox"] = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]


def _crop_to_base64(pil: Image.Image, bbox: List[float], pad: int = 12) -> Optional[str]:
    try:
        w, h = pil.size
        x1 = max(0, int(bbox[0]) - pad)
        y1 = max(0, int(bbox[1]) - pad)
        x2 = min(w, int(bbox[2]) + pad)
        y2 = min(h, int(bbox[3]) + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = pil.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def health() -> dict:
    url = _build_url(config.UNLIMITED_OCR_HEALTH_PATH)
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            body = json.loads(r.read().decode("utf-8"))
            body["ok"] = True
            return body
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Unlimited-OCR 容器不可达 ({url}): {e}"}


def parse(images: List[str], prompt: str, doc_id: str = "") -> dict:
    """解析图片，返回统一 {ok, pages, error}。"""
    url = _build_url(config.UNLIMITED_OCR_PARSE_PATH)
    data_uris = [_to_data_uri(x) for x in images]
    payload = {
        "images": data_uris,
        "prompt": prompt or config.UNLIMITED_OCR_PROMPT,
        "enable_grounding": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:500]}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Unlimited-OCR 转发失败 ({url}): {e}"}

    if not isinstance(resp, dict) or not resp.get("success", True):
        return {"ok": False, "error": resp.get("error") or "Unlimited-OCR 返回失败"}

    results = resp.get("results", []) or []
    if not results:
        return {"ok": True, "pages": []}

    pages = []
    for i, img_b64 in enumerate(images):
        pil, width, height = _decode_dims(img_b64)
        res = results[i] if i < len(results) else results[0]
        markdown = res.get("text", "") or ""
        det_blocks = _parse_det_blocks(markdown)
        elements = []
        crops = []

        if det_blocks:
            # 优先：从 markdown 的 <|det|> 标注取 label + bbox + 整段内容（三者同源，最准）
            for _idx, (label, bx, content) in enumerate(det_blocks):
                etype = _label_map.get(label, "paragraph")
                content = _normalize_text(content)
                el = {
                    "type": etype,
                    "bbox": bx,
                    "content": "",
                    "confidence": None,
                    "latex": None,
                    "html": None,
                    "caption": None,
                    "is_caption": False,
                }
                if etype == "table":
                    tbl = _TABLE_RE.search(content)
                    if tbl:
                        el["html"] = tbl.group(0)
                    el["content"] = _TABLE_RE.sub("", content).strip()
                elif etype == "formula":
                    el["latex"] = _clean_latex(content)
                    el["content"] = content
                elif etype == "figure":
                    el["caption"] = content
                else:
                    el["content"] = content
                # 图注识别：普通文字疑似图片说明 → 归并为 figure 占位（不裁真实图）
                # 仅当文字较短且匹配明确图注标记时才归并，避免把含"如图所示"的题目正文误判为图注
                if etype in ("paragraph", "title") and content and len(content) <= _CAPTION_MAX_LEN and _CAPTION_RE.search(content):
                    el["type"] = "figure"
                    el["caption"] = content
                    el["is_caption"] = True
                elements.append(el)

            # 去重（引擎偶发同块重复）→ 合并相邻同类行级框为块级段落
            elements = _dedup_elements(elements)
            elements = _merge_adjacent_elements(elements, height or 1200)
            # 题号去重：VLM 常把同一道题拆成多块（如上块含A/B、下块含C/D），按内容签名合并
            elements, merged_y_ranges = _dedup_question_duplicates(elements)

            # 坐标归一化修正：引擎返回 ~1000 空间坐标 → 缩放到实际像素
            _detect_and_scale_coords(elements, width, height)

            # 校正夹在合并组之间的 figure bbox：
            # VLM 拆块时下块的 figure（如C/D选项图）bbox y1 偏低 → 裁剪时顶部被截。
            # 将落在合并组 y 范围内（或紧随其后）的 figure 的 y1 向上扩展到合并组顶部。
            # 容差：上半区 20px（figure 可能略高于文字），下半区 200px（第二子块的 figure
            # 在其文字段下方，距合并组 y_max 可能较远）。
            if merged_y_ranges:
                for e in elements:
                    if (e.get('type') == 'figure' and not e.get('is_caption')
                            and e.get('bbox') and len(e['bbox']) == 4):
                        b = e['bbox']
                        for gy_min, gy_max in merged_y_ranges:
                            # figure 顶部在合并组范围内或紧随其后 → 需要校正
                            if gy_min - 20 <= b[1] <= gy_max + 200 and b[1] > gy_min + 5:
                                e['bbox'] = [b[0], gy_min, b[2], b[3]]
                                break

            # 生成 crops：仅"真实图片"（figure 且非图注文字）才从原图裁区域
            for ei, e in enumerate(elements):
                if e["type"] == "figure" and not e.get("is_caption") and pil is not None:
                    crop_b64 = _crop_to_base64(pil, e["bbox"])
                    if crop_b64:
                        crops.append({
                            "filename": f"{doc_id}_page{i+1}_fig{ei}.png",
                            "base64": crop_b64,
                            "element_index": ei,
                            "bbox": list(e["bbox"]),
                        })
        else:
            # 回退：bboxes 数组（label 字段，desc 通常为空），仅给位置，无内容
            bboxes = res.get("bboxes", []) or []
            for _idx, box in enumerate(bboxes):
                bx = box.get("bbox", box.get("bbox_2d"))
                if not bx:
                    bx = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
                if not bx or len(bx) != 4:
                    continue
                etype = _label_map.get(box.get("label", "text"), "paragraph")
                content = box.get("desc", "") or ""
                el = {
                    "type": etype,
                    "bbox": [float(v) for v in bx],
                    "content": content,
                    "confidence": None,
                    "latex": content if etype == "formula" else None,
                    "html": None,
                    "caption": content if etype == "figure" else None,
                    "is_caption": False,
                }
                elements.append(el)
            elements = _merge_adjacent_elements(elements, height or 1200)
            _detect_and_scale_coords(elements, width, height)

        postprocess_layout(elements)
        pages.append({
            "page": i + 1,
            "width": width,
            "height": height,
            "markdown": markdown,
            "elements": elements,
            "crops": crops,
        })
    return {"ok": True, "pages": pages}
