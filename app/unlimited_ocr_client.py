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
# 图注识别：普通文字若疑似图片说明，则归并为 figure 占位（不裁真实图）
_CAPTION_RE = re.compile(r"(插图|图片|图示|如图所示|见[上下]图|图\s*[:：]|上图|下图)")
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


def _merge_adjacent_elements(elements: list, page_h: float = 1200) -> list:
    """将相邻同类的行级 bbox 合并为块级段落。

    Unlimited-OCR 基于 Qwen2-VL grounding 返回的是逐行检测框。
    合并条件（全部满足才合并）：
      1. 类型完全相同且属于可合并集（paragraph/header/footer，title 不并入）
      2. 与上一组的 y 间距 <= 页面高度 * 2%
      3. 水平方向重叠 >= 25%
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

        if type_ok and y_ok and x_ok:
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
    """检测引擎坐标是否为归一化坐标（~1000 空间），若是则逐元素缩放到像素坐标。

    Unlimited-OCR（baidu/Unlimited-OCR）基于 Qwen2-VL grounding，
    返回的 bbox 坐标为归一化值（0~1000），不匹配输入图片的实际像素尺寸。
    判据：任一 bbox 坐标超过图片宽高 → 判定为归一化坐标系。
    """
    if not elements or width <= 0 or height <= 0:
        return
    needs_scale = False
    for e in elements:
        b = e.get("bbox")
        if b and len(b) == 4:
            if b[2] > width or b[3] > height or b[0] < 0 or b[1] < 0:
                needs_scale = True
                break
    if not needs_scale:
        return
    sx = width / 1000.0
    sy = height / 1000.0
    for e in elements:
        b = e.get("bbox")
        if b and len(b) == 4:
            e["bbox"] = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]


def _crop_to_base64(pil: Image.Image, bbox: List[float], pad: int = 3) -> Optional[str]:
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
                if etype in ("paragraph", "title") and content and _CAPTION_RE.search(content):
                    el["type"] = "figure"
                    el["caption"] = content
                    el["is_caption"] = True
                elements.append(el)

            # 去重（引擎偶发同块重复）→ 合并相邻同类行级框为块级段落
            elements = _dedup_elements(elements)
            elements = _merge_adjacent_elements(elements, height or 1200)

            # 坐标归一化修正：引擎返回 ~1000 空间坐标 → 缩放到实际像素
            _detect_and_scale_coords(elements, width, height)

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

        pages.append({
            "page": i + 1,
            "width": width,
            "height": height,
            "markdown": markdown,
            "elements": elements,
            "crops": crops,
        })
    return {"ok": True, "pages": pages}
