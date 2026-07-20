"""
GLM-OCR 解析客户端（转发本机 llamacpp VLM :8089）

链路：FastAPI /glmocr/* → glmocr SDK(selfhosted) → llama.cpp VLM(8089) → Markdown + elements + crops

设计要点：
- 无状态：crops 以 base64 内联返回，不落盘。
- 本地模型：通过 _dotted 覆盖 pipeline.layout.model_dir 指向本地 PP-DocLayoutV3 快照，
  绕开 HuggingFace 联网（本机无外网时必备）。
- 懒加载单例：按 (host,port,device) 缓存 GlmOcr 实例，避免每次请求重载版面模型。
- 输出统一：pages[{page,width,height,markdown,elements[],crops[]}]，bbox 反归一化为像素。
- crops：优先用 SDK 原生裁剪图；本机 selfhosted 下 image_files 为 None，统一改为按元素 bbox
  从原图裁剪（与 elements 一一对应）。
"""
import os
import base64
import io
import re
import json
import tempfile
import threading
import traceback
import urllib.request
from typing import List, Optional, Any, Tuple, Dict

from PIL import Image

from . import config
from .layout_postprocess import postprocess_layout

try:
    from glmocr import GlmOcr
    _GLMOCR_AVAILABLE = True
    _GLMOCR_IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    GlmOcr = None
    _GLMOCR_AVAILABLE = False
    _GLMOCR_IMPORT_ERR = str(e)
    print(f"[glmocr_client] WARNING: glmocr SDK 未导入: {e}", flush=True)


# ---------- 懒加载单例 ----------
_parsers: Dict[Tuple[str, int, str], Any] = {}
_parsers_lock = threading.Lock()
_parse_lock = threading.Lock()  # 串行化解析（SDK 实例非线程安全）

# GLM-OCR label → 统一元素类型
_LABEL_TYPE_MAP = {
    "doc_title": "title",
    "paragraph_title": "subtitle",
    "title": "title",
    "subtitle": "subtitle",
    "text": "paragraph",
    "content": "paragraph",
    "abstract": "paragraph",
    "algorithm": "paragraph",
    "reference_content": "paragraph",
    "vision_footnote": "paragraph",
    "formula_number": "paragraph",
    "seal": "paragraph",
    "table": "table",
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula": "formula",
    "image": "figure",
    "chart": "figure",
    "figure": "figure",
    "header": "header",
    "footer": "footer",
}


def glmocr_available() -> bool:
    return _GLMOCR_AVAILABLE


def engine_health() -> dict:
    """探测本机 llama.cpp VLM 引擎（8089）是否可达。"""
    url = f"http://{config.GLM_OCR_API_HOST}:{config.GLM_OCR_API_PORT}/health"
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            return {"ok": r.status == 200, "llama_port": config.GLM_OCR_API_PORT}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "llama_port": config.GLM_OCR_API_PORT, "error": str(e)}


def _get_parser(host: str, port: int, device: str):
    key = (host, port, device)
    with _parsers_lock:
        p = _parsers.get(key)
        if p is None:
            overrides = {}
            if config.GLM_OCR_LAYOUT_MODEL_DIR:
                overrides["_dotted"] = {
                    "pipeline.layout.model_dir": config.GLM_OCR_LAYOUT_MODEL_DIR
                }
            p = GlmOcr(
                mode="selfhosted",
                ocr_api_host=host,
                ocr_api_port=port,
                layout_device=device,
                **overrides,
            )
            _parsers[key] = p
        return p


def _decode_image(b64: str) -> Tuple[Image.Image, Optional[str]]:
    """解码 base64（支持 data: 前缀）为 PIL 图；同时落盘临时文件供 SDK 使用。

    返回 (PIL Image, temp_path_or_None)。
    """
    header, b64_data = (b64.split(",", 1) + [""])[:2] if "," in b64 else ("", b64)
    img_bytes = base64.b64decode(b64_data)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # 落盘临时文件（GLM-OCR SDK 需要文件路径输入）
    fd, path = tempfile.mkstemp(suffix=".png", prefix="glmocr_")
    with os.fdopen(fd, "wb") as f:
        f.write(img_bytes)
    return pil, path


def _parse_bbox(raw) -> Optional[List[float]]:
    """解析 bbox（可能是字符串 '[x1,y1,x2,y2]' 或列表），返回 4 个浮点，失败返回 None。"""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            return [float(x) for x in raw]
        except Exception:
            return None
    if isinstance(raw, str):
        s = raw.strip()
        try:
            return [float(x) for x in json.loads(s)]
        except Exception:
            nums = re.findall(r"-?\d+\.?\d*", s)
            if len(nums) >= 4:
                return [float(x) for x in nums[:4]]
    return None


def _denorm(bbox_norm: List[float], width: int, height: int) -> List[int]:
    """归一化 0-1000 → 像素坐标，并裁剪到图像范围内。"""
    x1 = max(0, min(width, int(bbox_norm[0] / 1000 * width)))
    y1 = max(0, min(height, int(bbox_norm[1] / 1000 * height)))
    x2 = max(0, min(width, int(bbox_norm[2] / 1000 * width)))
    y2 = max(0, min(height, int(bbox_norm[3] / 1000 * height)))
    return [x1, y1, x2, y2]


def _crop_to_base64(pil: Image.Image, bbox_px: List[int], pad: int = 3) -> Optional[str]:
    """按像素 bbox 从原图裁剪为 PNG base64（data URI）。"""
    try:
        w, h = pil.size
        x1 = max(0, bbox_px[0] - pad)
        y1 = max(0, bbox_px[1] - pad)
        x2 = min(w, bbox_px[2] + pad)
        y2 = min(h, bbox_px[3] + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = pil.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:  # noqa: BLE001
        return None


def _build_elements_and_crops(
    jr_page, pil: Image.Image, width: int, height: int, doc_id: str, page: int
):
    """由 GLM-OCR json_result 单页构造统一 elements + crops（按 bbox 从原图裁剪）。"""
    elements: List[dict] = []
    crops: List[dict] = []
    if not isinstance(jr_page, list):
        return elements, crops
    for idx, item in enumerate(jr_page):
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("native_label") or "text"
        etype = _LABEL_TYPE_MAP.get(label, "paragraph")
        bbox_norm = _parse_bbox(item.get("bbox_2d"))
        content = item.get("content", "") or ""
        confidence = item.get("score")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None

        el: dict = {
            "type": etype,
            "bbox": [0, 0, 0, 0],
            "content": content,
            "confidence": confidence,
            "latex": None,
            "html": None,
            "caption": None,
        }
        if etype == "formula":
            el["latex"] = content
        elif etype == "table":
            el["html"] = None  # GLM-OCR 文本态未单独返回 HTML，留空

        if bbox_norm:
            bbox_px = _denorm(bbox_norm, width, height)
            el["bbox"] = bbox_px
            crop_b64 = _crop_to_base64(pil, bbox_px)
            if crop_b64:
                fname = f"{doc_id}_page{page}_type{idx}.png"
                crops.append({
                    "filename": fname,
                    "base64": crop_b64,
                    "element_index": len(elements),   # 该 el 即将 append，索引即 len(elements)
                    "bbox": bbox_px,
                })
        elements.append(el)
    return elements, crops


def parse(
    images: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    doc_id: str = "",
    llama_host: Optional[str] = None,
    llama_port: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> dict:
    """解析图片，返回统一 {ok, pages, error, json_result}。

    - images: base64 data URI 列表（远程调用方）
    - image_paths: 本地文件路径列表（兼容旧格式）
    - 无状态：crops 以 base64 内联，不落盘。
    """
    if not _GLMOCR_AVAILABLE:
        raise RuntimeError(f"glmocr SDK 未安装: {_GLMOCR_IMPORT_ERR}")

    host = llama_host or config.GLM_OCR_API_HOST
    port = int(llama_port if llama_port is not None else config.GLM_OCR_API_PORT)
    device = config.GLM_OCR_LAYOUT_DEVICE

    originals: List[Image.Image] = []
    temp_files: List[Optional[str]] = []
    try:
        if images:
            for b64 in images:
                pil, path = _decode_image(b64)
                originals.append(pil)
                temp_files.append(path)
        if image_paths and not originals:
            for p in image_paths:
                pil = Image.open(p).convert("RGB")
                originals.append(pil)
                temp_files.append(None)  # 不删除调用方的文件

        if not originals:
            raise ValueError("images 或 image_paths 必须提供至少一项")

        resolved_paths = [t for t in temp_files if t] if images else list(image_paths)

        parser = _get_parser(host, port, device)
        with _parse_lock:
            results = parser.parse(resolved_paths, preserve_order=True)
        if not isinstance(results, list):
            results = [results]

        pages = []
        raw_json = []
        for i, result in enumerate(results):
            pil = originals[i]
            width, height = pil.size
            md = result.markdown_result or ""
            # GLM-OCR 的 json_result 是「list-of-pages」：json_result[page_idx][region_idx]
            jr = result.json_result
            if isinstance(jr, list) and jr and isinstance(jr[0], list):
                jr_page = jr[i] if i < len(jr) else jr[-1]
            elif isinstance(jr, list):
                jr_page = jr  # 已是扁平元素列表（兼容）
            else:
                jr_page = []
            raw_json.append(jr_page)
            elements, crops = _build_elements_and_crops(
                jr_page, pil, width, height, doc_id, i + 1
            )
            postprocess_layout(elements)
            pages.append({
                "page": i + 1,
                "width": width,
                "height": height,
                "markdown": md,
                "elements": elements,
                "crops": crops,
            })

        return {
            "ok": True,
            "engine": f"glmocr:{host}:{port}",
            "doc_id": doc_id,
            "pages": pages,
            "json_result": raw_json,
        }
    finally:
        for t in temp_files:
            if t:
                try:
                    os.remove(t)
                except Exception:  # noqa: BLE001
                    pass
