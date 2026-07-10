"""
PaddleOCR-VL 客户端（转发到 Docker 容器）

链路：FastAPI /paddleocr-vl/* → Docker 容器（默认 localhost:8091）

按规格：请求 images=[{page, image_data(base64)}] + task_type + language；
容器返回 pages=[{page, width, height, text_blocks:[{type,bbox,content,confidence,latex,html}], markdown}]。
text_blocks 已接近统一元素结构，直接映射；crops 按 bbox 从原图裁剪（base64 内联）。
容器未启动 / 不可达时，返回明确错误，不抛异常中断主服务。
（注：容器真实端点待容器启动后核实，当前由 PADDLEOCR_VL_PARSE_PATH 配置。）
"""
import base64
import io
import json
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from PIL import Image

from . import config


def _build_url(path: str) -> str:
    base = config.PADDLEOCR_VL_URL.rstrip("/")
    return f"{base}{path}"


def _decode_dims(image_data: str) -> tuple:
    try:
        _, data = (image_data.split(",", 1) + [""])[:2] if "," in image_data else ("", image_data)
        pil = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        return pil, pil.size[0], pil.size[1]
    except Exception:  # noqa: BLE001
        return None, 0, 0


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
    url = _build_url(config.PADDLEOCR_VL_HEALTH_PATH)
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            body = json.loads(r.read().decode("utf-8"))
            body["ok"] = True
            return body
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"PaddleOCR-VL 容器不可达 ({url}): {e}"}


def parse(images: List[dict], task_type: str = "general", language: str = "ch", doc_id: str = "") -> dict:
    """images: [{page, image_data}]。返回统一 {ok, pages, error}。"""
    url = _build_url(config.PADDLEOCR_VL_PARSE_PATH)
    payload = {"images": images, "task_type": task_type, "language": language}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:500]}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"PaddleOCR-VL 转发失败 ({url}): {e}"}

    if not isinstance(resp, dict) or not resp.get("success", resp.get("ok", True)):
        return {"ok": False, "error": resp.get("error") or "PaddleOCR-VL 返回失败"}

    raw_pages = resp.get("pages", []) or []
    pages = []
    for idx, img in enumerate(images):
        page_no = img.get("page", idx + 1)
        image_data = img.get("image_data", "")
        pil, dw, dh = _decode_dims(image_data)

        # 匹配容器返回的对应页（按 page 字段）
        rp = None
        for p in raw_pages:
            if isinstance(p, dict) and p.get("page") == page_no:
                rp = p
                break
        if rp is None and raw_pages:
            rp = raw_pages[0]

        width = int(rp.get("width", dw)) if rp else dw
        height = int(rp.get("height", dh)) if rp else dh
        markdown = (rp.get("markdown", "") if rp else "") or ""
        blocks = (rp.get("text_blocks", []) or []) if rp else []

        elements = []
        crops = []
        for b_idx, blk in enumerate(blocks):
            if not isinstance(blk, dict):
                continue
            bbox = blk.get("bbox") or []
            if len(bbox) != 4:
                continue
            etype = blk.get("type", "paragraph")
            content = blk.get("content", "") or ""
            confidence = blk.get("confidence")
            confidence = float(confidence) if isinstance(confidence, (int, float)) else None
            el = {
                "type": etype,
                "bbox": [float(v) for v in bbox],
                "content": content,
                "confidence": confidence,
                "latex": blk.get("latex"),
                "html": blk.get("html"),
                "caption": None,
            }
            elements.append(el)
            if pil is not None:
                crop_b64 = _crop_to_base64(pil, bbox)
                if crop_b64:
                    crops.append({"filename": f"{doc_id}_page{page_no}_type{b_idx}.png", "base64": crop_b64})
        pages.append({
            "page": page_no,
            "width": width,
            "height": height,
            "markdown": markdown,
            "elements": elements,
            "crops": crops,
        })
    return {"ok": True, "pages": pages}
