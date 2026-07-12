"""
Unlimited-OCR 客户端（转发到 Docker 容器）

链路：FastAPI /unlimited-ocr/* → Docker 容器（默认 localhost:8090）

真实端点 /v1/ocr/multi：
- images 支持 路径 / URL / base64；实测需用 data: URI（裸 base64 会被当成文件路径）。
- enable_grounding=true 返回 bboxes（像素坐标 + text + type）→ 直接映射为 elements。
- crops 返回的 path 是容器内路径，代理不可达 → 统一按 bbox 从原图裁剪（base64 内联）。
容器未启动 / 不可达时，返回明确错误，不抛异常中断主服务。
"""
import base64
import io
import json
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
}

_lock = threading.Lock()  # urllib 全局锁，简单串行化


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
        bboxes = res.get("bboxes", []) or []
        elements = []
        crops = []
        for idx, box in enumerate(bboxes):
            bx = box.get("bbox", box.get("bbox_2d"))
            if not bx:
                bx = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
            if not bx or len(bx) != 4:
                continue
            etype = _label_map.get(box.get("type", "text"), "paragraph")
            content = box.get("text", "") or ""
            el = {
                "type": etype,
                "bbox": [float(v) for v in bx],
                "content": content,
                "confidence": None,
                "latex": content if etype == "formula" else None,
                "html": None,
                "caption": box.get("caption") if etype == "figure" else None,
            }
            elements.append(el)
            if pil is not None:
                crop_b64 = _crop_to_base64(pil, bx)
                if crop_b64:
                    crops.append({
                        "filename": f"{doc_id}_page{i+1}_type{idx}.png",
                        "base64": crop_b64,
                        "element_index": len(elements) - 1,   # 该 crop 对应元素的 elements[] 下标
                        "bbox": [float(v) for v in bx],
                    })
        pages.append({
            "page": i + 1,
            "width": width,
            "height": height,
            "markdown": markdown,
            "elements": elements,
            "crops": crops,
        })
    return {"ok": True, "pages": pages}
