"""
PaddleOCR-VL 引擎服务（方案A容器内，监听 :8091）

职责：
- 直接调用 PaddleOCRVL pipeline（PP-DocLayoutV3 布局 + VLM 识别，均 GPU）
- 提供 /health 和 /parse 两个端点，输出格式对齐 paddleocr_vl_client 期望

不依赖 app/ 模块（那是统一代理的代码），纯引擎。
"""
import os
import io
import json
import base64
import threading
import traceback
import time
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List

from PIL import Image
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ---------- 配置（全部从环境变量注入） ----------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("ENGINE_PORT", "8091"))
ENGINE = os.getenv("PADDLE_OCR_ENGINE", "paddle")
DEVICE = os.getenv("PADDLE_OCR_DEVICE", "cpu")
PIPELINE_VERSION = os.getenv("PADDLE_OCR_PIPELINE_VERSION", "v1.6")
VL_SERVER_URL = os.getenv("PADDLE_OCR_VL_SERVER_URL", "http://127.0.0.1:8118/v1")
VL_MODEL = os.getenv("PADDLE_OCR_VL_MODEL", "PaddleOCR-VL-1.6-0.9B")
VL_API_KEY = os.getenv("PADDLE_OCR_VL_API_KEY", "EMPTY")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# PaddleX 缓存根（挂载 /home/paddleocr/.paddlex）
PDX_CACHE = os.getenv("PADDLE_PDX_CACHE_HOME", "/home/paddleocr/.paddlex")
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", PDX_CACHE)

# ---------- 全局状态 ----------
_LOCK = threading.Lock()
_PIPELINE = None
_MODEL_READY = False
_MODEL_ERROR: Optional[str] = None

# ---------- 标签映射 ----------
LABEL_MAP: Dict[str, Dict[str, str]] = {
    "title":    {"type": "title",    "layout_label": "title"},
    "text":     {"type": "paragraph","layout_label": "body"},
    "subtitle": {"type": "subtitle", "layout_label": "subtitle"},
    "formula":  {"type": "formula",  "layout_label": "body"},
    "table":    {"type": "table",    "layout_label": "body"},
    "image":    {"type": "figure",   "layout_label": "body"},
    "figure":   {"type": "figure",   "layout_label": "body"},
    "header":   {"type": "header",   "layout_label": "header"},
    "footer":   {"type": "footer",   "layout_label": "footer"},
    "list":     {"type": "paragraph","layout_label": "body"},
    "code":     {"type": "paragraph","layout_label": "body"},
    "reference":{"type": "paragraph","layout_label": "body"},
    "abstract": {"type": "paragraph","layout_label": "body"},
    "chart":    {"type": "figure",   "layout_label": "body"},
    "seal":     {"type": "figure",   "layout_label": "body"},
    "equation": {"type": "formula",  "layout_label": "body"},
}


def _build_pipeline():
    """构造 PaddleOCRVL pipeline（布局检测走 CPU，VLM 识别走 vLLM GPU）"""
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "true")
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_DISABLE_CERT_VERIFICATION", "true")
    from paddleocr import PaddleOCRVL

    kwargs: Dict[str, Any] = {"pipeline_version": PIPELINE_VERSION}
    # 强制显式传 device（cpu 或 gpu），避免 paddle 自动检测到 GPU 但无 sm_120 kernel
    kwargs["device"] = DEVICE

    pipeline = PaddleOCRVL(
        engine="paddle",
        vl_rec_backend="vllm-server",
        vl_rec_server_url=VL_SERVER_URL,
        vl_rec_model_name=VL_MODEL,
        vl_rec_api_key=VL_API_KEY,
        **kwargs,
    )
    return pipeline


def ensure_model_loaded():
    global _PIPELINE, _MODEL_READY, _MODEL_ERROR
    if _MODEL_READY:
        return
    if _MODEL_ERROR:
        raise RuntimeError(_MODEL_ERROR)
    with _LOCK:
        if _PIPELINE is not None:
            return
        try:
            _PIPELINE = _build_pipeline()
            _MODEL_READY = True
            print(f"[engine] model loaded OK: engine={ENGINE} device={DEVICE} vllm={VL_SERVER_URL}")
        except Exception as e:
            _MODEL_ERROR = f"模型加载失败: {e}\n{traceback.format_exc()}"
            print(f"[engine] FATAL: {_MODEL_ERROR}", flush=True)
            raise


app = FastAPI(title="PaddleOCR-VL Engine", version="1.0.0")


# ---------- 请求模型 ----------
class ImageSpec(BaseModel):
    page: int = 1
    image_data: str = Field(..., description="base64 data URI (data:image/xxx;base64,...)")


class ParseRequest(BaseModel):
    images: List[ImageSpec] = Field(..., description="图片列表")
    task_type: str = Field("general", description="general|table|formula|layout")
    language: str = Field("ch", description="ch|en|japanese")


# ---------- 端点 ----------
@app.get("/health")
def health():
    return {
        "status": "ok" if _MODEL_READY else ("error" if _MODEL_ERROR else "loading"),
        "model_ready": _MODEL_READY,
        "engine": ENGINE,
        "device": DEVICE,
        "pipeline_version": PIPELINE_VERSION,
        "vllm_url": VL_SERVER_URL,
        "error": _MODEL_ERROR,
    }


@app.post("/parse")
def parse(body: ParseRequest):
    """接收 {images: [{page, image_data(base64 data URI)}], task_type, language} → 返回统一结果"""
    if not _MODEL_READY:
        raise HTTPException(status_code=503, detail=f"引擎未就绪: {_MODEL_ERROR or '模型未加载'}")

    pages_out = []

    for img_spec in body.images:
        page_no = img_spec.page
        image_data = img_spec.image_data

        # 解码 base64
        raw_bytes = None
        if isinstance(image_data, str) and image_data:
            if "," in image_data:
                _, b64 = image_data.split(",", 1)
            else:
                b64 = image_data
            try:
                raw_bytes = base64.b64decode(b64)
            except Exception:
                pass

        if raw_bytes is None:
            pages_out.append({"page": page_no, "width": 0, "height": 0,
                              "markdown": "", "text_blocks": [], "error": "无效图片数据"})
            continue

        pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        width, height = pil.size

        # 调 PaddleOCRVL pipeline（必须转 numpy，predict() 不接受 PIL）
        try:
            result_list = _PIPELINE.predict(np.array(pil))
        except Exception as e:
            pages_out.append({"page": page_no, "width": width, "height": height,
                              "markdown": "", "text_blocks": [], "error": f"识别失败: {e}"})
            continue

        # predict() 返回 list[PaddleOCRVLResult]，取第一个
        if not isinstance(result_list, list) or len(result_list) == 0:
            pages_out.append({"page": page_no, "width": width, "height": height,
                              "markdown": "", "text_blocks": [], "error": "predict 返回空列表"})
            continue

        r0 = result_list[0]
        res = getattr(r0, "json", {}).get("res", {})
        parsing_list = res.get("parsing_res_list", []) or []
        md_info = getattr(r0, "markdown", {})
        markdown = md_info.get("markdown_texts", "") if isinstance(md_info, dict) else ""
        if not isinstance(markdown, str):
            markdown = str(markdown) if markdown else ""

        text_blocks = []
        for blk in parsing_list:
            if not isinstance(blk, dict):
                continue
            label = blk.get("block_label", "text")
            mapping = LABEL_MAP.get(label, {"type": "paragraph", "layout_label": "body"})
            bbox = blk.get("block_bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            tb = {
                "id": blk.get("block_order", len(text_blocks)),
                "type": mapping["type"],
                "content": blk.get("block_content", "") or "",
                "bbox": [float(v) for v in bbox[:4]],
                "confidence": float(blk.get("confidence", 0)) if blk.get("confidence") is not None else None,
                "layout_label": mapping["layout_label"],
                "latex": blk.get("latex"),
                "html": blk.get("html"),
            }
            text_blocks.append(tb)

        pages_out.append({
            "page": page_no,
            "width": width,
            "height": height,
            "markdown": markdown,
            "text_blocks": text_blocks,
        })

    return JSONResponse({"success": True, "pages": pages_out, "engine": "paddleocr-vl-1.6"})


@app.on_event("startup")
def _startup():
    # 等待 vLLM 就绪（轮询最多 120s）
    vllm_url = f"{VL_SERVER_URL.rstrip('/')}/models"
    print(f"[engine] waiting for vLLM at {vllm_url} ...", flush=True)
    for i in range(60):
        try:
            req = urllib.request.Request(vllm_url)
            req.add_header("Authorization", f"Bearer {VL_API_KEY}")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    print("[engine] vLLM ready", flush=True)
                    break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("[engine] WARNING: vLLM not ready after 120s, proceeding anyway", flush=True)

    # 加载 PaddleOCRVL（含 PP-DocLayoutV3 布局检测，走 GPU）
    try:
        ensure_model_loaded()
    except Exception:
        print(f"[engine] model load failed, will retry on first request\n{_MODEL_ERROR}", flush=True)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
