"""
彤熠OCR网关（TongYiOCR）—— 统一 OCR 代理 FastAPI 入口（监听 :8088）

按路径前缀路由到三个后端 OCR 引擎（纯转发代理）：
  /glmocr/*        → llama.cpp VLM（本机 localhost:8089），经 glmocr SDK
  /unlimited-ocr/* → Docker 容器（默认 localhost:8090）
  /paddleocr-vl/*  → Docker 容器（默认 localhost:8091）

统一输出：pages[{page,width,height,markdown,elements[],crops[]}]（crops 为 base64，无状态）。

任务队列（显存限制解决方案）：
  /task/submit  POST  提交异步任务（返回 task_id）
  /task/{id}/status  GET  查询状态/进度
  /task/{id}/result  GET  获取结果
  /task/{id}/cancel  POST 取消
  /task/queue/info   GET  队列概览
"""
import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from . import config
from . import glmocr_client
from . import unlimited_ocr_client
from . import paddleocr_vl_client
from . import task_queue
from .schemas import (
    GlmOcrParseRequest, UnlimitedOcrRequest, PaddleOcrVlRequest,
    OCRParseResponse,
    TaskSubmitRequest, TaskSubmitResponse, TaskStatusResponse,
    TaskResultResponse, TaskCancelResponse, QueueInfoResponse,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    task_queue.start_worker()
    yield
    # shutdown (no-op)


app = FastAPI(
    title="彤熠OCR网关 (TongYiOCR)",
    description="GLM-OCR / Unlimited-OCR / PaddleOCR-VL 三引擎统一代理，按路径前缀路由转发；含任务队列。",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== 访问日志（写入文件，便于远程排障） =====================
class AccessLogMiddleware(BaseHTTPMiddleware):
    _log_path = str(BASE_DIR / ".workbuddy" / "proxy_access.log")

    async def dispatch(self, request, call_next):
        import time as _t
        client = request.client.host if request.client else "-"
        method = request.method
        path = request.url.path
        start = _t.time()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        dur_ms = (_t.time() - start) * 1000
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{_t.strftime('%Y-%m-%d %H:%M:%S')} {client} {method} {path} -> {status} ({dur_ms:.0f}ms)\n"
                )
        except Exception:
            pass
        return response


app.add_middleware(AccessLogMiddleware)


# ===================== GLM-OCR 子服务 =====================
glmocr_router = APIRouter(prefix="/glmocr", tags=["GLM-OCR"])


@glmocr_router.get("/health")
def glmocr_health():
    return {
        "ok": True,
        "status": "ok",
        "glmocr_available": glmocr_client.glmocr_available(),
        "llama_port": config.GLM_OCR_API_PORT,
        "engine": glmocr_client.engine_health(),
    }


@glmocr_router.post("/parse", response_model=OCRParseResponse)
def glmocr_parse(req: GlmOcrParseRequest):
    if not config.GLM_OCR_ENABLED:
        return JSONResponse(status_code=503, content={"ok": False, "error": "GLM-OCR 未启用"})
    if not glmocr_client.glmocr_available():
        return JSONResponse(status_code=500, content={"ok": False, "error": f"glmocr SDK 未安装: {glmocr_client._GLMOCR_IMPORT_ERR}"})
    try:
        result = glmocr_client.parse(
            images=req.images,
            image_paths=req.image_paths,
            doc_id=req.doc_id,
            llama_host=req.llama_host,
            llama_port=req.llama_port,
        )
        return result
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"ok": False, "error": f"解析异常: {e}"})


# ===================== Unlimited-OCR 子服务 =====================
unlimited_router = APIRouter(prefix="/unlimited-ocr", tags=["Unlimited-OCR"])


@unlimited_router.get("/health")
def unlimited_health():
    if not config.UNLIMITED_OCR_ENABLED:
        return {"ok": False, "status": "disabled"}
    return unlimited_ocr_client.health()


@unlimited_router.post("/parse", response_model=OCRParseResponse)
def unlimited_parse(req: UnlimitedOcrRequest):
    if not config.UNLIMITED_OCR_ENABLED:
        return JSONResponse(status_code=503, content={"ok": False, "error": "Unlimited-OCR 未启用"})
    result = unlimited_ocr_client.parse(images=req.images, prompt=req.prompt, doc_id=req.doc_id)
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=result)
    return result


# ===================== PaddleOCR-VL 子服务 =====================
paddle_router = APIRouter(prefix="/paddleocr-vl", tags=["PaddleOCR-VL"])


@paddle_router.get("/health")
def paddle_health():
    if not config.PADDLEOCR_VL_ENABLED:
        return {"ok": False, "status": "disabled"}
    return paddleocr_vl_client.health()


@paddle_router.post("/parse", response_model=OCRParseResponse)
def paddle_parse(req: PaddleOcrVlRequest):
    if not config.PADDLEOCR_VL_ENABLED:
        return JSONResponse(status_code=503, content={"ok": False, "error": "PaddleOCR-VL 未启用"})
    result = paddleocr_vl_client.parse(
        images=[img.model_dump() for img in req.images],
        task_type=req.task_type,
        language=req.language,
        doc_id=req.doc_id,
    )
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=result)

    # 引擎返回 text_blocks，统一 schema 叫 elements —— 做映射对齐
    for p in (result.get("pages") or []):
        tb = p.pop("text_blocks", None)
        if tb is not None and "elements" not in p:
            p["elements"] = tb

    return result


# ===================== 任务队列 =====================
@app.post("/task/submit", response_model=TaskSubmitResponse)
def task_submit(req: TaskSubmitRequest):
    try:
        return task_queue.submit_task(
            engine=req.engine,
            images=req.images,
            doc_id=req.doc_id,
            llama_port=req.llama_port,
            prompt=req.prompt,
            task_type=req.task_type,
            language=req.language,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))


@app.get("/task/{task_id}/status", response_model=TaskStatusResponse)
def task_status(task_id: str):
    st = task_queue.get_status(task_id)
    if st is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return st


@app.get("/task/{task_id}/result", response_model=TaskResultResponse)
def task_result(task_id: str):
    st = task_queue.get_result(task_id)
    if st is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if st["status"] not in ("completed", "failed"):
        raise HTTPException(status_code=400, detail=f"任务尚未完成（当前状态: {st['status']}）")
    return st


@app.post("/task/{task_id}/cancel", response_model=TaskCancelResponse)
def task_cancel(task_id: str):
    st = task_queue.cancel_task(task_id)
    if st is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return st


@app.get("/task/queue/info", response_model=QueueInfoResponse)
def task_queue_info():
    return task_queue.queue_info()


# ===================== 路由注册 =====================
app.include_router(glmocr_router)
app.include_router(unlimited_router)
app.include_router(paddle_router)


# ===================== 聚合健康检查 =====================
@app.get("/health")
def health():
    try:
        return {
            "status": "ok",
            "service": "tongyi-ocr",
            "version": "1.0.0",
            "host": config.HOST,
            "port": config.PORT,
            "engines": {
                "glmocr": {
                    "enabled": config.GLM_OCR_ENABLED,
                    "available": glmocr_client.glmocr_available(),
                    "api_host": config.GLM_OCR_API_HOST,
                    "api_port": config.GLM_OCR_API_PORT,
                    "engine": glmocr_client.engine_health(),
                },
                "unlimited_ocr": {
                    "enabled": config.UNLIMITED_OCR_ENABLED,
                    "url": config.UNLIMITED_OCR_URL,
                    "health": unlimited_ocr_client.health() if config.UNLIMITED_OCR_ENABLED else {"ok": False, "status": "disabled"},
                },
                "paddleocr_vl": {
                    "enabled": config.PADDLEOCR_VL_ENABLED,
                    "url": config.PADDLEOCR_VL_URL,
                    "health": paddleocr_vl_client.health() if config.PADDLEOCR_VL_ENABLED else {"ok": False, "status": "disabled"},
                },
            },
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "error",
            "service": "tongyi-ocr",
            "version": "1.0.0",
            "error": str(e),
        }


@app.get("/", response_class=HTMLResponse)
def index():
    if STATIC_DIR.joinpath("index.html").exists():
        return HTMLResponse(STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8"))
    return HTMLResponse("<h1>彤熠OCR网关 (TongYiOCR) 运行中</h1><p>见 /docs 查看接口。</p>")


if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
