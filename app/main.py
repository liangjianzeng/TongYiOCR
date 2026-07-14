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
import re
import json
import asyncio
import collections
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Request, HTTPException, File, UploadFile, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from . import config
from . import glmocr_client
from . import unlimited_ocr_client
from . import paddleocr_vl_client
from . import task_queue
from . import pdf_utils
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


# ===================== 请求日志（内存 ring buffer + SSE 实时推送） =====================
_log_lock = threading.Lock()
_request_logs = collections.deque(maxlen=500)
_log_seq = 0

# 日志时间统一使用北京时间（UTC+8），避免前端按 UTC 显示造成 8 小时偏差
BEIJING_TZ = timezone(timedelta(hours=8))

# 过滤探活 / 静态 / 文档 / 队列轮询等噪音，避免刷屏
_NOISE_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}

def _is_noise(path: str) -> bool:
    if path.startswith("/logs") or path.startswith("/static"):
        return True
    if path in _NOISE_PATHS or path == "/task/queue/info":
        return True
    return False

def _push_log(entry: dict):
    global _log_seq
    with _log_lock:
        _log_seq += 1
        entry["id"] = _log_seq
        _request_logs.append(entry)


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
                    f"{datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')} {client} {method} {path} -> {status} ({dur_ms:.0f}ms)\n"
                )
        except Exception:
            pass
        # 实时日志：推送到内存 buffer（过滤噪音路径，避免刷屏）
        if not _is_noise(path):
            _push_log({
                "ts": datetime.now(BEIJING_TZ).isoformat(timespec="seconds"),
                "client": client,
                "method": method,
                "path": path,
                "query": request.url.query,
                "status": status,
                "duration_ms": round(dur_ms, 1),
                "ua": request.headers.get("user-agent", "")[:100],
            })
        return response


app.add_middleware(AccessLogMiddleware)


# ===================== PDF 上传解析（PDF → 图片 → 引擎） =====================
def _attach_pdf_meta(result: dict, file_id: str) -> dict:
    """给 PDF 解析结果挂上「逐页懒加载背景图」所需元数据：

    - 顶层 pdf_file_id / pdf_page_url_template（前端可据此拼出任意页 URL）
    - 每页 pages[].page_image_url = /pdf_page/{file_id}/{page}?dpi=...

    背景图不再内联（避免几百页 base64 撑爆响应），改由前端的 /pdf_page 懒加载。
    """
    if not file_id:
        return result
    tpl = f"/pdf_page/{file_id}/{{page}}?dpi={pdf_utils.PDF_BG_DPI}"
    result["pdf_file_id"] = file_id
    result["pdf_page_url_template"] = tpl
    for p in (result.get("pages") or []):
        pg = int(p.get("page") or 0)
        if pg > 0:
            p["page_image_url"] = f"/pdf_page/{file_id}/{pg}?dpi={pdf_utils.PDF_BG_DPI}"
    return result


# ===================== GLM-OCR 子服务 =====================
glmocr_router = APIRouter(prefix="/glmocr", tags=["GLM-OCR"])


@glmocr_router.get("/health")
def glmocr_health():
    # 真实探测本机 llama.cpp VLM（8089），不可用则标红
    eng = glmocr_client.engine_health()
    sdk_ok = glmocr_client.glmocr_available()
    ok = bool(eng.get("ok")) and sdk_ok
    return {
        "ok": ok,
        "status": "ok" if ok else "error",
        "glmocr_available": sdk_ok,
        "llama_port": config.GLM_OCR_API_PORT,
        "engine": eng,
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


@glmocr_router.post("/parse_pdf", response_model=OCRParseResponse)
async def glmocr_parse_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form("pdf"),
    llama_port: int = Form(None),
):
    """PDF 上传 → 每页转图 → GLM-OCR 解析，返回统一 pages[]。"""
    if not config.GLM_OCR_ENABLED:
        return JSONResponse(status_code=503, content={"ok": False, "error": "GLM-OCR 未启用"})
    if not glmocr_client.glmocr_available():
        return JSONResponse(status_code=500, content={"ok": False, "error": f"glmocr SDK 未安装: {glmocr_client._GLMOCR_IMPORT_ERR}"})
    data = await file.read()
    try:
        file_id = pdf_utils.store_pdf(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF 转换失败: {e}"})
    try:
        ocr_images = pdf_utils.pdf_bytes_to_images(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF 转换失败: {e}"})
    if not ocr_images:
        return JSONResponse(status_code=400, content={"ok": False, "error": "PDF 无可用页面"})
    try:
        result = glmocr_client.parse(
            images=ocr_images,
            doc_id=doc_id,
            llama_host=None,
            llama_port=llama_port,
        )
        _attach_pdf_meta(result, file_id)
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


@unlimited_router.post("/parse_pdf", response_model=OCRParseResponse)
async def unlimited_parse_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form("pdf"),
    prompt: str = Form("请准确识别图片中的所有文字，保持原始排版"),
):
    """PDF 上传 → 每页转图 → Unlimited-OCR 解析，返回统一 pages[]。"""
    if not config.UNLIMITED_OCR_ENABLED:
        return JSONResponse(status_code=503, content={"ok": False, "error": "Unlimited-OCR 未启用"})
    data = await file.read()
    try:
        file_id = pdf_utils.store_pdf(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF 转换失败: {e}"})
    try:
        ocr_images = pdf_utils.pdf_bytes_to_images(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF 转换失败: {e}"})
    if not ocr_images:
        return JSONResponse(status_code=400, content={"ok": False, "error": "PDF 无可用页面"})
    result = unlimited_ocr_client.parse(images=ocr_images, prompt=prompt, doc_id=doc_id)
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=result)
    _attach_pdf_meta(result, file_id)
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


@paddle_router.post("/parse_pdf", response_model=OCRParseResponse)
async def paddle_parse_pdf(
    file: UploadFile = File(...),
    doc_id: str = Form("pdf"),
    task_type: str = Form("general"),
    language: str = Form("ch"),
):
    """PDF 上传 → 每页转图 → PaddleOCR-VL 解析，返回统一 pages[]。"""
    if not config.PADDLEOCR_VL_ENABLED:
        return JSONResponse(status_code=503, content={"ok": False, "error": "PaddleOCR-VL 未启用"})
    data = await file.read()
    try:
        file_id = pdf_utils.store_pdf(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF 转换失败: {e}"})
    try:
        ocr_images = pdf_utils.pdf_bytes_to_images(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF 转换失败: {e}"})
    if not ocr_images:
        return JSONResponse(status_code=400, content={"ok": False, "error": "PDF 无可用页面"})
    img_objs = [{"page": i + 1, "image_data": b64} for i, b64 in enumerate(ocr_images)]
    result = paddleocr_vl_client.parse(
        images=img_objs,
        task_type=task_type,
        language=language,
        doc_id=doc_id,
    )
    if not result.get("ok"):
        return JSONResponse(status_code=502, content=result)
    for p in (result.get("pages") or []):
        tb = p.pop("text_blocks", None)
        if tb is not None and "elements" not in p:
            p["elements"] = tb
    _attach_pdf_meta(result, file_id)
    return result


# ===================== PDF 单页懒加载（背景图） =====================
@app.get("/pdf_page/{file_id}/{page}")
async def pdf_page(file_id: str, page: int, dpi: int = pdf_utils.PDF_BG_DPI):
    """按 file_id + 页码（1 起）懒加载某一页的背景 PNG，供前端「排版还原」叠加。

    带磁盘缓存（pdf_utils.render_page_png）与浏览器缓存头，重复请求不重复渲染。
    """
    if not re.match(r"^[A-Za-z0-9]{8,64}$", file_id):
        raise HTTPException(status_code=400, detail="非法 file_id")
    if page < 1:
        raise HTTPException(status_code=400, detail="非法页码")
    try:
        png = pdf_utils.render_page_png(file_id, page, dpi)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"渲染失败: {e}")
    if png is None:
        raise HTTPException(status_code=404, detail="页面不存在")
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


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


@app.post("/task/submit_pdf", response_model=TaskSubmitResponse)
async def task_submit_pdf(
    file: UploadFile = File(...),
    engine: str = Form(...),
    doc_id: str = Form("pdf"),
    llama_port: int = Form(None),
    prompt: str = Form(None),
    task_type: str = Form(None),
    language: str = Form(None),
):
    """PDF 上传 → 每页转图 → 作为多页任务提交到队列（与 /task/submit 等价，但入参为 PDF 文件）。"""
    if engine not in ("glmocr", "unlimited-ocr", "paddleocr-vl"):
        raise HTTPException(status_code=400, detail=f"不支持的引擎: {engine}")
    data = await file.read()
    try:
        file_id = pdf_utils.store_pdf(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF 转换失败: {e}"})
    try:
        ocr_images = pdf_utils.pdf_bytes_to_images(data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"PDF 转换失败: {e}"})
    if not ocr_images:
        return JSONResponse(status_code=400, content={"ok": False, "error": "PDF 无可用页面"})
    pdf_page_url_template = f"/pdf_page/{file_id}/{{page}}?dpi={pdf_utils.PDF_BG_DPI}"
    try:
        return task_queue.submit_task(
            engine=engine,
            images=ocr_images,
            doc_id=doc_id,
            llama_port=llama_port,
            prompt=prompt,
            task_type=task_type,
            language=language,
            pdf_file_id=file_id,
            pdf_page_url_template=pdf_page_url_template,
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


@app.get("/logs")
def get_logs(limit: int = 100):
    limit = max(1, min(limit, 500))
    with _log_lock:
        items = list(_request_logs)[-limit:]
    return {"count": len(items), "logs": items}


@app.get("/logs/stream")
async def log_stream():
    async def gen():
        # 初始回填最近 30 条，避免打开面板时空白
        with _log_lock:
            backlog = list(_request_logs)[-30:]
        last_id = backlog[-1]["id"] if backlog else 0
        for e in backlog:
            yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
        while True:
            await asyncio.sleep(0.4)
            with _log_lock:
                new_items = [e for e in _request_logs if e["id"] > last_id]
            for e in new_items:
                last_id = e["id"]
                yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/", response_class=HTMLResponse)
def index():
    if STATIC_DIR.joinpath("index.html").exists():
        return HTMLResponse(STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8"))
    return HTMLResponse("<h1>彤熠OCR网关 (TongYiOCR) 运行中</h1><p>见 /docs 查看接口。</p>")


if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
