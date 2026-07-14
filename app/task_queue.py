"""
任务队列子系统（显存限制解决方案）

设计：
- 单一全局队列 + 单后台 worker 串行处理（同一时间只有一个任务在执行）。
- engine_lock 保证引擎独占（切换引擎时标记 loading）。
- 逐页调用引擎客户端，实时更新 current_page / progress。
- 无状态：结果仅存于内存（task_status），不持久化。

状态机：queued → loading → processing → completed | failed | cancelled
"""
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from . import config
from . import glmocr_client
from . import unlimited_ocr_client
from . import paddleocr_vl_client


# ---------- 全局状态 ----------
task_queue: asyncio.Queue = asyncio.Queue(maxsize=config.MAX_QUEUE_SIZE)
task_status: Dict[str, Dict[str, Any]] = {}
current_task_id: Optional[str] = None
engine_lock = asyncio.Lock()
_current_engine: Optional[str] = None
_worker_task: Optional[asyncio.Task] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_engine_one(engine: str, image: str, task: Dict[str, Any], page_index: int) -> Dict[str, Any]:
    """对单页调用对应引擎客户端，返回统一 {ok, pages}。"""
    doc_id = task.get("doc_id", "")
    if engine == "glmocr":
        return glmocr_client.parse(
            images=[image],
            doc_id=doc_id,
            llama_port=task.get("llama_port"),
        )
    if engine == "unlimited-ocr":
        return unlimited_ocr_client.parse(
            images=[image],
            prompt=task.get("prompt") or config.UNLIMITED_OCR_PROMPT,
            doc_id=doc_id,
        )
    if engine == "paddleocr-vl":
        return paddleocr_vl_client.parse(
            images=[{"page": page_index + 1, "image_data": image}],
            task_type=task.get("task_type") or "general",
            language=task.get("language") or "ch",
            doc_id=doc_id,
        )
    raise ValueError(f"未知引擎: {engine}")


async def _process_task(task: Dict[str, Any]) -> None:
    global current_task_id, _current_engine
    tid = task["task_id"]
    engine = task["engine"]
    images = task["images"]
    total = len(images)

    current_task_id = tid
    st = task_status[tid]
    st["status"] = "loading"
    st["started_at"] = _now()

    async with engine_lock:
        # 引擎切换：标记 loading（真实环境下无法卸载 VLM，仅作状态语义）
        if _current_engine is not None and _current_engine != engine:
            st["status"] = "loading"
            await asyncio.sleep(0)  # 让出，便于状态被读取
        st["status"] = "processing"
        _current_engine = engine

        pages: list = []
        page_errors: Dict[int, str] = {}
        try:
            for i, img in enumerate(images):
                try:
                    res = await asyncio.wait_for(
                        asyncio.to_thread(_call_engine_one, engine, img, task, i),
                        timeout=config.REQUEST_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    # 单页超时：降级跳过，不拖垮整批（已成功页保留）
                    page_errors[i + 1] = f"引擎 {engine} 单页调用超时（>{config.REQUEST_TIMEOUT}s）"
                    st["current_page"] = i + 1
                    st["progress"] = int((i + 1) / total * 100) if total else 100
                    continue
                except Exception as e:  # noqa: BLE001
                    page_errors[i + 1] = f"单页异常: {e}"
                    st["current_page"] = i + 1
                    st["progress"] = int((i + 1) / total * 100) if total else 100
                    continue
                if not res.get("ok"):
                    # 引擎返回失败：记录该页错误，继续下一页
                    page_errors[i + 1] = res.get("error") or "引擎返回失败"
                    st["current_page"] = i + 1
                    st["progress"] = int((i + 1) / total * 100) if total else 100
                    continue
                for p in res.get("pages", []):
                    p["page"] = i + 1  # 重新编号为全局页码
                    pages.append(p)
                st["current_page"] = i + 1
                st["progress"] = int((i + 1) / total * 100) if total else 100
            # 整批完成：允许部分页失败，不丢弃已成功的页
            st["status"] = "completed"
            st["completed_at"] = _now()
            st["result"] = {
                "pages": pages,
                "total_pages": total,
                "page_errors": page_errors,
            }
            # PDF 任务：把 file_id / 背景图 URL 模板挂到结果，供前端懒加载排版还原背景
            if task.get("pdf_file_id"):
                res = st["result"]
                res["pdf_file_id"] = task["pdf_file_id"]
                res["pdf_page_url_template"] = task.get("pdf_page_url_template")
                for p in pages:
                    pg = int(p.get("page") or 0)
                    if pg > 0 and task.get("pdf_page_url_template"):
                        p["page_image_url"] = task["pdf_page_url_template"].replace(
                            "{page}", str(pg)
                        )
        except Exception as e:  # noqa: BLE001
            # 仅当循环本身（非单页）崩溃才整批失败
            st["status"] = "failed"
            st["error"] = str(e)
            st["completed_at"] = _now()
        finally:
            current_task_id = None


async def _worker() -> None:
    while True:
        task = await task_queue.get()
        try:
            await _process_task(task)
        finally:
            task_queue.task_done()


def start_worker() -> None:
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker())


def submit_task(engine: str, images: list, doc_id: str = "", llama_port=None,
                prompt=None, task_type=None, language=None,
                pdf_file_id: str = None, pdf_page_url_template: str = None) -> Dict[str, Any]:
    """提交任务，返回 {task_id, status, position, estimated_wait_seconds}。"""
    if task_queue.full():
        raise RuntimeError(f"队列已满（最大 {config.MAX_QUEUE_SIZE}）")
    task_id = str(uuid.uuid4())
    now = _now()
    task_status[task_id] = {
        "task_id": task_id,
        "engine": engine,
        "status": "queued",
        "progress": 0,
        "current_page": 0,
        "total_pages": len(images),
        "submitted_at": now,
        "started_at": None,
        "completed_at": None,
        "position": None,
        "error": None,
        "result": None,
    }
    task = {
        "task_id": task_id,
        "engine": engine,
        "images": images,
        "doc_id": doc_id,
        "llama_port": llama_port,
        "prompt": prompt,
        "task_type": task_type,
        "language": language,
        "pdf_file_id": pdf_file_id,
        "pdf_page_url_template": pdf_page_url_template,
    }
    task_queue.put_nowait(task)
    position = _position_of(task_id)
    est = position * config.ESTIMATED_TASK_SECONDS
    return {
        "task_id": task_id,
        "status": "queued",
        "position": position,
        "estimated_wait_seconds": est,
    }


def _position_of(task_id: str) -> int:
    """计算任务在队列中的顺位（1 起）。"""
    order = 1
    for tid, st in task_status.items():
        if st["status"] in ("queued", "processing") and tid != task_id:
            # submitted 更早的排前面
            if st.get("submitted_at", "") <= task_status[task_id].get("submitted_at", ""):
                order += 1
    return order


def get_status(task_id: str) -> Optional[Dict[str, Any]]:
    st = task_status.get(task_id)
    if st is None:
        return None
    st = dict(st)
    if st["status"] == "queued":
        st["position"] = _position_of(task_id)
    # 不向外暴露 result 大对象
    st.pop("result", None)
    return st


def get_result(task_id: str) -> Optional[Dict[str, Any]]:
    st = task_status.get(task_id)
    if st is None:
        return None
    if st["status"] == "failed":
        return {"task_id": task_id, "status": "failed", "error": st.get("error")}
    if st["status"] != "completed":
        return {"task_id": task_id, "status": st["status"]}
    res = st.get("result") or {}
    return {
        "task_id": task_id,
        "status": "completed",
        "engine": st.get("engine"),
        "pages": res.get("pages", []),
        "total_pages": res.get("total_pages"),
        "page_errors": res.get("page_errors", {}),
        "pdf_file_id": res.get("pdf_file_id"),
        "pdf_page_url_template": res.get("pdf_page_url_template"),
        "completed_at": st.get("completed_at"),
        "error": None,
    }


def cancel_task(task_id: str) -> Optional[Dict[str, Any]]:
    st = task_status.get(task_id)
    if st is None:
        return None
    if st["status"] in ("completed", "failed", "cancelled"):
        return {"task_id": task_id, "status": st["status"]}
    st["status"] = "cancelled"
    return {"task_id": task_id, "status": "cancelled"}


def queue_info() -> Dict[str, Any]:
    return {
        "queue_size": task_queue.qsize(),
        "current_task_id": current_task_id,
        "total_tasks": len(task_status),
        "completed_tasks": sum(1 for s in task_status.values() if s["status"] == "completed"),
        "failed_tasks": sum(1 for s in task_status.values() if s["status"] == "failed"),
    }
