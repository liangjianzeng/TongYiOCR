"""PDF → 图片工具，供测试台 PDF 上传使用。

依赖 pymupdf（fitz），代理 .venv 已安装（1.25.x）。

两类能力：
1. pdf_bytes_to_images：把 PDF 每页渲染为高 DPI（默认 200）图，送 OCR 引擎识别。
2. 文件存储 + 单页懒加载：store_pdf 把上传的 PDF 落盘并返回 file_id；
   render_page_png(file_id, page, dpi) 按需渲染某一页（带磁盘缓存），
   供前端「排版还原」逐页懒加载背景，避免一次性内联几百张 base64。

上限 MAX_PDF_PAGES（默认 500）保护代理内存；超过即拒绝。
"""
import base64
import os
import re
import tempfile
import uuid

import fitz  # pymupdf

# 单 PDF 最多转页数（保护代理内存与浏览器渲染），支持 ≥200 页
MAX_PDF_PAGES = 500
# 排版还原背景图 DPI（低分辨率即可，仅作坐标叠加底图）
PDF_BG_DPI = 110
# file_id 白名单（防路径穿越）
_FILE_ID_RE = re.compile(r"^[A-Za-z0-9]{8,64}$")


def _store_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "tongyi_pdf")
    os.makedirs(d, exist_ok=True)
    return d


def _page_cache_dir(file_id: str) -> str:
    d = os.path.join(_store_dir(), file_id)
    os.makedirs(d, exist_ok=True)
    return d


def store_pdf(pdf_bytes: bytes) -> str:
    """把上传的 PDF 字节落盘，返回 file_id（uuid hex）。

    落盘后前端即可通过 /pdf_page/{file_id}/{page} 懒加载任意一页背景图。
    顺便做轻量清理：store 目录超过 80 个文件时，删除最旧的 PDF。
    """
    if not pdf_bytes or not pdf_bytes[:5].startswith(b"%PDF-"):
        raise ValueError("不是合法的 PDF 文件")
    fid = uuid.uuid4().hex
    pdf_path = os.path.join(_store_dir(), fid + ".pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    _page_cache_dir(fid)  # 预建缓存子目录
    _maybe_cleanup()
    return fid


def _maybe_cleanup(keep: int = 80) -> None:
    d = _store_dir()
    try:
        pdfs = [f for f in os.listdir(d) if f.endswith(".pdf")]
        if len(pdfs) <= keep:
            return
        pdfs.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)))
        for f in pdfs[: len(pdfs) - keep]:
            try:
                os.remove(os.path.join(d, f))
                cdir = os.path.join(d, f[:-4])
                if os.path.isdir(cdir):
                    import shutil

                    shutil.rmtree(cdir, ignore_errors=True)
            except OSError:
                pass
    except OSError:
        pass


def get_page_count(file_id: str) -> int:
    if not _FILE_ID_RE.match(file_id):
        return 0
    path = os.path.join(_store_dir(), file_id + ".pdf")
    if not os.path.isfile(path):
        return 0
    try:
        doc = fitz.open(path)
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return 0


def render_page_png(file_id: str, page: int, dpi: int = PDF_BG_DPI) -> bytes | None:
    """渲染 PDF 的某一页（1 起）为 PNG 字节；带磁盘缓存。

    返回 None 表示 file_id 不存在或页码越界。
    """
    if not _FILE_ID_RE.match(file_id):
        return None
    path = os.path.join(_store_dir(), file_id + ".pdf")
    if not os.path.isfile(path):
        return None

    cache_dir = _page_cache_dir(file_id)
    cache_path = os.path.join(cache_dir, f"{page}.png")
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            return f.read()

    try:
        doc = fitz.open(path)
    except Exception:
        return None
    try:
        if page < 1 or page > doc.page_count:
            return None
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = doc[page - 1].get_pixmap(matrix=mat)
    finally:
        doc.close()

    png = pix.tobytes("png")
    try:
        with open(cache_path, "wb") as f:
            f.write(png)
    except OSError:
        pass
    return png


def pdf_bytes_to_images(
    pdf_bytes: bytes,
    ocr_dpi: int = 200,
    max_pages: int = MAX_PDF_PAGES,
) -> list:
    """把 PDF 字节转成 OCR 用高 DPI 图列表（data URI）。

    返回 [] 表示 PDF 无可用页面。页数超过 max_pages 抛 ValueError。
    （背景图不再在此内联，改由 /pdf_page 懒加载，故此处只返回 OCR 图。）
    """
    if not pdf_bytes:
        raise ValueError("PDF 内容为空")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        n = doc.page_count
        if n == 0:
            return []
        if n > max_pages:
            raise ValueError(f"PDF 页数({n})超过上限({max_pages})，请拆分后上传")
        zoom = ocr_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        out = []
        for i in range(n):
            pix = doc[i].get_pixmap(matrix=mat)
            b64 = base64.standard_b64encode(pix.tobytes("png")).decode("ascii")
            out.append(f"data:image/png;base64,{b64}")
        return out
    finally:
        doc.close()
