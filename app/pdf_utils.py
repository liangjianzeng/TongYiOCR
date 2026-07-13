"""PDF → 图片（base64 PNG）工具，供测试台 PDF 上传使用。

依赖 pymupdf（fitz），代理 .venv 已安装（1.25.x）。
把 PDF 每页渲染为两张图：
  - ocr 图：高 DPI（默认 200），送 OCR 引擎识别；
  - bg  图：低 DPI（默认 110），仅用于前端「排版还原」背景叠加，省流量。
返回两个 base64 data URI 列表（按页顺序），调用方按顺序送引擎即可。
"""
import base64

import fitz  # pymupdf

# 单 PDF 最多转页数，保护代理内存与浏览器渲染
MAX_PDF_PAGES = 100
# 超过该页数则不在响应里内联背景图（避免响应报文过大），排版还原退化为「仅还原层」
MAX_INLINE_BG_PAGES = 20


def pdf_bytes_to_images(
    pdf_bytes: bytes,
    ocr_dpi: int = 200,
    bg_dpi: int = 110,
    max_pages: int = MAX_PDF_PAGES,
) -> tuple:
    """把 PDF 字节转成 (ocr_images, bg_images) 两个 data URI 列表。

    返回 ([], []) 表示 PDF 无可用页面。页数超过 max_pages 抛 ValueError。
    """
    if not pdf_bytes:
        raise ValueError("PDF 内容为空")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        n = doc.page_count
        if n == 0:
            return [], []
        if n > max_pages:
            raise ValueError(f"PDF 页数({n})超过上限({max_pages})，请拆分后上传")
        ocr_zoom = ocr_dpi / 72.0
        bg_zoom = bg_dpi / 72.0
        ocr_mat = fitz.Matrix(ocr_zoom, ocr_zoom)
        bg_mat = fitz.Matrix(bg_zoom, bg_zoom)
        ocr_images, bg_images = [], []
        for i in range(n):
            page = doc[i]
            opix = page.get_pixmap(matrix=ocr_mat)
            ob64 = base64.standard_b64encode(opix.tobytes("png")).decode("ascii")
            ocr_images.append(f"data:image/png;base64,{ob64}")
            # 页数过多时不生成背景图，省内存/流量
            if n <= MAX_INLINE_BG_PAGES:
                bpix = page.get_pixmap(matrix=bg_mat)
                bb64 = base64.standard_b64encode(bpix.tobytes("png")).decode("ascii")
                bg_images.append(f"data:image/png;base64,{bb64}")
            else:
                bg_images.append("")
        return ocr_images, bg_images
    finally:
        doc.close()
