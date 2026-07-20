"""
彤熠OCR网关（TongYiOCR）—— 统一 OCR 代理 请求 / 响应数据模型

对齐 OCR 远程服务接口需求：
- 统一输出：pages[{page,width,height,markdown,elements[],crops[]}]
- crops 以 base64 内联（无状态，不持久化）
- 三个引擎共用同一套统一响应结构（GLM-OCR 额外带 json_result 原文）

坐标系统：bbox = [x1, y1, x2, y2]，像素，基于页面原始尺寸，左上角原点。
"""
from typing import List, Optional, Any, Literal
from pydantic import BaseModel, Field


# ============================================================
# 统一输出结构（三引擎一致）
# ============================================================
class OCRCrop(BaseModel):
    """裁剪图（base64 内联，PNG）"""
    filename: str = Field(..., description="裁剪图文件名，命名 {doc_id}_page{page}_type{index}.png")
    base64: str = Field(..., description="data:image/png;base64,... 格式")
    element_index: Optional[int] = Field(None, description="该裁剪图对应元素在 elements[] 中的下标（用于前端排版还原匹配）")
    bbox: Optional[List[float]] = Field(None, description="[x1,y1,x2,y2] 像素坐标（与对应元素 bbox 一致）")


class OCRElement(BaseModel):
    """结构化元素"""
    type: str = Field(..., description="title|subtitle|paragraph|formula|table|figure|header|footer")
    bbox: List[float] = Field(..., description="[x1,y1,x2,y2] 像素坐标（原始页面尺寸）")
    content: str = Field("", description="识别文本")
    confidence: Optional[float] = Field(None, description="置信度 0-1")
    latex: Optional[str] = Field(None, description="公式 LaTeX（formula 类型）")
    html: Optional[str] = Field(None, description="表格 HTML（table 类型）")
    caption: Optional[str] = Field(None, description="图片说明（figure 类型）")
    # 代理层版面后处理（layout_postprocess）统一补充的字段，用于前端「结构化视图」
    reading_order: Optional[int] = Field(None, description="阅读顺序：按中心y聚类视觉行→行内按x排序（代理层统一分配）")
    block_id: Optional[int] = Field(None, description="逻辑块编号：同一视觉行/选项组共享，前端结构化分组用")
    is_option: Optional[bool] = Field(None, description="是否多选题选项（A./B./C./D.），前端选项块用 grid 成组")


class OCRPage(BaseModel):
    """单页结果"""
    page: int
    width: int = Field(0, description="页面原始宽度(px)")
    height: int = Field(0, description="页面原始高度(px)")
    markdown: str = Field("", description="完整 Markdown 文本")
    elements: List[OCRElement] = Field(default_factory=list, description="结构化元素列表")
    crops: List[OCRCrop] = Field(default_factory=list, description="裁剪图列表（base64）")
    page_image: Optional[str] = Field(None, description="该页原图 base64（单图上传时由前端用原图兜底，PDF 已改为懒加载 url）")
    page_image_url: Optional[str] = Field(None, description="该页原图懒加载地址（PDF 上传时由 /pdf_page/{file_id}/{page} 提供，供排版还原背景）")


class OCRParseResponse(BaseModel):
    """统一解析响应（三引擎共用）"""
    ok: bool
    pages: List[OCRPage] = Field(default_factory=list)
    error: Optional[str] = None
    engine: Optional[str] = None
    # GLM-OCR 额外保留原始 json_result（结构兼容）
    json_result: Optional[Any] = None
    # PDF 上传：逐页懒加载背景图所需元数据
    pdf_file_id: Optional[str] = Field(None, description="上传 PDF 的 file_id（/pdf_page 懒加载用）")
    pdf_page_url_template: Optional[str] = Field(None, description="逐页背景图 URL 模板，含 {page} 占位")


# ============================================================
# 同步解析请求（/glmocr/parse、/unlimited-ocr/parse、/paddleocr-vl/parse）
# ============================================================
class GlmOcrParseRequest(BaseModel):
    # 方式 1：base64 data URI 列表（远程调用方常用）
    images: Optional[List[str]] = Field(None, description="base64 图片列表（支持 data:image/xxx;base64, 前缀）")
    # 方式 2：兼容旧格式，本地文件路径列表
    image_paths: Optional[List[str]] = Field(None, description="本地图片路径列表（兼容旧格式）")
    doc_id: str = Field("", description="文档标识，用于命名 crops")
    # 可选：覆盖默认 GLM-OCR 引擎地址（默认取 config.GLM_OCR_API_HOST:PORT）
    llama_host: Optional[str] = Field(None, description="覆盖 GLM-OCR 引擎 host")
    llama_port: Optional[int] = Field(None, description="覆盖 GLM-OCR 引擎 port")
    # 保留字段（无状态下不再落盘，仅作兼容，忽略持久化）
    save_dir: Optional[str] = Field(None, description="已废弃：无状态服务不持久化，忽略")


class UnlimitedOcrRequest(BaseModel):
    # base64 data URI 列表
    images: List[str] = Field(..., description="base64 图片列表（data:image/xxx;base64,）")
    prompt: str = Field(
        "请准确识别图片中的所有文字，保持原始排版，包括表格、公式、图表说明等",
        description="识别指令",
    )
    doc_id: str = Field("", description="文档标识，用于命名 crops")


class PaddleOcrVlImage(BaseModel):
    page: int
    image_data: str  # base64 data URI


class PaddleOcrVlRequest(BaseModel):
    images: List[PaddleOcrVlImage]
    task_type: str = Field("general", description="general | table | formula | layout")
    language: str = Field("ch", description="ch | en | ...")
    doc_id: str = Field("", description="文档标识，用于命名 crops")


# ============================================================
# 任务队列请求 / 响应
# ============================================================
class TaskSubmitRequest(BaseModel):
    engine: Literal["glmocr", "unlimited-ocr", "paddleocr-vl"] = Field(..., description="目标引擎")
    images: List[str] = Field(..., description="base64 图片列表（data:image/xxx;base64,）")
    doc_id: str = Field("", description="文档标识")
    llama_port: Optional[int] = Field(None, description="GLM-OCR 可选端口")
    prompt: Optional[str] = Field(None, description="Unlimited-OCR 可选 prompt")
    task_type: Optional[str] = Field(None, description="PaddleOCR-VL 任务类型")
    language: Optional[str] = Field(None, description="PaddleOCR-VL 语言")


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str
    position: int
    estimated_wait_seconds: int


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # queued|loading|processing|completed|failed|cancelled
    engine: Optional[str] = None
    progress: Optional[int] = Field(None, description="进度百分比 0-100")
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    submitted_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    position: Optional[int] = None
    error: Optional[str] = None


class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    engine: Optional[str] = None
    pages: List[OCRPage] = Field(default_factory=list)
    total_pages: Optional[int] = None
    page_errors: Optional[dict] = None
    pdf_file_id: Optional[str] = None
    pdf_page_url_template: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class TaskCancelResponse(BaseModel):
    task_id: str
    status: str


class QueueInfoResponse(BaseModel):
    queue_size: int
    current_task_id: Optional[str] = None
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
