"""
彤熠OCR网关（TongYiOCR）—— 统一 OCR 代理 配置（FastAPI 监听 :8088）

本服务是纯转发代理，按路径前缀把请求路由到三个后端 OCR 引擎：
  /glmocr/*        → llama.cpp VLM（本机 localhost:8089），经 glmocr SDK(selfhosted)
  /unlimited-ocr/* → Docker 容器（默认 localhost:8090）
  /paddleocr-vl/*  → Docker 容器（默认 localhost:8091）

所有配置均可被环境变量 / .env 覆盖。

设计要点（对齐 OCR 远程服务接口需求）：
- 无状态：代理不持久化任何数据，crops 以 base64 内联返回。
- 统一输出：pages[{page,width,height,markdown,elements[],crops[]}]。
- 任务队列：串行单 worker + engine_lock，支持进度/取消（显存只够一个引擎）。
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _env(key, default):
    return os.getenv(key, default)


def _as_bool(v, default=False):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ---------- 服务自身 ----------
HOST = _env("HOST", "0.0.0.0")
PORT = int(_env("PORT", "8088"))
WORKERS = int(_env("WORKERS", "1"))
REQUEST_TIMEOUT = int(_env("REQUEST_TIMEOUT", "600"))  # 大图/多页需较长超时

# ---------- GLM-OCR（llama.cpp VLM，本机 8089）----------
# 经 glmocr SDK（selfhosted 模式）转发到 llama.cpp 的 VLM 推理端点。
# SDK 内部用 PP-DocLayoutV3 做版面辅助（layout_device 指定其运行设备）。
GLM_OCR_ENABLED = _as_bool(_env("GLM_OCR_ENABLED", "1"))
GLM_OCR_API_HOST = _env("GLM_OCR_API_HOST", "127.0.0.1")
GLM_OCR_API_PORT = int(_env("GLM_OCR_API_PORT", "8089"))
GLM_OCR_LAYOUT_DEVICE = _env("GLM_OCR_LAYOUT_DEVICE", "cuda")  # PP-DocLayoutV3 运行设备
# 本地 PP-DocLayoutV3 模型目录（绕过联网 from_pretrained）。
# 留空则依赖 HF_HUB_OFFLINE=1 + HuggingFace 本地缓存（部署机建议此方式）。
GLM_OCR_LAYOUT_MODEL_DIR = _env("GLM_OCR_LAYOUT_MODEL_DIR", "")

# ---------- Unlimited-OCR（Docker 容器，默认 8090）----------
# 容器内运行 Unlimited-OCR 模型；FastAPI 代理接收 base64 并转发。
# 真实端点 /v1/ocr/multi，支持 data URI(base64) 入参，enable_grounding 返回 bbox。
UNLIMITED_OCR_ENABLED = _as_bool(_env("UNLIMITED_OCR_ENABLED", "1"))
UNLIMITED_OCR_URL = _env("UNLIMITED_OCR_URL", "http://localhost:8090")  # 容器映射出的主机地址
UNLIMITED_OCR_PARSE_PATH = _env("UNLIMITED_OCR_PARSE_PATH", "/v1/ocr/multi")
UNLIMITED_OCR_HEALTH_PATH = _env("UNLIMITED_OCR_HEALTH_PATH", "/health")
UNLIMITED_OCR_PROMPT = _env(
    "UNLIMITED_OCR_PROMPT",
    "请准确识别图片中的所有文字，保持原始排版，包括表格、公式、图表说明等",
)

# ---------- PaddleOCR-VL（Docker 容器，默认 8091）----------
# 容器内运行 PaddleOCR-VL 模型（自带 PP-DocLayoutV3 版面分析）；FastAPI 代理转发。
PADDLEOCR_VL_ENABLED = _as_bool(_env("PADDLEOCR_VL_ENABLED", "1"))
PADDLEOCR_VL_URL = _env("PADDLEOCR_VL_URL", "http://localhost:8091")  # 容器映射出的主机地址
PADDLEOCR_VL_PARSE_PATH = _env("PADDLEOCR_VL_PARSE_PATH", "/parse")
PADDLEOCR_VL_HEALTH_PATH = _env("PADDLEOCR_VL_HEALTH_PATH", "/health")

# ---------- 任务队列（显存限制解决方案）----------
MAX_QUEUE_SIZE = int(_env("MAX_QUEUE_SIZE", "10"))
TASK_TIMEOUT = int(_env("TASK_TIMEOUT", "3600"))  # 单任务超时（秒）
# 每个任务预估耗时（秒），用于估算等待时间
ESTIMATED_TASK_SECONDS = int(_env("ESTIMATED_TASK_SECONDS", "900"))
# 代理侧临时文件目录（base64 解码落盘用，请求完成后清理）
OCR_WORKSPACE_DIR = _env("OCR_WORKSPACE_DIR", "")
