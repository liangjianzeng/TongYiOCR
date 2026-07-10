#!/bin/bash
# PaddleOCR-VL 引擎容器 — 引擎服务（:8091）+ vLLM VLM（:8118）
# 容器外部由统一代理（本机 :8088）按 /paddleocr-vl/* 路由到本容器 :8091。
set -e

export PYTHONHTTPSVERIFY=0
export MODELSCOPE_DOWNLOAD_DISABLE_CERT_VERIFICATION=true
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true
export PADDLE_PDX_CACHE_HOME=/home/paddleocr/.paddlex

# ---- 引擎服务（OCRVl engine）配置 ----
export PADDLE_OCR_ENGINE=paddle
export PADDLE_OCR_DEVICE=cpu
export PADDLE_OCR_PIPELINE_VERSION=v1.6
export PADDLE_OCR_VL_SERVER_URL=http://127.0.0.1:8118/v1
export PADDLE_OCR_VL_MODEL=PaddleOCR-VL-1.6-0.9B
export PADDLE_OCR_VL_API_KEY=EMPTY
export ENGINE_PORT=8091
export HOST=0.0.0.0

# ---- 1) 启动 vLLM VLM 服务（后台） ----
cd /app
echo "[entrypoint] starting vLLM VLM server on :8118 ..."
paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --host 0.0.0.0 --port 8118 --backend vllm > /var/log/vllm.log 2>&1 &
VLLM_PID=$!

# ---- 2) 等待 vLLM 就绪 ----
echo "[entrypoint] waiting for vLLM at http://127.0.0.1:8118/v1/models ..."
python - <<'PY'
import sys, time, urllib.request
url = "http://127.0.0.1:8118/v1/models"
for i in range(120):
    try:
        req = urllib.request.Request(url)
        req.add_header("Authorization", "Bearer EMPTY")
        with urllib.request.urlopen(req, timeout=5) as r:
            if r.status == 200:
                print("[entrypoint] vLLM ready")
                sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
print("[entrypoint] ERROR: vLLM did not become ready in time", file=sys.stderr)
sys.exit(1)
PY

# ---- 3) 启动引擎服务（前台，接管容器主进程） ----
echo "[entrypoint] starting PaddleOCR-VL engine on :8091 (layout on GPU) ..."
exec python engine_server.py
