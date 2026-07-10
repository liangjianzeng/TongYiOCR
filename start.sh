#!/bin/bash
# 启动 PaddleOCR-VL 1.6 接口服务（宿主 venv + 可选 Docker VLM 推理）
set -e
cd "$(dirname "$0")"
if [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate; fi
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PADDLEX_HOME="$PWD/.cache/paddlex"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$PADDLEX_HOME"
exec python run.py
