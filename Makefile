.PHONY: help install install-glmocr run dev test lint format docker-build docker-up docker-down clean

help:  ## 显示本帮助
	@echo "TongYiOCR 开发命令："
	@echo "  make install         安装代理依赖（requirements.txt）"
	@echo "  make install-glmocr  额外安装 GLM-OCR 引擎客户端依赖"
	@echo "  make run             启动代理（python run.py，默认 :8088）"
	@echo "  make dev             开发模式（uvicorn --reload）"
	@echo "  make test            运行接口冒烟测试（pytest，离线可跑）"
	@echo "  make lint            代码检查（ruff）"
	@echo "  make format          代码格式化（ruff format）"
	@echo "  make docker-build    构建 PaddleOCR-VL 引擎镜像"
	@echo "  make docker-up       启动 Docker 引擎（compose up -d）"
	@echo "  make docker-down     停止 Docker 引擎"
	@echo "  make clean           清理 __pycache__ / *.pyc"

install:
	pip install -r requirements.txt

install-glmocr:
	pip install -r requirements.glmocr.txt

run:
	python run.py

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8088

test:
	pytest

lint:
	ruff check app tests

format:
	ruff format app tests

docker-build:
	docker compose -f docker-compose.gpu.yml build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
