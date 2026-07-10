"""
彤熠OCR网关（TongYiOCR）统一代理 —— 接口冒烟测试（离线可跑）。

不依赖任何 OCR 引擎：引擎未启动时，对应 /parse 路由应返回 502 且 ok=False，
/health 路由始终返回 200。可用于 CI 或本地快速校验代理本身是否正常。

运行：  pip install -r requirements.txt && pytest
"""
import base64

import pytest
from fastapi.testclient import TestClient

from app.main import app

# 1x1 透明 PNG，仅用于构造合法请求体（无需真实内容）
PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLv"
    "AAAAAElFTkSuQmCC"
)
IMG = f"data:image/png;base64,{PNG_1PX}"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "tongyi-ocr"
    assert body["version"] == "1.0.0"
    assert "engines" in body


def test_engine_health_endpoints(client):
    # 引擎未启动时这些端点也应返回 200（ok 可能为 False）
    for path in ("/glmocr/health", "/unlimited-ocr/health", "/paddleocr-vl/health"):
        r = client.get(path)
        assert r.status_code == 200


def test_index_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_openapi_schema(client):
    # 校验所有路由都能生成合法的 OpenAPI schema
    schema = app.openapi()
    assert schema["info"]["title"]
    paths = schema["paths"]
    for p in ("/glmocr/parse", "/unlimited-ocr/parse", "/paddleocr-vl/parse",
             "/task/submit", "/task/{task_id}/status"):
        assert p in paths


def test_paddleocr_vl_parse_response_shape(client):
    # 不假设引擎是否启动：返回 200（ok=True, 含 pages）或 502（ok=False，引擎不可达）均视为正常
    r = client.post(
        "/paddleocr-vl/parse",
        json={"images": [{"page": 1, "image_data": IMG}], "task_type": "general", "language": "ch"},
    )
    assert r.status_code in (200, 502)
    body = r.json()
    assert "ok" in body
    if r.status_code == 502:
        assert body["ok"] is False
    else:
        assert body["ok"] is True
        assert "pages" in body


def test_parse_validation_error(client):
    # 缺必填字段应返回 422
    r = client.post("/paddleocr-vl/parse", json={"task_type": "general"})
    assert r.status_code == 422


def test_task_submit_and_queue_info(client):
    r = client.post(
        "/task/submit",
        json={"engine": "paddleocr-vl", "images": [IMG], "doc_id": "test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "task_id" in body
    assert body["status"] in ("queued", "processing")

    info = client.get("/task/queue/info")
    assert info.status_code == 200
    assert "queue_size" in info.json()


def test_task_submit_invalid_engine(client):
    r = client.post("/task/submit", json={"engine": "not-a-real-engine", "images": [IMG]})
    assert r.status_code == 422
