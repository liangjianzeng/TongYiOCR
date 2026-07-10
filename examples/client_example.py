#!/usr/bin/env python3
"""
彤熠OCR网关（TongYiOCR）调用示例。

演示如何：
  1) 同步调用 /paddleocr-vl/parse
  2) 异步提交 /task/submit 并轮询 /task/{id}/status 直至完成

用法：
  python examples/client_example.py --image path/to/page.png [--engine paddleocr-vl]

依赖：仅标准库（urllib / base64），无需安装任何第三方包。
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8088"


def _post(path: str, payload: dict, timeout: int = 300) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:500]}") from e


def _image_to_data_uri(path: str) -> str:
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def sync_parse(image_path: str, engine: str) -> dict:
    img = _image_to_data_uri(image_path)
    if engine == "paddleocr-vl":
        return _post("/paddleocr-vl/parse", {
            "images": [{"page": 1, "image_data": img}],
            "task_type": "general", "language": "ch",
        })
    if engine == "unlimited-ocr":
        return _post("/unlimited-ocr/parse", {"images": [img]})
    if engine == "glmocr":
        return _post("/glmocr/parse", {"images": [img]})
    raise ValueError(f"未知引擎: {engine}")


def async_submit_and_wait(image_path: str, engine: str, poll: float = 2.0) -> dict:
    img = _image_to_data_uri(image_path)
    sub = _post("/task/submit", {"engine": engine, "images": [img], "doc_id": "example"})
    task_id = sub["task_id"]
    print(f"[example] 已提交任务 {task_id}，排队位置 {sub.get('position')}")
    while True:
        st = _post(f"/task/{task_id}/status", {})
        status = st.get("status")
        prog = st.get("progress")
        print(f"[example] 状态={status} 进度={prog}%")
        if status in ("completed", "failed", "cancelled"):
            break
        time.sleep(poll)
    return _post(f"/task/{task_id}/result", {})


def main():
    ap = argparse.ArgumentParser(description="TongYiOCR 调用示例")
    ap.add_argument("--image", required=True, help="待识别图片路径")
    ap.add_argument("--engine", default="paddleocr-vl",
                    choices=["glmocr", "unlimited-ocr", "paddleocr-vl"])
    ap.add_argument("--mode", default="sync", choices=["sync", "async"])
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    if not Path(args.image).exists():
        print(f"[example] 图片不存在: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "sync":
        result = sync_parse(args.image, args.engine)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    else:
        result = async_submit_and_wait(args.image, args.engine)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
