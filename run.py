"""
服务启动入口：python run.py
自动加载 .env（若存在），监听 HOST:PORT（默认 0.0.0.0:8088），可被其他服务器调用。
"""
import os
from dotenv import load_dotenv

load_dotenv()

from app import config
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        reload=False,
    )
