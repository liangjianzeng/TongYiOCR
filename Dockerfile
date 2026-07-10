# PaddleOCR-VL 引擎镜像
# 基础镜像已含 paddleocr/paddlex/torch(cu128)/vllm/fastapi，仅缺 paddlepaddle(布局检测需要) 与本服务依赖。
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu

# 官方基础镜像默认以非 root 用户(paddleocr)运行，会导致 chmod 与 paddlex 临时目录写权限失败；
# 本地单用户部署直接切 root（与之前运行时 --user root 等价）。
USER root

# 证书 / 模型源校验兜底（国内网络）
ENV PYTHONHTTPSVERIFY=0 \
    MODELSCOPE_DOWNLOAD_DISABLE_CERT_VERIFICATION=true \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true \
    PADDLE_PDX_CACHE_HOME=/home/paddleocr/.paddlex

# 安装：paddlepaddle CPU（PP-DocLayoutV3 布局检测走 CPU，模型很小足够快）
# ⚠️ RTX 5060 Ti = Blackwell(sm_120)，当前所有 paddlepaddle-gpu wheel 均缺 sm_120 kernel。
#    因此布局走 CPU，VLM 识别仍走 vLLM GPU。
RUN pip install --no-cache-dir paddlepaddle==3.3.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
        --trusted-host www.paddlepaddle.org.cn \
 && pip install --no-cache-dir pymupdf python-multipart edge-tts

WORKDIR /app

# 复制应用代码（不复制 .env，配置全部由 entrypoint/compose 通过环境变量注入）
COPY app/ ./app/
COPY run.py ./
COPY engine_server.py ./
COPY static/ ./static/

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8091 8118
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
