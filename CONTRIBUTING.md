# 贡献指南（Contributing）

感谢你关注 **彤熠OCR网关（TongYiOCR）**！任何形式的贡献（Issue、PR、文档、用例）都欢迎。

## 开发环境

要求 Python ≥ 3.10。推荐用虚拟环境：

```bash
python -m venv --system-site-packages .venv
source .venv/Scripts/activate      # Windows
# 或 source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

- 仅使用 GLM-OCR 引擎时，额外安装：`pip install -r requirements.glmocr.txt`
- PaddleOCR-VL / Unlimited-OCR 引擎运行在 Docker 容器内，无需在本机安装其依赖

## 运行与测试

```bash
# 启动代理（默认监听 0.0.0.0:8088）
python run.py
# 或开发模式（热重载）
uvicorn app.main:app --reload --port 8088

# 运行接口冒烟测试（不需要任何 OCR 引擎，离线可跑）
pip install pytest httpx
pytest

# 代码风格（可选）
pip install ruff
ruff check app tests
ruff format app tests
```

测试不依赖 GPU / 模型权重：引擎未启动时，对应 `/parse` 路由返回 `502` 且 `ok=false`，
`/health` 始终返回 `200`。CI 中也仅运行这类离线冒烟测试。

## 提交规范

1. 先开 Issue 讨论较大的改动（新引擎、破坏性 API 变更等）。
2. Fork 后从 `main` 切出特性分支，分支名建议 `feat/`、`fix/`、`docs/`。
3. 保持 PR 聚焦、描述清晰；如有关联 Issue 请在说明中引用。
4. 确保所有测试通过（CI 会运行 `pytest`）。
5. 不要提交 `.env`、模型权重、日志等（见 `.gitignore`）。

## 代码约定

- 代理（`app/`）只做**无状态转发**，不在服务端持久化用户数据；裁剪图以 base64 内联返回。
- 三引擎统一输出结构见 `app/schemas.py`：`pages[{page,width,height,markdown,elements[],crops[]}]`。
- 新增引擎请实现同名 `*_client.py`，并在 `app/main.py`、`app/config.py`、`app/schemas.py` 中接入，
  保持统一响应结构不变。
- 配置一律走环境变量 / `.env`（见 `.env.example`），不要在代码中硬编码地址或密钥。

## 行为准则

请友善、专业地交流；对他人的提问与评审保持耐心。我们采用常见的开源社区行为准则精神。
