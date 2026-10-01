# Deployment

## 本地开发与 Milvus

Python 3.11+，先安装 `pip install -e ".[dev]"`。`.env.example` 使用 `RAG_BACKEND=memory` 的离线开发模式；文档不持久化。完整 Milvus 配置及工程边界见 [milvus.md](milvus.md)。

Windows 已配置 Docker Desktop Linux 容器后可运行：

```powershell
powershell -File scripts/start_milvus.ps1 -Python .venv/Scripts/python.exe
```

脚本启动数据库，随后为这次 API 进程选择 Milvus，不改写 `.env` 中的模型密钥。

## 全栈 Compose

要求 Docker Compose 2.20+，先从 `.env.example` 创建本地 `.env`。执行：

```powershell
docker compose up -d --build --wait --wait-timeout 300
```

服务组成：FastAPI/LangGraph API、Milvus Standalone、etcd 元数据存储、MinIO 对象存储。API 使用 Milvus 后端，数据库服务挂载命名卷，SQLite 数据绑定 `./data`。

- 网页：http://127.0.0.1:8010/
- 存活：http://127.0.0.1:8010/health
- 依赖就绪：http://127.0.0.1:8010/health/ready（包含 Collection/索引验证）

Milvus 不自动填充默认制度，先上传合成 PDF 再问答。停止服务使用 `docker compose down`，不要加 `-v`。更换 Embedding 必须换 Collection、修订号并重导文档。

## 验收与兼容模式

真实 Standalone 和 HTTP 验收见 [milvus-validation.json](milvus-validation.json)。本机尚未配置 Docker/WSL，数据库实测发生在 GitHub Ubuntu runner，不等同于本地 Windows 启动。

原 Redis 部署保存在 `compose.redis.yml`，可执行 `docker compose -f compose.redis.yml up -d`；`RAG_BACKEND=redis` 仍为兼容选项。它不会自动迁移到 Milvus。

HTTP Reranker 默认关闭；配置提供方后保留 `RERANK_FAIL_OPEN=true` 可在超时或异常时回退 RRF。真实 Embedding/Reranker 质量与负载表现需单独评测。
