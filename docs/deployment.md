# Deployment

当前项目已经具备 Docker Compose 本地部署和冒烟检查能力。

## 本地开发启动

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8010
```

## Docker Compose 启动

```powershell
docker compose up --build
```

启动后访问：

```text
http://127.0.0.1:8010/docs
http://127.0.0.1:8010/health
http://127.0.0.1:8010/health/ready
```

## 冒烟检查

本地服务启动后运行：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py --base-url http://127.0.0.1:8010
```

检查内容：

- `/health` 和 `/health/ready` 是否可用。
- `/api/v1/tools` 是否能列出工具。
- `/api/v1/chat` 是否能完成一次 RAG 问答。

## Compose 服务

```text
api: FastAPI + LangGraph Agent 服务
redis: Redis Stack，保存 RAG 文档、分片和向量索引
```

本地直接启动时由 `.env` 决定使用 `memory` 或 `redis`。Docker Compose 会启用 Redis 后端，并通过 AOF、快照和具名数据卷保存数据；启动后可通过 `/health/ready` 检查 Redis 与索引状态。

Redis 后端会建立中文全文 BM25 与 KNN 向量索引，再用 RRF 融合两路结果。`idx:rag_chunks:v2` 是本次检索结构对应的默认索引名；从旧版升级时应使用新索引名重新入库。

Reranker 默认关闭。部署外部精排服务后设置 `RERANK_PROVIDER=http`、`RERANK_URL`、`RERANK_MODEL` 和密钥；保留 `RERANK_FAIL_OPEN=true` 可在精排服务超时或响应异常时回退到 RRF 结果。
