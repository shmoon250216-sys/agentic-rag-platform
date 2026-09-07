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

本地直接启动时由 `.env` 决定使用 `memory` 或 `redis`。Docker Compose 会启用 Redis 后端，并通过 AOF、快照和具名数据卷保存数据；具体验收步骤见 `docs/redis-rag-milestone.md`。
