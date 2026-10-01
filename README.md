# Agentic RAG Platform｜企业制度与办事智能助手

GitHub：<https://github.com/shmoon250216-sys/agentic-rag-platform>

面向企业制度、报销规范和办事材料分散的场景，系统把文档入库、混合检索、来源引用、工具调用、任务规划、会话记忆和异常兜底组织为一条可运行的 Agent 工作流。用户可以上传 PDF/DOCX 制度文档，通过统一聊天入口查询规则、调用工具或生成办事计划，并在管理接口中追踪会话、记忆和工作流快照。

## 解决的问题

- **入口分散**：将知识问答、结构化工具和复杂任务规划统一到一个聊天接口，由 Supervisor 决定执行分支。
- **文档难检索**：解析 PDF/DOCX 后切分文档，分别执行 BM25 关键词召回和向量召回，通过 RRF 融合排名，并可选用 Reranker 精排；回答同时返回来源片段。
- **连续追问缺少前文**：将最近对话、较早用户摘录和相关长期记忆注入模型；对常见省略追问补充前文主题后检索。语言与详略偏好按字段更新，记录设有过期和容量限制。
- **外部依赖影响本地开发**：默认使用确定性的本地模型和 Hash Embedding，LLM、Embedding 与检索后端均通过适配层切换。全栈部署默认使用 Milvus 持久化，离线开发显式使用内存。
- **修改容易破坏链路**：固定评测覆盖 RAG、工具、规划、聊天和安全兜底，并以自动化测试验证文档、缓存、记忆和接口行为。

## 核心流程

```mermaid
flowchart LR
    U[用户请求] --> API[FastAPI]
    API --> S[LangGraph Supervisor]
    S -->|制度与知识| R[RAG]
    S -->|结构化动作| T[Tool]
    S -->|复杂任务| P[Plan]
    S -->|普通对话| C[Chat]
    S -->|安全边界| F[Fallback]
    R --> B[BM25 召回]
    R --> V[KNN 向量召回]
    B --> H[RRF 融合]
    V --> H
    H --> X[可选 Reranker]
    B --> K[(Milvus / Memory / Redis Stack)]
    V --> K
    R --> L[LLM Adapter]
    T --> L
    P --> L
    C --> L
    S --> M[(SQLite 会话与记忆)]
```

## 已实现能力

- FastAPI 服务、Token 校验、统一聊天接口和 SSE 流式响应。
- LangGraph `StateGraph`，包含 `rag`、`tool`、`plan`、`chat`、`fallback` 五类分支。
- PDF/DOCX 上传校验、文本解析、重叠切分、BM25/向量双路召回、加权 RRF 融合和来源返回。
- 可配置 HTTP Reranker；服务超时或响应异常时按配置回退到 RRF 排名。
- 本地 Hash Embedding 与 OpenAI-compatible Embedding 适配器。
- Milvus 持久化后端：Jieba/BM25 稀疏倒排索引、HNSW/COSINE 向量索引、模型版本检查、文档删除和加权 RRF；保留内存与 Redis Stack 兼容适配。
- SQLite 分层上下文：最近消息窗口、有界用户摘录摘要、跨会话长期记忆；偏好更新、过期/容量清理、会话归属校验和最终运行快照。详见 [`docs/memory.md`](docs/memory.md)。
- MCP 风格工具注册表、参数 Schema 校验和标准化工具结果。
- TTL 检索缓存与回答缓存，文档变更时主动失效。
- 存活/就绪检查、Docker Compose、质量门禁与自动化测试。

## 快速启动

要求 Python 3.11+。下面是无数据库服务的离线开发模式（`RAG_BACKEND=memory`）。**Milvus 完整部署见 [docs/milvus.md](docs/milvus.md)**；全栈 `docker compose up` 默认使用 Milvus，需 Docker Compose 2.20+，Windows 需先配置 Linux 容器环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn app.main:create_app --factory --reload --port 8010
```

启动后访问：

- 操作界面：<http://127.0.0.1:8010/docs>
- API 文档：<http://127.0.0.1:8010/api/docs>
- 存活检查：<http://127.0.0.1:8010/health>
- 依赖检查：<http://127.0.0.1:8010/health/ready>

```powershell
curl -X POST "http://127.0.0.1:8010/api/v1/chat" `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer dev-token" `
  -d '{"message":"差旅住宿标准是什么？","user_id":"demo-user"}'
```

## 真实文档验证

`examples/documents/` 提供一份 8 页的企业差旅与费用报销制度测试 PDF；`examples/evaluation_questions.json` 给出预期查询方向。该文档用于验证上传、解析、切分、召回和引用链路，不含真实企业或个人数据。

```powershell
python scripts/run_evaluation.py
pytest -q
```

Milvus 更新后的本地回归为 107 项通过、2 项真实数据库集成按环境跳过（Redis、Milvus）。真实 Milvus 验收另由 GitHub Actions 启动 Standalone，结果以 Actions 与上传日志为准；见 [Milvus 说明](docs/milvus.md)。评测数据与结果见 `app/evaluation/`、`docs/evaluation-results.json` 和 `docs/retrieval-evaluation-results.json`。

检索专项评测使用仓库内 8 页、5764 字的合成制度文档和 8 个标注问题。当前离线 Hash Embedding 不具备真实语义能力，因此默认将 BM25/RRF 权重设为 `1.0/0.1`；专项结果中 BM25 的 Recall@3 为 100%，RRF 的 Recall@3 为 87.5%。该结果用于暴露本地向量模型的限制，不包装为线上效果。切换真实 Embedding 后应重新标定权重。

```powershell
python scripts/evaluate_retrieval.py
```

## 检索与精排配置

默认不依赖外部模型，使用 BM25 与低权重 Hash 向量结果做 RRF 融合：

```text
RAG_CANDIDATE_MULTIPLIER=4
RAG_RRF_K=60
RAG_BM25_WEIGHT=1.0
RAG_VECTOR_WEIGHT=0.1
RERANK_PROVIDER=none
```

接入支持 `model/query/documents/top_n` 请求格式以及 `results[index,relevance_score]` 响应格式的 HTTP Reranker 时：

```text
RERANK_PROVIDER=http
RERANK_URL=https://your-rerank-service/v1/rerank
RERANK_API_KEY=your-key
RERANK_MODEL=your-reranker-model
RERANK_CANDIDATE_COUNT=12
RERANK_FAIL_OPEN=true
```

系统先取 RRF 候选，再调用精排服务；`RERANK_FAIL_OPEN=true` 时，精排服务不可用不会阻断知识问答，而是保留 RRF 顺序。

## 配置与边界

- 默认 `FakeLLMClient` 用于离线、确定性开发；配置 OpenAI-compatible Provider 后才会生成模型回答。
- 默认 Hash Embedding 侧重可复现与零外部依赖，专项评测已显示它会给 RRF 引入噪声；真实语义检索效果需要配置外部 Embedding 模型后重建索引并重新评测。
- Supervisor 与工具选择当前采用规则决策；`plan` 分支输出结构化计划，不执行自主多步工具循环。
- SQLite 保存有界会话历史、用户摘录摘要、长期记忆和最终快照；进程内 LangGraph 检查点在每次图运行结束后释放，不支持跨进程从中间节点续跑。上下文采用字符预算，摘要为原文摘录，非 LLM 语义压缩；不使用 Mem0/Memobase。
- 本机尚无 Docker/WSL，未声称已在 Windows 启动 Milvus；CI 真实服务测试与本地运行须区分。内存及 Redis 文档不会自动迁移，需重新上传源文件。
- HTTP Reranker 的请求、排序和故障回退由自动化测试覆盖，但仓库不附带第三方模型密钥，也不宣称完成真实模型效果评测。
- 开发 Token 与客户端传入的 `user_id` 适合本地验证，不等同于企业多租户鉴权。

详细设计见 [`docs/architecture.md`](docs/architecture.md)，部署配置见 [`docs/deployment.md`](docs/deployment.md)。

## 记忆管理验证

本次新增 18 项记忆专项用例，验证普通与 SSE 上下文传递、连续追问检索、偏好更新、过期清理、容量上限、旧库迁移与同会话并发控制。全套测试结果见上文；记忆专项仍为 18 项。测试为本地机制验证，不代表真实大模型代词理解准确率。运行 `python -m pytest tests/test_memory_context.py -q` 可复现；设计、默认保留规则与配置见 [记忆说明](docs/memory.md)。
