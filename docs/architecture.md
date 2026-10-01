# Architecture

## Milvus 更新（2026-10-01）

全栈部署现使用 Milvus Standalone，部署、数据 Schema、中文 BM25/HNSW 检索、生命周期及故障边界见 [milvus.md](milvus.md)。本文原 Redis 配置仍为兼容模式，启动用 `docker compose -f compose.redis.yml up -d`；不再是默认全栈配置。


## 分层结构

```text
app/api       HTTP 接口、鉴权、请求响应
app/services 业务编排层
app/graph    Supervisor 路由与 Agent 工作流
app/rag      文档处理、BM25/向量召回、RRF 融合与可选精排
app/tools    MCP 工具注册、调用、结果解析
app/memory   SQLite 会话存储与 LangGraph checkpoint
app/llm      LLM provider 适配层
app/schemas  Pydantic schema
```

## 工作流

当前版本使用 LangGraph `StateGraph` 组织确定性工作流，在不配置外部模型时也可运行和回归测试：

```text
START -> supervisor -> rag/tool/plan/chat/fallback -> END
```

每个节点只做一件事：

- `supervisor`: 输出结构化路由决策。
- `rag`: 执行 BM25/向量双路召回、RRF 融合与可选精排，并返回来源。
- `tool`: 通过项目内 MCP 风格工具注册表调用本地工具，并由模型解释结果（不是 MCP 协议客户端）。
- `plan`: 将复杂业务请求拆为可执行步骤并生成计划。
- `chat`: 处理普通闲聊。
- `fallback`: 处理命中规则的安全边界，返回固定提示；不是通用异常恢复器。

SQLite 保存会话、用户长期记忆和每次运行的最终图状态快照；进程内 checkpointer 管理本次运行状态，结束后释放。该实现用于审计和调试，不等同于跨进程的节点级断点续跑。

## 检索链路

```text
query -> tokenize/embed
      -> BM25 top-N --------┐
      -> vector KNN top-N --┴-> weighted RRF -> optional HTTP reranker -> top-K
```

- 内存后端使用项目内的标准 BM25 公式和余弦相似度遍历分片。
- Redis Stack 后端通过中文全文索引的 `BM25STD` scorer 和向量 KNN 查询形成两张候选榜单。
- RRF 只使用名次，不直接比较 BM25 分数和余弦分数；两路权重和常数 `k` 均可配置。
- 默认 Hash Embedding 只是确定性开发替身，所以默认降低向量榜单权重。换用真实 Embedding 后需要依据标注集重新调参。
- Reranker 位于融合之后，只处理少量候选。HTTP 服务失败时可选择保留 RRF 结果，避免外部精排依赖阻断主链路。

## 上下文和记忆

请求执行前由 ChatService 读取会话历史和持久化摘录，构建有界最近消息窗口及较早用户摘要；常见省略追问补充前文主题，query 用于检索、message 保留原始问题。长期记忆按用户存储，语言/详略槽支持更新，其他事实按同类型同文本去重。完整策略、限制和验证入口见 [memory.md](memory.md)。
