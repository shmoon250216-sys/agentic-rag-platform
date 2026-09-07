# Architecture

## 分层结构

```text
app/api       HTTP 接口、鉴权、请求响应
app/services 业务编排层
app/graph    Supervisor 路由与 Agent 工作流
app/rag      文档处理、Embedding、Redis 向量检索
app/tools    MCP 工具注册、调用、结果解析
app/memory   SQLite 会话存储与 LangGraph checkpoint
app/llm      LLM provider 适配层
app/schemas  Pydantic schema
```

## 工作流

当前版本使用 LangGraph `StateGraph` 组织确定性工作流，在不配置外部模型时也可运行和回归测试：

```text
START -> supervisor -> rag/tool/plan/chat/fallback -> answer -> END
```

每个节点只做一件事：

- `supervisor`: 输出结构化路由决策。
- `rag`: 执行 Redis 向量检索并返回来源。
- `tool`: 通过 MCP 协议调用外部工具。
- `plan`: 将复杂业务请求拆为可执行步骤并生成计划。
- `chat`: 处理普通闲聊。
- `fallback`: 统一处理低置信度、异常和安全边界。

SQLite 保存会话、用户长期记忆和每次运行的最终图状态快照；进程内 checkpointer 管理当前实例的图状态。该实现用于审计和调试，不等同于跨进程的节点级断点续跑。
