# 简历项目表述（与源码同步）

Agentic RAG｜企业制度与办事智能助手

Python / FastAPI / LangGraph / BM25 / Vector / RRF / Reranker / Redis Stack / SQLite

- 为制度和报销材料提供 PDF/DOCX 入库、带来源问答、工具处理与规划入口。
- 使用 LangGraph StateGraph 组织规则 Supervisor 五分支，采用 BM25/向量双路召回与加权 RRF，支持 HTTP Reranker 和异常回退。
- 将最近对话、有界历史摘录和相关长期记忆接入模型，补全省略追问的主题后检索；用 SQLite 保存用户偏好，支持语言/详略更新、过期清理和容量限制，并验证普通与 SSE 接口的一致性。
- 96 项测试中 95 项通过，1 项 Redis 实机集成按环境跳过；新增 18 项记忆专项验证。

范围：本地机制测试；默认 FakeLLM/Hash Embedding。记忆是自定义实现，摘要是原文摘录，预算按字符；没有 Mem0/Memobase、通用语义冲突消解、真实模型记忆准确率或生产业务指标。
