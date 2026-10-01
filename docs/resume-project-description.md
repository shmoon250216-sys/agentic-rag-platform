# 项目描述与证据口径

Agentic RAG｜企业制度与办事智能助手

Python｜FastAPI｜LangGraph｜Milvus｜BM25｜RRF / Reranker｜SQLite

- 将制度问答、结构化工具、办事计划组织为 LangGraph 五分支工作流，支持 PDF/DOCX 入库和来源引用。
- 接入 Milvus Standalone 保存文档分片与向量，配置 Jieba/BM25 稀疏倒排及 HNSW/COSINE 索引，以加权 RRF 融合双路召回，提供可选 HTTP Reranker 和故障回退。
- 补齐文档列表/整份删除、Schema 与模型版本校验、缓存失效、依赖就绪检查、SDK 线程卸载及容器卷部署；真实数据库生命周期和重启验证由 CI 运行，具体结果见 Actions。
- 以 SQLite 管理最近消息、有界历史摘录与长期偏好，支持更新、过期、容量限制和会话归属检查。
- 现有 8 个标注问题用于检索策略对比；历史离线数字不能代表 Milvus 或真实语义模型表现。本地回归 107 通过、2 个真实数据库测试因环境跳过。

AI 协作实现的本地项目，不宣称真实企业接入、生产级多租户鉴权或线上收益。Hash Embedding 为测试向量，实际语义效果、Reranker 模型效果和负载指标待后续独立评测。原始 PDF 没有归档，内存/Redis 数据不自动迁移。
