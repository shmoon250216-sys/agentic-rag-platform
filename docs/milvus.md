# Milvus 持久化混合检索

## 设计与数据去向

选择 `RAG_BACKEND=milvus` 后，PDF/DOCX → 文本解析 → 500 字符分片（80 重叠）→ Embedding → Milvus Collection。每条记录对应一个分片，保存 `chunk_id` 主键、`doc_id`、标题、正文、序号、文档分片数、原文字符数以及向量。Milvus 的 etcd、MinIO 和服务数据目录分别挂载 Docker 命名卷，容器重建不删除卷。原始 PDF 文件没有归档；SQLite 仍负责会话和长期记忆，不负责文档向量。

检索分两路：

1. `text` 字段通过 Jieba 分词和 Milvus BM25 Function 生成稀疏向量，以 `SPARSE_INVERTED_INDEX / BM25` 检索。没有每次全量拉取文档到 Python 的操作。
2. 问题经相同 Embedding 转成向量，以 `HNSW / COSINE` 检索 `dense` 字段。HNSW 参数为 M=16、efConstruction=200；查询 ef 至少覆盖候选数。
3. 应用按分片主键进行加权 RRF，取候选后沿用 HTTP Reranker 和超时回退；最终返回来源分片。BM25 和余弦分数不直接相加。

默认 Hash Embedding 只是确定性测试向量，没有真实语义能力。Milvus 不会自动使它变成语义模型。接入真实 Embedding 后需要重建 Collection、调整权重，再扩充评测集；不能沿用旧内存后端的 Recall 数字证明 Milvus 的效果。

## 部署与运行

Windows 需要先具备 Docker Desktop 的 Linux 容器环境（通常依赖 WSL2）。该项目不自动安装系统组件。Compose 基于 [Milvus 官方 Standalone 配置](https://github.com/milvus-io/milvus/blob/v2.6.0/deployments/docker/standalone/docker-compose.yml)，固定 Milvus 2.6.0 与 PyMilvus 2.6.x；对象存储使用 Milvus 官方维护的 `milvusdb/minio:RELEASE.2024-12-18T13-15-44Z` 镜像；本地端口仅绑定 127.0.0.1。

```powershell
pip install -e ".[dev]"
docker compose -f compose.milvus.yml up -d --wait --wait-timeout 300
```

在本机 `.env` 中修改以下非密钥项，其他模型和 Token 配置保留：

```dotenv
RAG_BACKEND=milvus
MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION=rag_chunks_v1
MILVUS_EMBEDDING_REVISION=hash-v1
EMBEDDING_PROVIDER=hash
EMBEDDING_DIMENSIONS=256
RAG_VECTOR_DIMENSIONS=256
```

```powershell
uvicorn app.main:create_app --factory --port 8010
curl http://127.0.0.1:8010/health/ready
```

首次就绪检查创建并校验空 Collection，不自动添加示例制度。Windows 也可用 `powershell -File scripts/start_milvus.ps1 -Python .venv/Scripts/python.exe` 启动数据库与 API；脚本只为本次进程选择 Milvus，不改写 `.env` 或密钥。随后从网页上传 `examples/documents/` 的制度 PDF，再在系统信息中确认 backend=milvus、persistent=true。`/health` 只检查 API 存活，`/health/ready` 同时检查数据库与索引。

全栈启动也可执行 `docker compose up -d --wait --wait-timeout 300`；它使用 Milvus 后端。至少 Docker Compose 2.20，支持 include。停服务使用 `docker compose down`；不要加 `-v`，该参数会删除命名卷及文档。

## 生命周期和故障边界

- 入库前计算并验证全部向量；插入失败按 doc_id 尝试清理残留，失败返回 503。不宣称跨系统事务或任意故障下原子入库。
- 文档列表用 query_iterator 分批读取每份文档的首片元数据；删除用 doc_id 标量过滤删除全部分片。UUID 校验避免将任意用户输入拼入过滤表达式。
- 强一致查询用于入库后检索与删除后读取。SDK/Embedding 调用通过 asyncio.to_thread 执行，并以客户端锁串行化；适合当前单机学习规模，不宣称高并发吞吐。
- Collection 描述记录 Schema 版本、模型标识/修订号和维度，启动时校验必需字段、维度与索引度量；不兼容返回 MILVUS_SCHEMA_MISMATCH，不静默退回内存。即便新模型维度相同也必须换 Collection 并重新导入。
- 检索缓存区分连接、Collection 和模型版本，API 文档变更清空检索与回答缓存。直接在外部修改数据库不会通知进程内缓存，需清缓存或等 TTL；多 API 副本缓存广播尚未实现。
- SDK 请求设超时、就绪检查返回安全错误码，API 不返回 URI/Token 或原始 SDK 异常。服务关闭时释放连接。
- 内存/Redis 数据不会自动搬迁；内存进程退出后丢失的向量无法恢复，需要重新上传源文件。Redis 保留为兼容后端。

## 可复现验证

```powershell
pytest -q tests/test_milvus_store.py tests/test_rag_factory.py
$env:MILVUS_TEST_URI="http://127.0.0.1:19530"
pytest -q tests/test_milvus_integration.py
python scripts/check_milvus_restart.py write
docker compose -f compose.milvus.yml restart milvus
# 等待 http://127.0.0.1:9091/healthz 就绪后执行
python scripts/check_milvus_restart.py read
```

真实服务测试使用独立随机 Collection，并清理自己的数据，覆盖中文 BM25/向量/RRF、客户端重建、全分片删除、模型版本拒绝；重启脚本跨两个 Python 进程和一次数据库服务重启检查数据保留。GitHub Actions 启动真实 Standalone，并上传 JUnit、服务日志、重启结果和 HTTP 验收结果。HTTP 验收另启动独立 FastAPI 进程，上传仓库合成 PDF，核对问答来源并删除文档。无服务时集成测试明确跳过，不当成通过。

## 下一阶段评测

先建立多文档、证据标注与无答案问题，按文档/主题划分开发集和独立测试集；对 BM25、向量、RRF、Reranker 比较 Recall@K、MRR、引用正确性、拒答及 P50/P95 延迟。先接真实 Embedding 再调融合权重，记录模型、Collection、切分参数和数据版本。当前改动不宣称检索准确率或性能提升。

## 已完成的验收（2026-10-01）

[真实服务 CI](https://github.com/shmoon250216-sys/agentic-rag-platform/actions/runs/36874042116) 全部通过：107 项离线回归、1 项真实数据库集成、数据库服务重启保留，以及 HTTP PDF 上传/问答引用/删除。离线阶段 2 项数据库测试因环境跳过，Milvus 已在后续专用步骤真实执行；Redis 未实测。详细口径见 [milvus-validation.json](milvus-validation.json)。本机没有 Docker/WSL，尚未启动 Windows 本地 Milvus，`.env` 也未被静默切换。
