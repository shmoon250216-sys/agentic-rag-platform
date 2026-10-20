# 本机持久化存储使用说明

已启用项目现有的 Milvus 后端。文档标题、正文片段和 Embedding 向量保存在 Docker 持久卷中。上传的 PDF/DOCX 原件保存在本项目 `data/documents/originals`，上传指纹和入库记录保存在 `data/documents/uploads.sqlite3`。

## 日常启动（PowerShell）

打开 Docker Desktop，等到引擎运行后，在本项目根目录执行：

```powershell
& 'C:\Users\Administrator\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe' compose -f compose.milvus.yml up -d --wait --wait-timeout 300
& .\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8010
```

如果后端已经运行，无需再次启动。页面地址：http://127.0.0.1:8010/docs 。

## 上传与费用

- 同一份已入库文档，重启后端或数据库后不需要重新上传。
- 首次上传的文档会保存原件、提取文本、切分并调用 Embedding API。相同内容的文件再次上传（包括改名），会复用当前索引中仍存在的成功入库结果，不再次解析或生成向量。
- 普通重启从 Milvus 读取已有片段和向量，不重新解析 PDF、不为整份文档重新计算向量。
- 提问仍可能调用查询 Embedding 和聊天模型，按服务商实际用量计费。
- 修改聊天模型无需重建文档向量。更换 Embedding 模型、维度或切分方式通常需要计划重建索引。

## 存储配置和数据保留

本机 `.env` 的存储配置为：

```dotenv
RAG_BACKEND=milvus
MILVUS_COLLECTION=rag_chunks_qwen37_1024_v1
MILVUS_EMBEDDING_REVISION=qwen37-1024-v1
```

其余模型配置与 API 密钥保持不变。Milvus、etcd 和 MinIO 的数据由 `compose.milvus.yml` 中三个命名卷保存。关闭程序不会删除这些卷；不要执行 `docker compose ... down -v` 或在 Docker Desktop 中删除这些数据卷，否则会丢失知识库。持久卷不是备份。

## 本次验收

- Milvus 集成测试通过：写入、重新连接、检索与删除；使用独立测试集合和本地测试向量。
- 实际重启 Milvus 后，独立测试集合中的数据仍能读取、检索，随后清理了该测试集合。
- 迁入原有 3 份文档、16 个片段，其中报销制度测试 PDF 为 14 个片段。
- 后端切换后，接口返回 `backend=milvus`、`persistent=true`、`connected=true`，文档 ID 与数量和迁移记录一致。
- 本机验收记录：`data/persistence-validation.json`（已被 Git 忽略，无 API 密钥）。

原件保存、去重规则与故障边界见 [文件上传说明](document-uploads.md)。
