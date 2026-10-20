# 真实检索评测与证据引用

本次升级保留原有 Milvus Schema 与 Collection，不重建或清空用户知识库。
在线引用达到**分片级可核验**，不是 PDF 页码标注、答案事实正确性证明或制度生效版本管理。

## 一次问答怎样获得可核验引用

1. 用户问题（追问先补充前文主题）进入检索。
2. Milvus 返回 BM25、HNSW/COSINE 候选；应用端按原权重做 RRF。
3. `SourceChunk` 保留 `doc_id`、`chunk_id`、`chunk_index` 和完整片段。对片段 UTF-8
   文本计算 `content_sha256`；它是内容指纹，不是正确概率、签名或完整文档版本号。
4. 完成可选重排后，按最终顺序赋予本回答内的 `S1`、`S2` 等标签。模型收到带编号、
   分片 ID 和原文的上下文，提示在事实后使用 `[S1]`。标签只在当前回答内有效。
5. 返回结果中附带引用编号检查：未知编号和完全缺失引用会产生警告。此检查不判断
   “该片段是否真的支持该句话”，也不会把检索候选自动说成全部被答案采用的证据。
6. 网页执行详情显示编号、分片 ID、内容指纹和完整片段；排序分不再显示成正确率百分比。
   点击“核验当前原文片段”，带现有 Bearer 凭据读取：
   `GET /api/v1/documents/{doc_id}/evidence/{chunk_id}?sha256=...`
7. API 使用精确分片 ID 回查当前 Milvus：片段不存在返回 404；指纹不同返回 409；
   一致则返回当前片段。删除后的旧引用不能仅凭聊天缓存冒充可用原文。

分片文本是解析、空白归一化和切分后的入库文本，不是原始 PDF 字节。
已有上传原文件下载功能可用于人工对照；本次不为缺乏可靠位置映射的旧数据编造页码。
`chunk_index` 为零起始序号。现有 memory 后端也附带分片身份；在线回查接口目前只支持
Milvus。Redis 等旧后端仍兼容，但不能宣称它们已有同等在线核验能力。

## 为什么评测不用原来 8 题的脚本

`scripts/evaluate_retrieval.py` 是旧的 Hash Embedding 本地对比。新增
`scripts/evaluate_live_retrieval.py` 调用 `MilvusKnowledgeBase.search_strategies`，
生产 `search()` 也调用这个共同实现，再取其中的 RRF 结果。不存在评测单独实现一套
BM25 或偷偷回落到 Hash 的情况。除显式 `--allow-hash` 外，脚本拒绝 Hash 配置。

它不经过意图路由、用户长期记忆或答案生成，目的是先隔离检索质量。多轮题仅通过
现有 `resolve_followup` 处理已标注的用户历史。故这不是端到端问答、模型拒答或权限评测。

## 标注集怎样构建

- `examples/retrieval_benchmark_v1.json`：8 份合成制度，16 个证据单元，56 题。
- 48 题分别为 16 个直接问法、16 个口语改写、16 个多轮追问。
- 4 题要求两个证据单元；4 题在资料中没有答案。
- 每个证据单元标注文档 key 和确切正文摘录，而非标题关键词；问题引用证据 ID。
- 42 题 dev，14 题 test。相同证据的改写和追问保留在同一分组，校验器禁止跨组。
  test 为预留主题；不以 test 结果反复调参。全部资料仍在检索语料中，这是正常检索设置。
- 标注为 AI 协作设计的合成标签，未冒称真实用户日志或人工双人审校。
  使用者应逐题检查证据是否充分，尤其无答案题和多条件题。
- 文档版本、数据 SHA256、代码 SHA256、Embedding 名称/修订、维度、RRF 权重写入报告。
- 可以运行 `python scripts/build_retrieval_benchmark.py` 重建同一静态数据，
  不会调用大模型生成新标签。

## 指标与运行原理

先给每题建立应召回的证据集合，再看前 K 个候选覆盖多少个不同证据：

- Hit@K：至少覆盖一条正确证据的题目比例。
- Recall@K：每题覆盖的证据数 / 该题全部标注证据数，然后按题平均。
- Complete@K：覆盖本题全部标注证据的题目比例。
- MRR@5：前五条中首条正确证据的倒数排名，未命中记零。

按文档 key 和确切正文同时匹配。相同标题、其他文档中的同样文字不能冒充该证据；
重叠分片覆盖同一个证据只计一次。这里衡量的是“证据单元召回率”，不是所有相关 chunk
的全集召回率。无答案题没有有效分母，单独记录返回候选情况，不伪造 Recall=1 或拒答率。
请求失败留在指标分母中记零，同时独立记录错误数。

同一个问题只生成一次查询向量，使用相同候选池比较 BM25、向量和 RRF。
配置 HTTP Reranker 时额外记录重排结果；未配置则不制造重排曲线。重排失败直接记录错误，
不把 fail-open 的 RRF 结果算作重排成功。报告的共享检索耗时包含查询 Embedding、
两路检索和融合，不应当作某一路单独性能；应用结果缓存未参与。

## 本地运行

在项目目录中运行（Windows PowerShell）：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_live_retrieval.py
```

脚本读取现有 `.env`，要求 `RAG_BACKEND=milvus`、可访问的 Milvus 与真实 Embedding。
它会将**合成制度和合成问题**发送到已配置的 Embedding 接口，产生正常 API 用量，
不读取上传的私人文件，不调用问答大模型。

每次新建 `eval_rag_<随机ID>` Collection，使用与生产相同的分片、索引和检索代码；
结束后只清理本次评测 Collection，不删除生产 Collection 或 Docker 卷。
若进程被强制杀死可能留下评测 Collection；报告中的名称可用于核对后人工清理。

```powershell
# 开发集：供之后调参使用
.\.venv\Scripts\python.exe scripts/evaluate_live_retrieval.py --split dev --output docs/retrieval-dev.json
# 只跑少量问题做接口连通性检查；仍需先为合成语料建立向量索引
.\.venv\Scripts\python.exe scripts/evaluate_live_retrieval.py --max-cases 3 --output data/retrieval-smoke.json
```

默认报告：`docs/live-retrieval-results.json`。分题结果含证据命中、排名、指纹和错误；
按 dev/test、题型分别汇总。不要只取最好的一轮当成绩，也不要把合成小集结果当线上收益。

CI 中只执行 `--allow-hash` 的真实数据库协议回归，不使用 API 密钥；
这种报告的模式为 `real_milvus_hash_contract`，不是语义效果评测。

## 进一步学习与限制

先看 `app/rag/evidence.py`，理解编号和指纹；再看 `milvus_store.py` 的
`search_strategies` / `get_evidence`；最后看 `app/evaluation/retrieval.py` 的
`score_sources`，用一个双证据问题手算 Hit 和 Recall。

本次没有调优权重、接入新 Reranker、修改路由、增加拒答判定或页码坐标。
代码支持“引用存在且内容一致”的检查；引用蕴含关系、答案完整性仍需独立的回答评测。
旧评测结果继续保留，不能和新的真实语义结果直接作百分比提升比较。


## 本次实际运行结果（2026-10-20）

真实 Milvus + `qwen3.7-text-embedding`（1024 维），56 题全部完成，无请求错误；评测 Collection 已清理。52 道有证据题参与以下指标，4 道无答案题单独诊断。

| 检索策略 | Hit@1 | Recall@3 | MRR@5 |
| --- | ---: | ---: | ---: |
| BM25 | 84.62% | 94.23% | 0.8958 |
| 向量 | 94.23% | 100% | 0.9712 |
| RRF | 88.46% | 96.15% | 0.9167 |

RRF 保留现有权重 1.0 / 0.1，未调参；本次未开启 Reranker。总体向量优于 RRF，但独立 test 中 RRF 的 MRR@5 为 0.9583、向量为 0.9167，不能只凭总体分数断定融合没有价值。合成集只有 16 个证据单元，多题为同一事实的不同问法，需增加独立文档和人工复核后再做工程决策。

4 道无答案题均返回候选，这不是回答失败率，也不是拒答准确率；本次未调用生成模型。共享的 Embedding + BM25 + 向量 + RRF 检索平均 1087.29 ms、P95 1369.79 ms，不是每路单独耗时或端到端回答耗时。

SQLite 消息记录新增兼容迁移字段保存引用快照和警告，刷新会话仍可查看当时证据；进入下一轮模型的普通对话历史仍只读取角色和正文，不会自动重复塞入全部证据。
