# Evaluation

评测分为两类：旧版离线链路回归，以及新增的真实 Milvus 语义检索评测。请勿混合两者的成绩。

## 新版真实检索评测

运行 `python scripts/evaluate_live_retrieval.py`，读取当前 Milvus / Embedding 配置，在独立 Collection 上评测 56 道合成问题并自动清理。实现、指标定义、标签与限制见 [检索与证据升级说明](retrieval-evidence-upgrade.md)，实际结果见 [运行报告](live-retrieval-results.json)。

2026-10-20 实测使用 Qwen Embedding、1024 维、RRF 权重 1.0 / 0.1，未启用 Reranker。52 道可回答问题的 Recall@3：BM25 94.23%、向量 100%、RRF 96.15%。另外 4 道无答案题均仍返回候选，说明召回结果不能代替拒答判断。此集规模小、为合成数据，不能推断线上效果。

以下介绍旧版回归。

## 评测目标

评测不是为了证明模型有多聪明，而是为了持续观察系统关键能力是否稳定：

- Supervisor 路由是否正确。
- RAG 分支是否能召回预期来源。
- Tool 分支是否能选对工具并成功执行。
- 请求平均耗时是否可接受。

## 当前评测集

默认评测集在：

```text
app/evaluation/dataset.py
```

当前包含 12 条样例：

| 类型 | 数量 | 目标 |
| --- | ---: | --- |
| RAG | 3 | 检查 route 和来源命中 |
| Tool | 3 | 检查 route、工具选择和执行结果 |
| Plan | 2 | 检查复杂任务是否进入规划分支 |
| Chat | 2 | 检查普通聊天是否进入 chat |
| Fallback | 2 | 检查边界请求是否进入 fallback |

## 指标

| 指标 | 说明 |
| --- | --- |
| route_accuracy | 实际 route 是否等于 expected_route |
| rag_hit_rate | RAG 来源标题是否命中预期来源 |
| tool_accuracy | 工具是否选对且执行成功 |
| average_latency_ms | 单次请求平均耗时 |

## 运行方式

```powershell
python scripts\run_evaluation.py
```

运行后会生成：

```text
docs/evaluation-results.json
```

## 当前限制

- 旧版离线回归的 LLM 是 `FakeLLMClient`，所以回答质量暂不做主观评分。
- 旧版回归使用本地 Hash Embedding，不代表真实语义检索效果。
- 当前评测集较小，适合作为回归门禁；业务验收仍需扩展问题类型和真实标注。

## 可扩展指标

- 回答引用正确率。
- 低置信度 fallback 率。
- SSE 首 token 延迟。
- Redis 缓存命中率。
- 真实 LLM 回答质量人工评分。

## 检索专项评测

运行：

```powershell
python scripts\evaluate_retrieval.py
```

脚本解析 `examples/documents/` 中的 8 页合成制度 PDF，以 8 个问题及预设的证据标题分别评测 BM25、Hash 向量检索和加权 RRF。结果写入 `docs/retrieval-evaluation-results.json`。

旧脚本依赖合成 PDF，输入缺失时需补齐；下表为保留的历史结果。旧代码将是否命中来源的指标命名为 Recall，实际为 Hit，不能与新版按证据覆盖计算的 Recall 直接比较。

| 策略 | Hit@1（旧名 Recall） | Hit@3（旧名 Recall） | MRR |
| --- | ---: | ---: | ---: |
| BM25 | 50.0% | 100.0% | 0.7292 |
| Hash 向量 | 25.0% | 50.0% | 0.3867 |
| 加权 RRF | 50.0% | 87.5% | 0.7014 |

这组结果没有证明 RRF 优于 BM25。它说明默认 Hash Embedding 缺少语义能力，即使把向量权重降至 0.1，仍可能扰动关键词侧的正确排序。保留该失败信号是为了指导下一步：接入真实中文 Embedding，按查询类型调整融合权重，再对 Reranker 做独立增量评测。当前 HTTP Reranker 只完成协议、排序和故障回退测试，没有第三方模型效果数据。
