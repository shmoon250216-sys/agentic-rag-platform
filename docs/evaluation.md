# Evaluation

当前系统已经加入基础评测体系，用来量化 Agentic RAG 的核心链路质量。

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

- 当前 LLM 仍是 `FakeLLMClient`，所以回答质量暂不做主观评分。
- 当前 RAG 使用本地 Hash Embedding，不代表真实语义检索效果。
- 当前评测集较小，适合作为回归门禁；业务验收仍需扩展问题类型和真实标注。

## 可扩展指标

- 回答引用正确率。
- 低置信度 fallback 率。
- SSE 首 token 延迟。
- Redis 缓存命中率。
- 真实 LLM 回答质量人工评分。
