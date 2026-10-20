# 语义路由与 LangGraph 条件边

## 为什么改

旧 Supervisor 按关键词选择分支，固定填 0.64 / 0.76 等置信度。“查一下住宿上限”可能误进 tool；工具节点又做关键词选择，未匹配时默认统计文档。现在真实模型路径一次产出结构化的分支和工具参数，执行前校验，不再二次猜测工具。

## 执行过程

1. ChatService 从 SQLite 构建有界历史，保留当前 message 和追问补全 query。
2. `AgentWorkflow._supervisor_node` 等待 `SemanticRouter.decide`。分类器只接收当前问题、辅助 query、最近最多四条消息（每条最多1000字符），不接收长期记忆、检索全文或过去的工具结果。总的模型调用受8秒墙钟超时限制。
3. 使用现有 LLM 配置调用 JSON mode，temperature=0，限制输出长度。模型看到五个分支的定义、注册工具描述及参数 Schema。当前问题优先，历史只帮助理解指代。
4. Pydantic 校验 JSON、分支枚举、必需字段、严格布尔值和一致性；拒绝额外字段。模型要调用工具时，注册表校验工具白名单和参数；不允许未知参数。
5. 明确意图生成 RouteDecision；缺信息返回 status=clarification 和具体问题。超时、接口失败、非法输出分别标记 timeout/provider_error/invalid_output，并转 fallback，本轮不执行工具。取消请求继续传播 CancelledError。
6. 普通接口由 `StateGraph` 的 `add_conditional_edges` 调用 `_select_branch`，根据已校验 route 选择 rag/tool/plan/chat/fallback。LangGraph 负责调度，不负责理解用户意图。
7. 工具节点使用验证后的 tool_name/tool_arguments，执行边界再次校验；不重新关键词匹配。澄清直接返回问题，不调用回答模型。

SSE 接口仍沿用项目原来的手动流式分发实现，复用同一个异步决策函数和工具执行函数；它不是通过 `graph.astream` 运行。测试覆盖普通图调用和 SSE 的工具及澄清行为。不要在面试中把两条执行路径说成完全相同的 LangGraph 流式图。

## 分支边界

| 用户目标 | 分支 |
| --- | --- |
| 查内部规定、现有流程、项目文档 | rag |
| 制定新的多步骤方案 | plan |
| 执行注册表中已支持且参数齐全的动作 | tool |
| 概念讲解、问候等一般对话 | chat |
| 能力不支持、安全拒绝、缺信息澄清、分类故障 | fallback（用 status 区分原因） |

例如“报销要经过哪些步骤”查已有规定，属于 rag；“给我制定接入报销制度的实施方案”是 plan。“天气”没有对应工具，不能默认调用 doc_stats。路线工具仍是占位实现，未因路由升级变成真实地图服务。

## 不再伪造置信度

保留响应字段 confidence 以兼容旧记录，新决策返回 null。前端显示 source 和 status，不显示正确概率。模型自报分数没有校准，不能据此声称“低于0.8就一定不可靠”。当前通过明确的缺信息标记、能力边界和结构校验决定澄清/拒绝；不保证模型总能意识到自己的误判。要做概率阈值，需要独立标注集和校准实验。

模型的 JSON mode 只帮助格式约束，不代表分类正确，也不是完整的安全防护。允许的工具仍由代码注册，增加写操作时必须另外实现鉴权与审批，不能让路由结果授权操作。

## 配置与复现

- ROUTER_MODE=auto（默认）：真实 openai-compatible 配置走模型，fake provider 走规则，标记 rule_offline。
- ROUTER_MODE=llm：强制真实分类；缺少配置时明确 configuration_error，不偷偷退回规则。
- ROUTER_MODE=rule：显式使用旧规则用于离线练习，不代表语义路由效果。
- ROUTER_TIMEOUT_SECONDS=8：限制分类阶段；不改变检索/生成模型各自的超时。

沿用 LLM_PROVIDER / LLM_MODEL / LLM_BASE_URL / LLM_API_KEY，不需要新增密钥。启动中的服务需要重启或自动重载以加载代码。真实路径每轮增加一次分类请求；没有自动重试，也不缓存路由结果。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_semantic_router.py -q
.\.venv\Scripts\python.exe scripts/evaluate_routing.py
```

评测脚本只调用真实分类器，不调用业务工具、不检索私人文档、不生成回答；发送的是合成问题及合成历史。结果会覆盖 `docs/routing-evaluation-results.json`。

## 首轮真实验收：2026-10-20

使用当前配置模型 deepseek-flash。25题中24题同时符合预设分支、状态和适用的工具/参数要求，严格匹配率96%；无超时或接口错误。题集在 examples/routing_benchmark_v1.json，逐题结果在 docs/routing-evaluation-results.json。

失败题保留：“帮我规划一个月学习这个项目的计划”，预设 plan，模型要求先确认具体项目，输出 clarification。此处是保守澄清与预设标签的分歧，没有为了通过修改标签或重跑择优。

这25题为AI编写的合成回归集，覆盖知识/概念差异、计划/现有步骤差异、工具参数、模糊请求、历史动作不重放和提示注入。不是独立盲测，不代表企业线上96%准确率；工具参数的真实抽取验收目前只有基础计算样例。新增复杂意图应扩展人工审阅用例。

## 源码阅读顺序

1. app/schemas/chat.py：RouteDecision 的字段。
2. app/graph/semantic_router.py：RouteProposal、POLICY 和 decide。
3. app/graph/workflow.py：_supervisor_node → _select_branch → 对应节点。
4. app/tools/registry.py：validate_arguments 与 call_tool。
5. tests/test_semantic_router.py：结构错误、超时、取消、澄清、普通/SSE工具路径。


## 旧规则同题对照

`python scripts/compare_routing.py` 将同一25题的旧规则结果与保留的首轮真实模型结果配对。单独route命中率40%（10/25）→96%（24/25）；同时要求状态和适用工具参数正确时为36%（9/25）→96%。原始结果、输入与标签通过脚本核对，报告保存原始运行文件哈希。此集刻意覆盖关键词弱点，不是线上流量分布。
