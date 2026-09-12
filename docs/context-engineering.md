# Waku 上下文接力、Notebook 与受控子代理

本实现以三个领域对象为边界：`Continuation`、notebook entry/checkpoint、`SubagentResult`。沿用 `Waku.respond → Session.build_system → run_loop`，不引入 ContextPacket、prompt middleware 或新的框架。原有 retrieval gate、记忆保存、consolidation、graph/gather 继续运行。

## 开启与回退

三个开关默认关闭，可独立启用做消融比较：

```bash
WAKU_CONTEXT_CONTINUATION=1 WAKU_CONTEXT_NOTEBOOK=1 WAKU_CONTEXT_SUBAGENTS=1 python -m waku
```

关闭开关恢复原有输入路径。启用后注册 `context_checkpoint`、六个 `notebook_*` 工具和 `delegate_subtasks`。worker 默认没有工具；应用层只有显式注入可信只读工具或隔离工具工厂才能扩大权限。`delegate_subtasks` 不接受模型指定的并发、成本或写权限，默认最多 4 并发、每 worker 8 次迭代，禁止递归。

工具可以追加任务状态，但不能删除现有约束、未决问题或下一步，也不能把新结论标为 confirmed。用户确认目标变化、完成行动或确认事实后，调用受信任的 `app.context.checkpoint(app.session, state=...)` 更新权威状态。普通自然语言不会被确定性代码猜测成 confirmed；首次自动 checkpoint 保留原始任务目标，后续保存最近观察。主代理可用 `context_checkpoint` 显式整理里程碑。

模型整合是可注入的第二步，默认使用不产生额外模型调用的确定性路径。需要时复用当前 provider：

```python
from waku.context_engineering.compaction import model_summarizer
app.context.summarizer = model_summarizer(app.client, app.settings.small_model)
```

模型输出必须通过 schema、来源和状态一致性验证。失败保留上一有效 checkpoint，并在恢复上下文、trace 和回答中提示需要确认。

## 文件与数据格式

所有目录跟随 `settings.home`（默认 `.waku`），不同实例不会共享笔记目录：

```text
.waku/continuations/task-<session-hash>/
  <checkpoint-id>.json
  latest.json
  previous.json
.waku/notebooks/task-<session-hash>/
  README.md
  state.json
  entries/F-001.json
  checkpoints/00000001-<id>.md
.waku/evals/context/
  report.json
  results.jsonl
  <fixture-hash>-<variant>.json
```

Continuation 包含 PRD 的 objective/status/current_phase/done/decisions/facts/constraints/open_questions/next_actions/risks/artifacts/recent_turn_digest/omitted_history 等字段。parent ID 串起历史；notebook 和 continuation 互相引用 checkpoint ID。JSON 是可读的 YAML 子集，避免引入解析依赖。

Notebook 条目的 ID、作者和时间由代码分配；支持 finding、decision、question、action、risk，以及 task/phase/kind/status/tag/source 查询。`NotebookStore(..., author='user')` 标记人工写入，`read_only=True` 提供只读能力。手工编辑 `entries/*.json` 后，下次恢复使用当前条目；`diff` 展示相对旧 checkpoint 的变化。保留合法 schema、ID 和来源。

写入先生成临时文件再原子替换；checkpoint 保持不可变。损坏的 continuation 最新文件回退到已提交父版本，不会捡起失败事务的孤立文件。Notebook 读取会跳过损坏 checkpoint。文件写锁防止跨进程覆盖；进程崩溃残留 `.write-lock` 时，确认没有活跃 writer 后人工移除锁再重试。锁争用或写入失败不会阻断主回答。

## 上下文预算与数据权限

接力最多 3000、笔记恢复最多 4000、最近历史目标 6000、单项工具观察最多 2000、工具观察总量最多 6000。采用 UTF-8 字节数作为保守 token 上界，不宣称是 provider 的精确 tokenizer 计数。

先移除完整旧轮次，不拆 tool-call/result 配对。过长工具结果保留首尾及原内容 SHA-256，完整工具输出仍在运行记录中。最近用户消息、未完成工具配对不可截断；单个必保留轮次超预算时保留该轮并披露上界，而不是静默删掉任务要求。结构化必保留字段无法装入预算时，拒绝新 checkpoint 并保留旧版。过多 open notebook work 会明确报出恢复预算问题。

Continuation、notebook 和子代理结果仅作为普通用户消息中的项目数据进入上下文，不拼入 system prompt、工具权限或 graph router。新增项目工具的 trace 只记录哈希/长度；跨任务 chat consolidation 不接收它们的原始工具正文。主代理最终回答仍按原有会话规则保存。

## 汇聚与隔离

`DelegationCoordinator.run(plan)` 返回每个 worker 的成功、失败、超时、取消或预算耗尽状态，以及 token、保守成本估算、延迟、模型、trace ID 和 deadline。运行中每次 provider 调用前，在共享锁内预留总 token/成本，失败请求的预留不退还。`token_cost` 必须是部署方核实的每 token 价格上界，默认值是保守估算，不是账单金额。

`aggregate_results` 去重相同 statement，合并来源，对同一 key 的不同说法显式列出冲突。主任务通过 `app.context.aggregate(session, result)` 写 proposed findings/open questions，再创建 continuation。子代理没有直接修改主任务状态的接口。默认工具没有文件写或网络能力；需要写隔离时，可信 `scoped_tools_factory` 必须返回独立且受限的 registry。

Python 无法强制终止已经进入的任意同步工具代码：deadline 会撤销后续模型/工具访问，并停止启动排队 worker；已经运行的调用可能稍后返回。因此默认只授权有界只读工具，外部写操作应使用具备自身取消机制的受控实现。graph/gather 的固定拓扑和“只提案、不执行”规则保持原样。

## 评测复现

```bash
python -m waku.ops.context_eval --dataset evals/context.jsonl \
  --variants baseline,optimized,continuation-only,notebook-only,subagent-only,all-on \
  --output .waku/evals/context

python -m waku.ops.context_eval --case-id workers-001 --variants optimized --hard-gate
python -m pytest -q evals/deterministic/test_context_runtime.py \
  evals/deterministic/test_context_delegation.py evals/deterministic/test_context_eval.py
```

包含 25 轮调试、多阶段研究、工具重试、目标改变、worker 重复/冲突/失败、人工修改笔记、闲聊和超长工具输出。`CaptureClient` 在真实 Waku provider 边界冻结每次调用的 system/messages/tools；baseline 没有人工简化 prompt。不同 variant 使用相同 fixture 的历史和权威任务状态；gold 只用于评分，测试明确禁止 gold 注入模型输入。

默认离线 scripted provider 只用于执行真实输入组装、工具循环和协调器，不伪造自然语言任务回答。优化比较依赖 fixture 提供的权威 checkpoint 状态，不能据此证明模型能自行从任意长对话准确抽取全部状态。报告中的字面保留率也不是语义任务成功率。

真实模型与 judge 是显式选项（会调用已配置 provider）：

```bash
python -m waku.ops.context_eval --variants baseline,optimized --live-agent --judge
```

Judge 每项至少执行两次，严格校验 JSON 字段、0–4 分数、有限数值和重复键，输出方差。未配置、调用失败或返回非法 JSON 时记录 unavailable，同时继续确定性检查。私有 fixture 默认不得发送到远端 agent/judge，磁盘报告只保留 artifact 哈希与长度；公开测试数据须显式标记 `privacy: public-synthetic`。只有显式 `--allow-private-remote` 才能放行私有 fixture 的外发，并在报告中记录授权事实。

报告含逐 case/variant、质量维度及 delta、均值/p50/p95、保留率、worker 成败与汇聚覆盖、模型实际 usage（可用时）、现有 pricing 表的估算成本、延迟、长度、judge 版本/模型/方差及 fixture 哈希。离线 token/成本/语义评分为 null。judge 输出不会写入用户长期记忆。

`test_optimized_fixtures_meet_release_hard_gate` 已加入原有 deterministic suite，因此既有 release gate 自动执行关键上下文检查，无需网络或 judge。P1 UI 本轮采用 CLI/JSONL 交付。
