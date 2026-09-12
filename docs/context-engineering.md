# Waku 上下文接力、Notebook 与受控子代理

上下文候选信息统一使用 `ContextPacket`，接力状态、笔记条目、子代理结果和汇聚结果通过 `kind` 区分。沿用 `Waku.respond → Session.build_system → run_loop`；现有 retrieval gate、记忆保存、consolidation、graph/gather 继续运行。

## 统一数据结构

`ContextPacket` 是一个普通 dataclass，只有九个字段，没有子类、类型注册表或 middleware。

| 字段 | 含义与默认值 |
|---|---|
| `content: str` | 信息正文，唯一必填项 |
| `timestamp: datetime` | 信息时间，默认当前 UTC；JSON 中为 ISO 字符串 |
| `token_count: int` | 正文 token 数；默认 0 时自动按 UTF-8 字节估算 |
| `relevance_score: float` | 相关性，默认 0.5，限制在 0–1；不代表事实可信度 |
| `metadata: dict[str, Any]` | 可选业务详情，默认独立空字典，也接受 `None` |
| `id: str` | 稳定引用 ID，默认自动生成 |
| `kind: str` | 信息种类，默认 `context` |
| `task_id: str` | 所属任务，独立候选可留空 |
| `source_refs: list[str]` | 消息、工具、文件或外部来源引用，默认空列表 |

```python
from waku.context_engineering import ContextPacket

packet = ContextPacket(
    content="日志显示请求在等待数据库锁。",
    kind="finding",
    task_id="investigation",
    source_refs=["tool:read-log:1"],
    relevance_score=0.9,
    metadata={"status": "hypothesis"},
)
saved = packet.to_dict()
restored = ContextPacket.from_dict(saved)
```

只把跨来源通用的属性放到顶层。接力的目标、约束、下一步和父 checkpoint ID 放在 `metadata`；笔记的标题、phase、作者、状态和 tags 也放在 `metadata`；子代理的 findings、evidence、uncertainties、failures、status 和调用统计同样放在 `metadata`，建议下一步放在 `content`。无需提供与当前来源无关的属性。

`compile_continuation(...).packet`、`NotebookStore.append/read(entry_id)/search/resume_context` 中的信息项，以及 `DelegationCoordinator.run(...)` 均使用这个类型。恢复时 `ContextRuntime.prepare` 将选中的信息序列化为一个 `packets` 列表，作为普通用户消息里的项目数据。Notebook 的目录索引和 checkpoint 文件仍负责记录存储关系；它们的条目和接力内容使用 packet 格式。

Notebook 写入示例（ID、任务、时间、作者由存储层分配）：

```python
store.create("investigation")
packet = store.append("investigation", {
    "kind": "finding",
    "content": "锁等待尚未解除。",
    "source_refs": ["tool:read-log:2"],
    "metadata": {"status": "proposed", "phase": "debug"},
})
```

筛选仍使用现有预算和简单规则：先保留未完成的问题/行动，再按相关性选取匹配证据和当前阶段笔记。预算检查计算整个序列化 packet（包括 metadata），不会只相信正文的 `token_count`。

`ContextRuntime.prepare` 把所有来源交给 `assemble(packets, max_tokens=6000)` 统一渲染：按接力状态、未完成工作、汇聚结果、其他证据的固定顺序排列，按 `id` 和归一化正文去重，并对整个序列化列表计算预算。受保护的 packet 永不因预算被丢；可选证据超出预算时记录 `{"id", "kind", "reason"}` 到 `context_restore` 事件的 `dropped_packets`。受保护内容自身超预算时 `over_budget=True`，原样保留而不是静默截断，评分只作为同层排序提示。

## 开启与回退

两个开关默认关闭，可独立启用做消融比较：

```bash
WAKU_CONTEXT_NOTEBOOK=1 WAKU_CONTEXT_SUBAGENTS=1 python -m waku
```

关闭开关恢复原有输入路径。Notebook 是任务恢复的唯一存储：启用后注册 `context_checkpoint`、六个 `notebook_*` 工具和 `delegate_subtasks`。worker 默认没有工具；应用层只有显式注入可信只读工具或隔离工具工厂才能扩大权限。`delegate_subtasks` 不接受模型指定的并发、成本或写权限，默认最多 4 并发、每 worker 8 次迭代，禁止递归。

工具可以追加任务状态，但不能删除现有约束、未决问题或下一步，也不能把新结论标为 confirmed。用户确认目标变化、完成行动或确认事实后，调用受信任的 `app.context.checkpoint(app.session, state=...)` 更新权威状态。普通自然语言不会被确定性代码猜测成 confirmed；首次自动 checkpoint 保留原始任务目标，后续保存最近观察。主代理可用 `context_checkpoint` 显式整理里程碑。

模型整合是可注入的第二步，默认使用不产生额外模型调用的确定性路径。需要时复用当前 provider：

```python
from waku.context_engineering.compaction import model_summarizer
app.context.summarizer = model_summarizer(app.client, app.settings.small_model)
```

模型输出必须符合 ContextPacket 格式，并通过接力状态、来源和确定性标记的一致性验证。失败保留上一有效 checkpoint，并在恢复上下文、trace 和回答中提示需要确认。

## 文件与数据格式

所有目录跟随 `settings.home`（默认 `.waku`），不同实例不会共享笔记目录：

```text
.waku/notebooks/task-<session-hash>/
  README.md
  state.json
  entries/F-001.json
  checkpoints/00000001-<id>.md
```

`kind="continuation"` 的 packet 在 `metadata` 中保留目标、约束、决策、事实、未决问题和下一步，正文保留最近观察；它作为任务状态嵌入 notebook checkpoint。notebook 文件链的 `parent_checkpoint_id` 串起 checkpoint 历史，packet 内的 `metadata.parent_checkpoint_id` 串起状态历史。已有 version-1 continuation 和旧笔记条目在读取时转换为 packet，不改写旧文件；后续写入使用新格式。JSON 避免引入解析依赖。

Notebook 条目的 ID、作者和时间由代码分配；支持 finding、decision、question、action、risk，以及 task/phase/kind/status/tag/source 查询。`NotebookStore(..., author='user')` 标记人工写入，`read_only=True` 提供只读能力。手工编辑 `entries/*.json` 后，下次恢复使用当前条目；`diff` 展示相对旧 checkpoint 的变化。保留 packet 格式、ID、任务归属和来源；正文改为编辑 `content`，作者和状态在 `metadata` 中。读取人工编辑后的正文时会重新估算 token 数。

写入先生成临时文件再原子替换；checkpoint 保持不可变，且是唯一的恢复来源。损坏的最新 checkpoint 会被跳过，读取回退到链上最后一个有效版本，不会捡起失败事务的孤立文件。文件写锁防止跨进程覆盖；进程崩溃残留 `.write-lock` 时，确认没有活跃 writer 后人工移除锁再重试。锁争用或写入失败不会阻断主回答。

## 上下文预算与数据权限

接力最多 3000、笔记恢复最多 4000、最近历史目标 6000、单项工具观察最多 2000、工具观察总量最多 6000。采用 UTF-8 字节数作为保守 token 上界，不宣称是 provider 的精确 tokenizer 计数。

先移除完整旧轮次，不拆 tool-call/result 配对。过长工具结果保留首尾及原内容 SHA-256，完整工具输出仍在运行记录中。最近用户消息、未完成工具配对不可截断；单个必保留轮次超预算时保留该轮并披露上界，而不是静默删掉任务要求。结构化必保留字段无法装入预算时，拒绝新 checkpoint 并保留旧版。过多 open notebook work 会明确报出恢复预算问题。

Continuation、notebook 和子代理结果仅作为普通用户消息中的项目数据进入上下文，不拼入 system prompt、工具权限或 graph router。新增项目工具的 trace 只记录哈希/长度；跨任务 chat consolidation 不接收它们的原始工具正文。主代理最终回答仍按原有会话规则保存。

## 汇聚与隔离

`DelegationCoordinator.run(plan)` 返回 `kind="aggregation"` 的 ContextPacket；`metadata.results` 包含各 worker 的序列化 packet，记录每个 worker 的成功、失败、超时、取消或预算耗尽状态，以及 token、保守成本估算、延迟、模型、trace ID 和 deadline。运行中每次 provider 调用前，在共享锁内预留总 token/成本，失败请求的预留不退还。`token_cost` 必须是部署方核实的每 token 价格上界，默认值是保守估算，不是账单金额。

`aggregate_results` 去重相同 statement，合并来源，对同一 key 的不同说法显式列出冲突。主任务通过 `app.context.aggregate(session, result)` 写 proposed findings/open questions，再创建 continuation。子代理没有直接修改主任务状态的接口。默认工具没有文件写或网络能力；需要写隔离时，可信 `scoped_tools_factory` 必须返回独立且受限的 registry。

Python 无法强制终止已经进入的任意同步工具代码：deadline 会撤销后续模型/工具访问，并停止启动排队 worker；已经运行的调用可能稍后返回。因此默认只授权有界只读工具，外部写操作应使用具备自身取消机制的受控实现。graph/gather 的固定拓扑和“只提案、不执行”规则保持原样。

## 测试

上下文测试全部离线运行，不调用 provider：

```bash
python -m pytest -q evals/deterministic/test_context_packet.py \
  evals/deterministic/test_context_assembly.py \
  evals/deterministic/test_context_continuation.py \
  evals/deterministic/test_context_runtime.py \
  evals/deterministic/test_context_notebook.py \
  evals/deterministic/test_context_delegation.py
```

覆盖 packet 序列化与预算选择、compaction 的确定性整轮裁剪与来源校验、运行时恢复（含失败回退）、notebook 存储与工具边界、子代理预算/隔离/汇聚。模型整合是可注入的第二步，默认走不产生额外模型调用的确定性路径；注入的 summarizer 无法改写受保护字段或凭空确认结论。真实模型质量评测不在本仓库的确定性 gate 内。
