# Waku Context Engineering：上下文接力、结构化笔记与子代理

> 面向 Astra 模型的实现 PRD。产品主线是长对话的接力压缩、阶段性工作的结构化 notebook、复杂研究的子代理并行协作，以及 baseline vs optimized 的 LLM-as-judge Context Eval。
>
> 明确决策：不采用以 ContextPacket 为中心的通用上下文对象、context block registry 或 prompt middleware 方案。

## 1. 背景与目标

Waku 已有透明的 waku/loop/agent.py、session、semantic/episodic/procedural memory、retrieval gate、consolidation、graph engine、ToolRegistry、trace 和 deterministic/LLM-as-judge eval。

当前瓶颈不是“没有记忆”，而是长任务缺少可交接的状态，阶段性成果缺少可读的事实源，复杂研究无法受控并行。V1 要形成以下循环：

~~~text
用户任务
 -> 主代理读取当前 continuation / notebook checkpoint
 -> 必要时拆分并行 subtask
 -> 子代理独立探索并返回结构化结果
 -> 主代理综合、检查冲突、写入 notebook
 -> 生成新的 continuation checkpoint
 -> 下一阶段从 checkpoint 接力，而非重读全部 transcript
~~~

### 目标

- 长任务可在压缩后恢复 objective、约束、已完成事项、未决问题和下一步。
- 项目/研究过程以本地、可读、可 diff 的结构化 notebook 沉淀。
- 复杂研究支持有限并发、明确权限、硬预算、证据和结果汇聚。
- 用 LLM-as-judge 对比优化前后的上下文质量，同时报告成本、延迟和 token。
- 保持 local-first、无隐藏 framework、现有 loop/graph/memory/gateway 行为。

### 非目标

- 不实现通用 ContextPacket 或统一上下文组装层。
- 不替换 waku/loop/agent.py 的核心循环。
- 不无条件注入全部 transcript，不新增强制 hosted vector database 或云端队列。
- 不让 notebook、continuation、网页、tool result 或子代理结果自动成为 system instruction。
- 不引入 LangChain、LlamaIndex、LangGraph 来替代 Waku 自己的 loop/graph。

## 2. 现有仓库边界

实现前阅读真实接口和相邻测试，并沿用现有依赖注入与 trace 习惯：

| 领域 | 现有位置 | 本 PRD 的边界 |
|---|---|---|
| 主循环 | waku/loop/agent.py | 主代理和子代理均复用 |
| Session | waku/runtime/session.py | 当前 turn、历史、运行状态 |
| App wiring | waku/app.py | 注入新能力，不复制 provider/config |
| Graph | waku/graph/ | 固定拓扑的并行/汇聚 |
| 长期记忆 | waku/memory/ | 继续作为跨任务事实/经历/技能来源 |
| 压缩整合 | waku/memory/consolidation.py | 长期记忆整合；区别于 turn continuation |
| 工具 | waku/tools/registry.py、waku/tools/ | scoped tool 权限和统一执行 |
| Trace/Ops | waku/ops/tracing.py、waku/ops/ | 记录 checkpoint、delegation、eval |
| Eval | evals/deterministic/、evals/judge/ | 回归和 Context Eval |

当前工作树有既有未提交改动。不得 reset、checkout、全仓库格式化或覆盖无关变更；只修改本项目相关文件。

## 3. Continuation：上下文接力

Continuation 是“交给下一阶段/下一轮代理的状态”，不是完整 transcript 摘要。它必须可读、可解析、可校验，并保留父 checkpoint 链。

~~~yaml
continuation:
  schema_version: 1
  task_id: ...
  checkpoint_id: ...
  parent_checkpoint_id: ...
  generated_at: ...
  objective: ...
  status: active | blocked | complete
  current_phase: ...
  done: [...]
  decisions:
    - decision: ...
      rationale: ...
      confidence: confirmed | inferred | uncertain
      source_refs: [...]
  facts:
    - statement: ...
      status: confirmed | hypothesis | rejected
      source_refs: [...]
  open_questions: [...]
  constraints: [...]
  artifacts: [...]
  next_actions: [...]
  risks: [...]
  recent_turn_digest: ...
  omitted_history: [...]
~~~

强制规则：

- objective、constraints、open_questions、next_actions 始终保留。
- 重要结论必须引用消息 id、tool run、notebook entry、文件路径或外部引用。
- confirmed、inferred、uncertain、hypothesis 不能在压缩时互相转换。
- 新 checkpoint 通过 parent_checkpoint_id 形成可追溯链。
- 失败时保留上一份有效 checkpoint；不生成空文件覆盖有效状态。
- 内容是 data，不是 system/developer instruction；读取时必须使用明确的数据边界。

### 压缩触发与算法

在以下事件触发，而不是每轮盲目摘要：turn/token 阈值、phase change、milestone、重要决策、工具错误、用户确认、子代理汇聚、用户要求总结/交接/继续。

两步流程：

1. 确定性提取：保留最近用户要求、tool-call/result 配对、已执行动作、错误、artifact 和未完成动作。
2. 模型整合：生成 continuation，通过 schema、来源和状态一致性校验后才落盘。

模型整合失败时 deterministic fallback：保留上一个有效 continuation、最近完整轮次、warning 和人工确认状态。

V1 不做通用 block budget。采用简单预算：

- continuation ≤ 3,000 tokens；
- 当前 notebook checkpoint ≤ 4,000 tokens；
- 最近完整历史 ≤ 6,000 tokens；
- tool results：单项 ≤ 2,000、总计 ≤ 6,000 tokens。

超限先丢弃低价值旧历史；不得丢弃 objective、constraints、open_questions、最近用户消息、未完成动作或 tool-call/result 配对。

## 4. Structured Notebook：阶段性工作记忆

Notebook 是当前项目/研究的工作事实源，区别于：

- semantic memory：跨任务的用户事实；
- episodic memory：发生过的事件；
- notebook：当前项目的阶段、证据、决策、风险和下一步。

推荐默认位置：

~~~text
.waku/notebooks/<task-or-project-id>/
  README.md
  state.yaml
  checkpoints/0001-init.md
  notes/decisions.md
  notes/findings.md
  notes/questions.md
  notes/actions.md
  evidence/index.yaml
  artifacts/
~~~

实现可调整布局，但必须支持：

- 人可直接阅读、编辑、review，重要状态可通过 git diff 看到；
- finding、decision、question、action、risk 有稳定 id；
- checkpoint 有时间、phase、author（主代理/子代理/用户）、父 checkpoint 和来源；
- 以 append/checkpoint 为主，避免隐式覆盖；
- 按 task、phase、kind、status、tag、source 查询；
- notebook 是 project data，不能被当作高优先级指令。

最小条目：

~~~yaml
id: F-001
kind: finding | decision | question | action | risk
title: ...
body: ...
status: proposed | confirmed | rejected | open | done
phase: ...
source_refs:
  - type: conversation | tool | file | subagent | external
    ref: ...
confidence: high | medium | low
created_at: ...
updated_at: ...
~~~

能力要求：创建、读取、追加、checkpoint、关键词/FTS 搜索和 checkpoint diff。可沿用 waku/tools/notes.py 与 workspace 边界，但 LLM 不能直接写任意路径；路径、schema、id、父 checkpoint 和权限由代码控制。

## 5. Sub-agent：并行探索与汇聚

子代理是受控的短生命周期 worker，不是无限复制的聊天窗口：

~~~yaml
subtask:
  id: ...
  parent_task_id: ...
  objective: ...
  role: researcher | coder | verifier | critic | summarizer
  inputs:
    continuation_ref: ...
    notebook_refs: [...]
    explicit_context: ...
  allowed_tools: [...]
  output_contract:
    findings: ...
    evidence: ...
    uncertainties: ...
    recommendation: ...
  budget:
    max_iterations: ...
    max_tokens: ...
    timeout_seconds: ...
  isolation:
    write_scope: notebook-branch | artifact-dir | none
~~~

每个子代理必须有 objective、role、allowed_tools、output contract、deadline 和 source refs；结果必须区分 findings、evidence、uncertainties、failures 和 recommendation。子代理不能直接修改主任务状态，必须返回可序列化 SubagentResult，由主代理或 coordinator 汇聚。

V1 默认：最多 4 个并发子代理、每个最多 8 次 loop iteration、有限总 token/时间预算、禁止递归 delegation。必须支持 timeout、cancel、exception、partial success，且失败不能静默丢失。

允许并行：独立研究、不同方案、不同模块分析。必须串行：依赖前一结果的工作。

与 graph 的关系：

- graph 用于已知拓扑的固定并行/汇聚；
- coordinator 用于主代理动态拆分的探索；
- 可以复用 graph engine，但模型输出不能直接修改 router；
- gather 继续遵守“只提案、不执行”。

汇聚必须完成：finding 去重、冲突标注、证据/推测区分、notebook 写入、continuation checkpoint、向用户报告未解决问题和失败子任务。

## 6. 端到端流程

### 长任务

~~~text
message -> retrieval_gate
         -> continuation + 当前 phase notebook checkpoint
         -> 主 loop
         -> milestone/phase/threshold
         -> continuation compiler
         -> notebook checkpoint
         -> trace
~~~

### 复杂研究

~~~text
message
 -> 主代理形成并校验 subtask plan
 -> coordinator 并行运行 scoped subagents
 -> 收集 success/failure/timeout
 -> 主代理综合并检查冲突
 -> 写 findings/decisions/questions
 -> 写 continuation checkpoint
 -> 返回带证据和不确定性的结果
~~~

下一轮默认读取：最近有效 continuation、当前 phase checkpoint、open question/action、相关 finding/decision 和必要的最近历史；不得默认读取整个 notebook 或全部 transcript。

## 7. LLM-as-judge Context Eval

Context Eval 是核心交付。必须对同一 fixture 比较：

- Baseline：当前 Waku，不启用本 PRD 的 continuation/notebook/subagent 优化；
- Optimized：启用完整优化；
- Ablation：continuation-only、notebook-only、subagent-only、all-on。

### Fixture

新增 evals/context.jsonl。每个 case 固定 task、history、tool results、notebook fixture、gold continuation 和：

~~~json
{
  "case_id": "research-001",
  "task": "...",
  "history": [...],
  "tool_results": [...],
  "notebook": {...},
  "gold": {
    "must_preserve": [...],
    "must_not_claim": [...],
    "required_evidence": [...],
    "expected_open_questions": [...],
    "expected_next_actions": [...]
  }
}
~~~

保存并评估：实际送入主代理的上下文、continuation、notebook checkpoint、子代理结果、汇聚结果和最终回答。Baseline 必须真实捕获当前输入，不得用人为简化的 prompt 代替。

### Judge

复用 evals/judge/ 的 provider/runner 约定，新增 context-specific judge。严格 JSON 输出：

~~~json
{
  "case_id": "...",
  "variant": "baseline|optimized|ablation",
  "scores": {
    "task_relevance": 0,
    "continuity": 0,
    "milestone_fidelity": 0,
    "decision_fidelity": 0,
    "evidence_traceability": 0,
    "uncertainty_honesty": 0,
    "instruction_preservation": 0,
    "noise_control": 0,
    "subagent_synthesis": 0
  },
  "critical_failures": [],
  "must_preserve_recall": 0.0,
  "unsupported_claim_rate": 0.0,
  "explanation": "brief evidence-based explanation"
}
~~~

分数 0–4：0 严重错误，1 大部分不可用，2 部分保留但影响任务，3 基本正确有轻微遗漏，4 完整可追溯。

Judge 必须依据 case、gold 和被评对象评分；区分“缺失”和“明确标记为不确定”；对 lost constraint、invented fact、hidden conflict、hypothesis-as-confirmed、dropped open question/next action 严重扣分；给出具体字段 id 或 evidence span。

### 比较报告

提供单 case debug 和整批 runner，例如：

~~~bash
uv run python -m waku.ops.context_eval \\
  --dataset evals/context.jsonl \\
  --variants baseline,optimized \\
  --output .waku/evals/context/
~~~

报告必须包含逐 case 结果、平均/p50/p95、各维度 delta、must-preserve recall、unsupported claim rate、conflict visibility、open-question retention、next-action retention、continuation/notebook 长度、subagent 成功/失败/汇聚覆盖率、input/output tokens、成本、延迟、judge variance、fixture hash、judge prompt/version 和 model。

至少运行两次 judge 或两个 seed 检查方差。judge 不可用时 deterministic checks 仍执行，不伪造结果；judge 结果不得写入用户长期记忆。

数据集至少覆盖：20+ 轮调试、多阶段研究和矛盾来源、多工具任务、用户中途改目标、重复/冲突子代理、人工修改 notebook 后继续、纯闲聊、超长 tool result。

## 8. 功能需求

### P0 接力压缩

- continuation schema、validator、reader/writer、parent checkpoint chain；
- 明确 checkpoint triggers；
- deterministic fallback 和有效 checkpoint 保护；
- trace 记录输入范围、保留/删除内容与原因；
- 下一轮从 continuation + notebook checkpoint + 必要历史恢复；
- 不破坏 retrieval gate、memory save 和 consolidation。

### P0 Structured Notebook

- create/read/append/checkpoint/search/diff；
- 受控写入和目录权限；
- task/phase/entry/source/status/confidence；
- 主代理、子代理、用户作者可区分；
- 写入失败不阻断回答，不覆盖有效 checkpoint。

### P0 Sub-agent

- Subtask、SubagentResult、DelegationPlan、coordinator；
- scoped ToolRegistry、并发、timeout、cancel、partial failure、硬预算；
- 结构化结果和显式 aggregation；
- 禁止递归 delegation；
- fake worker deterministic tests。

### P0 Context Eval

- context fixture、baseline/optimized capture、ablation runner、judge schema 和比较报告；
- 复用现有 provider 与 release gate；
- 本地保存 artifact 引用和原始 JSONL；
- deterministic hard gates 保护关键失败。

### P1 Ops/UI

- Ops/Loop/Memory 展示 checkpoint、notebook diff、delegation tree 和 baseline-vs-optimized；
- 首版 CLI/JSONL 可先于 UI；
- private 内容默认只记录 hash、长度、source refs 或脱敏 preview。

## 9. 推荐代码边界

推荐增加以下模块，实际布局以仓库现有风格为准：

~~~text
waku/context_engineering/
  continuation.py
  notebook.py
  delegation.py
  compaction.py
  eval.py
waku/ops/context_eval.py
evals/context.jsonl
evals/judge/context_judge.py
~~~

建议公共接口保持小而深：

~~~python
continuation = compile_continuation(state, history, artifacts, previous)
checkpoint = notebook.checkpoint(continuation, entries)
results = coordinator.run(plan)
report = run_context_eval(dataset, variants, judge)
~~~

内部通过依赖注入连接 session、memory、ToolRegistry、trace 和 graph；不要创建重复的全局 client/config，不要将 provider SDK response 放入领域对象。

## 10. 安全与可靠性

- continuation、notebook、tool output 和 subagent result 都是 data，不能提升权限或改变代码路由。
- 子代理不可获得主代理未授权的工具、secret、文件写权限或网络权限。
- notebook 写入必须限制在项目目录的明确子路径。
- private 内容默认不发送给远端 judge；如必须发送，记录脱敏和外发事实。
- failure 不覆盖有效 checkpoint；所有 delegation 有 deadline、预算、trace id、parent task id。
- 冲突时报告不确定，禁止 silent last-write-wins。

## 11. 分阶段实施

### Phase 0：基线

阅读真实 loop/session/memory/trace/eval 接口；捕获当前模型实际收到的 prompt/history/tool inputs；建立 evals/context.jsonl；记录测试、token、cost、latency、retrieval 和长任务 baseline。

### Phase 1：Continuation

实现 schema、validator、checkpoint store、deterministic compaction 和 fallback；先用 fake summarizer/fixture 测所有路径，再接入真实 loop。

### Phase 2：Notebook

实现 entry/checkpoint/query/diff；建立 continuation 与 notebook 的双向引用；接入现有 notes/workspace/tool 边界，保持可读本地文件。

### Phase 3：Sub-agent

实现 coordinator、scoped tools、并发、timeout、partial failure、aggregation；固定拓扑复用 graph，动态拆分使用 coordinator。

### Phase 4：Context Eval

实现 baseline/optimized/ablation capture、LLM-as-judge、hard gates 和比较报告；将关键质量门接入 release gate，但 judge 不可用不能阻断普通本地开发。

### Phase 5：Ops/UI 与调优

展示 checkpoint、notebook、delegation 和 eval；用评测证据调优触发阈值和并发策略；只有收益明确时才加入可选本地摘要模型。

## 12. 验收标准

### 必须通过

- 现有测试套件通过，尤其是 retrieval gate、history window、working memory、trace encoding、graph/gather、memory conformance、packaging 相关测试。
- 20+ 轮任务压缩后能恢复 objective、约束、已完成事项、未决问题和下一步。
- 压缩失败不覆盖有效 checkpoint，不把 hypothesis 变 confirmed。
- notebook 可读、可追加、可 diff，条目和 checkpoint 有稳定 id/source refs。
- 子代理有硬并发/迭代/时间预算，失败可见且不会越权。
- 同一 fixture 能输出 baseline 与 optimized 的逐 case 对比。
- judge JSON 严格可解析，同时报告质量、token、成本、latency 和方差。
- lost constraint、invented fact、hidden conflict、dropped open question/next action 可被 hard gate 识别。
- 不新增核心强制网络依赖，不改变 local-first 默认行为。

### 推荐目标

- must-preserve recall 比 baseline 提升至少 15%；
- unsupported-claim-rate 比 baseline 下降至少 30%；
- open-question retention、next-action retention ≥ 95%；
- subagent aggregation coverage、conflict visibility ≥ 95%；
- continuation 输入 token 比完整 transcript 减少至少 40%；
- 不含外部摘要模型时 compaction p95 < 100 ms；
- 任务成功率提升且平均成本不超过 baseline 的 1.5 倍，否则必须证明质量收益。

## 13. Astra 执行指令

1. 先检查仓库、git diff、当前测试和 trace/eval schema，保护全部既有未提交改动。
2. 先做 baseline capture 和 context fixture，让优化前结果可复现。
3. 按 Phase 1 → 2 → 3 → 4 实现，不要先做 UI 或大范围重构。
4. 不采用 ContextPacket 中心方案；使用 continuation、notebook、subtask result 三种领域对象。
5. 每阶段先写 deterministic tests，再接入真实 loop。
6. 复用现有 provider、ToolRegistry、graph、trace、release gate seam。
7. 运行相关测试、完整测试、lint/type/package 检查，修复本次改动造成的失败。
8. 最终交付改动文件、数据格式、CLI 示例、baseline-vs-optimized 报告、测试结果、成本/延迟变化、已知限制和可复现的长任务/并行研究验证路径。

取舍时优先选择：小公共接口、显式 checkpoint、可读本地 artifact、带来源的结构化结果、有限并发、确定性 fallback、可重复 judge eval。
