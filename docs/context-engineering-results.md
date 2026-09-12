# Context Engineering 验证报告

实际执行 Waku.respond 的离线输入捕获，8 个 fixture × 6 配置。fixture 提供相同历史与权威任务状态；优化路径将该状态保存成 checkpoint，gold 不进入模型输入。以下是字面/结构化确定性指标，不能替代真实模型回答质量。

| 配置 | 硬门通过 | 关键内容保留 | 未决问题保留 | 下一步保留 |
|---|---:|---:|---:|---:|
| baseline | 2/8 | 50.0% | 40.0% | 28.6% |
| optimized | 8/8 | 100.0% | 100.0% | 100.0% |
| continuation-only | 7/8 | 100.0% | 100.0% | 100.0% |
| notebook-only | 7/8 | 100.0% | 100.0% | 100.0% |
| subagent-only | 2/8 | 50.0% | 40.0% | 28.6% |
| all-on | 8/8 | 100.0% | 100.0% | 100.0% |

25 轮调试 fixture 的纯确定性 compaction（100 次）：p50 1.14 ms，p95 1.28 ms。原始历史 3013 UTF-8 字节，continuation 1639 字节，减少 45.6%。这是数据长度变化，不是 provider 的实测 token 节省；也不包含 system、工具定义和 notebook 的总输入开销。

worker fixture 运行真实协调器和 fake worker：3 个子任务，2 成功、1 明确超时失败，结果汇聚覆盖 100%，冲突显式可见。只开 continuation/notebook 的消融不能通过 worker conflict synthesis 门。

未运行付费 provider 或远端 judge。真实质量分数、unsupported-claim rate、账单成本、provider token 和 judge variance 为 null；每项仍记录两个 unavailable judge run。两次 fake judge 的严格 JSON 校验和方差计算另有确定性测试。当前不能声称真实任务成功率提升或成本控制在 baseline 的 1.5 倍以内。

最终完整测试：690 passed, 90 skipped in 40.30s。完整 Ruff 检查通过，新增 context 模块、评测入口和 loop 的 ty 类型检查通过，compileall 通过。wheel 与 sdist 已构建并检查包含新模块及 fixture。验证记录保存在 `.waku/evals/context/validation.json`。

初始测试曾有 1 个环境污染失败（本机 WAKU_LLM_TIMEOUT=300，测试期望默认 120）；使用隔离环境、清空 provider keys 并固定 timeout=120 后通过，不修改用户 .env。默认沙箱执行故障通过已授权的持续终端解决；打包下载依赖 TLS 失败后复用已安装 hatchling 离线构建。

复现步骤及设计限制见 [context-engineering.md](context-engineering.md)。本地完整数据位于 `.waku/evals/context/report.json`、`results.jsonl`、逐 case JSON 和 `compaction-benchmark.json`。
