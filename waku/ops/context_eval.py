"""Run context comparisons through the actual Waku app, offline by default.

python -m waku.ops.context_eval --dataset evals/context.jsonl --output .waku/evals/context
Use --live-agent / --judge explicitly to call the configured provider. Synthetic
capture replies have no model token/cost/quality measurements; reports say so.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from waku.context_engineering.eval import (
    JUDGE_PROMPT,
    JUDGE_VERSION,
    VARIANTS,
    CaptureClient,
    deterministic_checks,
    encoded,
    fixture_hash,
    summarize,
    validate_verdict,
)


class ReplayClient:
    """Exercises the real tool loop; never purports to answer the research task."""

    def __init__(self, case):
        self.case = case
        self.tool_index = 0
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        if "tools" not in kwargs:
            text = '{"retrieve":false,"query":"","reason":"isolated fixture"}'
            blocks = [SimpleNamespace(type="text", text=text)]
        elif self.tool_index < len(self.case.get("tool_results", [])):
            index = self.tool_index
            self.tool_index += 1
            blocks = [
                SimpleNamespace(
                    type="tool_use",
                    name="context_fixture_read",
                    id=f"fixture-{index}",
                    input={"index": index},
                )
            ]
        else:
            blocks = [
                SimpleNamespace(
                    type="text", text=("Offline capture only: no semantic answer generated.")
                )
            ]
        return SimpleNamespace(
            content=blocks,
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


def capture_case(case, variant, *, client=None, setup=None):
    from waku.app import Waku
    from waku.config import Settings
    from waku.tools.registry import Tool

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    flags = dict(
        zip(
            ("context_continuation", "context_notebook", "context_subagents"),
            VARIANTS[variant],
            strict=True,
        )
    )
    known = {f.name for f in dataclasses.fields(Settings)}
    if any(flags.values()) and not set(flags) <= known:
        raise RuntimeError("Context runtime flags are not installed")
    proxy = CaptureClient(client or ReplayClient(case), measured=client is not None)
    with tempfile.TemporaryDirectory(prefix="waku-context-eval-") as temporary:
        options: dict[str, Any] = {
            "home": Path(temporary) / ".waku",
            "api_key": "offline",
            "apple_calendar": False,
            "apple_tools": False,
            "google_calendar": False,
            "graph_workflows": False,
            "experimental": False,
            "gh_tool": False,
            "consolidate_every": 100000,
            "semantic_store": "sqlite",
            "episodic_store": "sqlite",
        }
        options.update({k: v for k, v in flags.items() if k in known})
        settings = Settings(**options)
        from waku.loop.models import PROVIDERS

        provider = PROVIDERS[settings.provider]
        settings.model = settings.model or provider.model
        settings.small_model = settings.small_model or provider.small_model
        app = Waku(settings=settings, client=proxy)
        events = []
        try:
            app.session.history = json.loads(encoded(case["history"]))
            if case.get("tool_results"):
                app.tools.register(
                    Tool(
                        "context_fixture_read",
                        "Read fixed evaluation tool output",
                        {
                            "type": "object",
                            "properties": {"index": {"type": "integer"}},
                            "required": ["index"],
                        },
                        lambda index: case["tool_results"][index]["output"],
                    )
                )
            runtime = getattr(app, "context", None)
            if runtime is not None and flags["context_notebook"]:
                task_id = runtime.task_id(app.session.session_id)
                runtime.notebook.create(task_id, title=case["state"]["objective"])
                for entry in case.get("notebook", {}).get("entries", []):
                    runtime.notebook.append(task_id, entry)
            if runtime is not None:
                runtime.checkpoint(app.session, state=case.get("state", {}), event="milestone")
            if runtime is not None and flags["context_subagents"] and case.get("subagents"):
                from waku.context_engineering.delegation import (
                    DelegationCoordinator,
                    DelegationPlan,
                    SubagentResult,
                    Subtask,
                    WorkerStopped,
                )

                task_id = runtime.task_id(app.session.session_id)
                fixtures = {item["id"]: item for item in case["subagents"]}

                def worker(task, tools, control):
                    item = fixtures[task.id]
                    if item.get("failure"):
                        raise WorkerStopped(item["failure"])
                    return SubagentResult(
                        task.id,
                        findings=[
                            {
                                "statement": text,
                                "key": "storage-durability",
                                "status": "hypothesis",
                                "source_refs": item["evidence"],
                            }
                            for text in item["findings"]
                        ],
                        evidence=item["evidence"],
                        recommendation="verify durability",
                    )

                coordinator = DelegationCoordinator(worker=worker, model="fixture-worker")
                plan = DelegationPlan(
                    task_id,
                    [
                        Subtask(
                            item["id"],
                            task_id,
                            "Compare storage evidence",
                            source_refs=item.get("evidence", []),
                        )
                        for item in case["subagents"]
                    ],
                )
                runtime.aggregate(app.session, coordinator.run(plan))
            if setup:
                setup(app, case, variant)
            continuation = notebook_checkpoint = None
            if runtime is not None:
                task_id = runtime.task_id(app.session.session_id)
                if flags["context_continuation"]:
                    latest = runtime.store.latest(task_id)
                    continuation = latest.to_dict() if latest else None
                if flags["context_notebook"]:
                    notebook_checkpoint = runtime.notebook.read(task_id).get("checkpoint")
            started = time.perf_counter()
            result = app.respond(
                case["task"],
                observer=lambda kind, ev: events.append(
                    {"kind": kind, "event": json.loads(encoded(ev))}
                ),
            )
            latency = (time.perf_counter() - started) * 1000
            # Persist artifact contents before the isolated home is cleaned up.
            artifacts = {}
            for root in (app.settings.home / "continuations", app.settings.home / "notebooks"):
                if root.exists():
                    for path in root.rglob("*"):
                        if path.is_file():
                            artifacts[str(path.relative_to(app.settings.home))] = path.read_text()
            main = [c["inputs"] for c in proxy.calls if "tools" in c["inputs"]]
            usage = [c["usage"] for c in proxy.calls if c["usage"] is not None]
            aggregation = runtime.last_aggregation if runtime else None
            workers = aggregation.get("results", []) if aggregation else []
            from waku.ops.pricing import price_for

            estimated_cost = (
                sum(
                    (
                        c["usage"]["input_tokens"]
                        * price_for(settings.provider, c["inputs"]["model"])[0]
                        + c["usage"]["output_tokens"]
                        * price_for(settings.provider, c["inputs"]["model"])[1]
                    )
                    / 1_000_000
                    for c in proxy.calls
                    if c["usage"] is not None
                )
                if usage
                else None
            )
            return {
                "case_id": case["case_id"],
                "variant": variant,
                "capture_mode": "live" if client else "offline-scripted",
                "provider_calls": proxy.calls,
                "main_inputs": main,
                "continuation": continuation,
                "notebook_checkpoint": notebook_checkpoint,
                "subagent_results": workers or None,
                "aggregation": aggregation,
                "local_artifacts": artifacts,
                "events": events,
                "final_answer": result.reply,
                "input_tokens": sum(u["input_tokens"] for u in usage) if usage else None,
                "output_tokens": sum(u["output_tokens"] for u in usage) if usage else None,
                "input_characters": sum(len(encoded(c)) for c in main),
                "cost": estimated_cost,
                "cost_basis": "existing pricing table estimate" if usage else None,
                "latency_ms": latency,
                "continuation_length": len(encoded(continuation)) if continuation else 0,
                "notebook_length": len(encoded(notebook_checkpoint)) if notebook_checkpoint else 0,
                "subagent_success_rate": sum(w["status"] == "success" for w in workers)
                / len(workers)
                if workers
                else None,
                "subagent_failure_rate": sum(w["status"] != "success" for w in workers)
                / len(workers)
                if workers
                else None,
                "aggregation_coverage": len(workers) / len(case["subagents"]) if workers else None,
                "missing_metrics": [
                    "cost uses existing table estimates when token usage is available",
                    "semantic answer quality unavailable in offline-scripted mode",
                    "subagent metrics unavailable unless workers actually ran",
                ],
            }
        finally:
            app.close()
            app.conn.close()


def run_context_eval(
    dataset,
    variants,
    judge=None,
    *,
    output=None,
    capture=capture_case,
    allow_private_remote=False,
    repeats=2,
    client=None,
):
    if not variants or set(variants) - VARIANTS.keys():
        raise ValueError("Unknown or empty variant list")
    if repeats < 2:
        raise ValueError("At least two independent judge runs are required")
    cases = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    records = []
    target = Path(output) if output else None
    if target:
        target.mkdir(parents=True, exist_ok=True)
    for case in cases:
        private = case.get("privacy") != "public-synthetic"
        if private and client is not None and not allow_private_remote:
            raise ValueError("Private fixtures may not be sent to a remote agent by default")
        for variant in variants:
            artifact = capture(case, variant, client=client)
            checks = deterministic_checks(case, artifact)
            runs: list[dict[str, Any]] = []
            for repeat in range(repeats):
                if judge is None:
                    runs.append({"unavailable": "judge not configured"})
                elif private and getattr(judge, "remote", True) and not allow_private_remote:
                    runs.append({"unavailable": "private fixture: remote judge refused"})
                else:
                    try:
                        verdict = judge(case, artifact, repeat=repeat)
                        runs.append(validate_verdict(encoded(verdict), case["case_id"], variant))
                    except Exception as exc:
                        runs.append({"unavailable": type(exc).__name__})
            values = [statistics.mean(r["scores"].values()) for r in runs if "scores" in r]
            record = {
                "case_id": case["case_id"],
                "variant": variant,
                "fixture_hash": fixture_hash(case),
                "deterministic": checks,
                "judge_runs": runs,
                "judge_variance": statistics.pvariance(values) if len(values) >= 2 else None,
                "judge_version": JUDGE_VERSION,
                "judge_prompt": JUDGE_PROMPT,
                "judge_model": getattr(judge, "model", None),
                "privacy": {
                    "private": private,
                    "remote_allowed": allow_private_remote,
                    "redaction": "hash-only artifacts" if private else "none",
                },
                "artifact": artifact
                if not private
                else {"sha256": fixture_hash(artifact), "characters": len(encoded(artifact))},
            }
            if target:
                name = f"{fixture_hash(case)[:16]}-{variant}.json"
                (target / name).write_text(encoded(record), encoding="utf-8")
                record["artifact_ref"] = name
            records.append(record)
    report = {
        "summary": summarize(records),
        "records": records,
        "limitations": [
            "Literal deterministic retention is not semantic task quality.",
            "Unavailable values are null; no cost or token estimates fabricated.",
        ],
    }
    if target:
        (target / "results.jsonl").write_text("".join(encoded(r) + "\n" for r in records))
        (target / "report.json").write_text(encoded(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/context.jsonl")
    parser.add_argument("--variants", default="baseline,optimized")
    parser.add_argument("--output", default=".waku/evals/context")
    parser.add_argument("--case-id")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--live-agent", action="store_true")
    parser.add_argument("--allow-private-remote", action="store_true")
    parser.add_argument("--hard-gate", action="store_true")
    args = parser.parse_args()
    client = judge = None
    if args.judge or args.live_agent:
        from waku.config import load_settings
        from waku.loop.models import get_client

        settings = load_settings()
        try:
            configured = get_client(settings)
            if args.live_agent:
                client = configured
            if args.judge:
                from waku.context_engineering.eval import ContextJudge

                judge = ContextJudge(client=configured, settings=settings)
        except (Exception, SystemExit) as exc:
            # Deterministic capture remains useful when provider setup fails.
            print(f"Provider unavailable ({type(exc).__name__}); running offline checks.")
    dataset = Path(args.dataset)
    with tempfile.TemporaryDirectory(prefix="context-filter-") as temporary:
        if args.case_id:
            cases = [json.loads(s) for s in dataset.read_text().splitlines() if s.strip()]
            cases = [c for c in cases if c["case_id"] == args.case_id]
            if not cases:
                parser.error("Unknown case id")
            dataset = Path(temporary) / "case.jsonl"
            dataset.write_text("\n".join(encoded(c) for c in cases))
        report = run_context_eval(
            dataset,
            args.variants.split(","),
            judge,
            output=args.output,
            client=client,
            allow_private_remote=args.allow_private_remote,
        )
    print(json.dumps(report["summary"], indent=2))
    if args.hard_gate and any(not r["deterministic"]["passed"] for r in report["records"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
