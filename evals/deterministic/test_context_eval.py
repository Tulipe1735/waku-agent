"""Context eval uses actual Waku inputs; missing judges never become invented scores."""

import json

import pytest

from evals.helpers import ScriptedClient, response, text_block
from waku.context_engineering.eval import CaptureClient, deterministic_checks, validate_verdict


def test_capture_freezes_inputs_before_loop_mutation():
    client = CaptureClient(ScriptedClient([response([text_block("ok")])]))
    messages = [{"role": "user", "content": "original"}]
    client.messages.create(model="test", messages=messages, tools=[])
    messages[0]["content"] = "mutated"
    assert client.calls[0]["inputs"]["messages"][0]["content"] == "original"


def test_hard_gates_find_lost_state_and_invented_fact():
    case = {
        "gold": {
            "must_preserve": ["keep offline"],
            "must_not_claim": ["deployed"],
            "expected_open_questions": ["which version?"],
            "expected_next_actions": ["run tests"],
            "conflicts": ["sources disagree"],
        }
    }
    result = deterministic_checks(case, {"final_answer": "deployed"})
    assert set(result["critical_failures"]) == {
        "lost_constraint",
        "invented_fact",
        "dropped_open_question",
        "dropped_next_action",
        "hidden_conflict",
    }


def test_judge_schema_rejects_prose_extra_keys_bool_and_wrong_case():
    from waku.context_engineering.eval import DIMENSIONS

    verdict = {
        "case_id": "x",
        "variant": "baseline",
        "scores": dict.fromkeys(DIMENSIONS, 4),
        "critical_failures": [],
        "must_preserve_recall": 1.0,
        "unsupported_claim_rate": 0.0,
        "explanation": "gold constraint retained",
    }
    assert validate_verdict(json.dumps(verdict), "x", "baseline") == verdict
    for change in ({"extra": 1}, {"case_id": "other"}, {"scores": dict.fromkeys(DIMENSIONS, True)}):
        with pytest.raises(ValueError):
            validate_verdict(json.dumps(verdict | change), "x", "baseline")
    with pytest.raises(ValueError):
        validate_verdict("prose " + json.dumps(verdict), "x", "baseline")


def test_baseline_capture_is_actual_app_prompt_and_tool_result():
    from waku.ops.context_eval import capture_case

    case = {
        "case_id": "capture",
        "history": [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
        ],
        "task": "Read fixture",
        "tool_results": [{"output": "fixed evidence: retry failed"}],
    }
    artifact = capture_case(case, "baseline")
    assert len(artifact["main_inputs"]) == 2
    first, second = artifact["main_inputs"]
    assert "You are Waku" in first["system"]
    assert first["messages"][0]["content"] == "old request"
    assert second["messages"][-1]["content"][0]["type"] == "tool_result"
    assert second["messages"][-1]["content"][0]["content"] == "fixed evidence: retry failed"
    assert artifact["input_tokens"] is None
    assert artifact["cost"] is None


def test_unavailable_judge_still_checks_and_private_is_not_sent(tmp_path):
    from waku.ops.context_eval import run_context_eval

    dataset = tmp_path / "private.jsonl"
    dataset.write_text(json.dumps({"case_id": "private", "gold": {"must_preserve": ["secret"]}}))

    def capture(case, variant, **kwargs):
        return {"case_id": case["case_id"], "variant": variant, "final_answer": "lost"}

    def judge(*args, **kwargs):
        raise AssertionError("must never transmit private fixture")

    report = run_context_eval(
        dataset, ["baseline"], judge, capture=capture, output=tmp_path / "out"
    )
    row = report["records"][0]
    assert row["deterministic"]["passed"] is False
    assert len(row["judge_runs"]) == 2
    assert all("private fixture" in r["unavailable"] for r in row["judge_runs"])
    assert row["judge_variance"] is None
    assert "final_answer" not in row["artifact"]


def test_judge_runs_twice_and_reports_variance(tmp_path):
    from waku.context_engineering.eval import DIMENSIONS
    from waku.ops.context_eval import run_context_eval

    dataset = tmp_path / "public.jsonl"
    dataset.write_text(json.dumps({"case_id": "public", "privacy": "public-synthetic", "gold": {}}))

    def capture(case, variant, **kwargs):
        return {"case_id": case["case_id"], "variant": variant, "final_answer": "ok"}

    def judge(case, artifact, repeat):
        return {
            "case_id": case["case_id"],
            "variant": artifact["variant"],
            "scores": dict.fromkeys(DIMENSIONS, 2 + repeat),
            "critical_failures": [],
            "must_preserve_recall": 1,
            "unsupported_claim_rate": 0,
            "explanation": "span: ok",
        }

    report = run_context_eval(dataset, ["baseline"], judge, capture=capture)
    assert report["records"][0]["judge_variance"] == 0.25
    assert report["summary"]["baseline"]["average_score"] == 2.5


def test_disk_only_state_does_not_count_as_model_context():
    case = {"gold": {"must_preserve": ["private constraint"]}}
    assert not deterministic_checks(
        case, {"continuation": {"constraints": ["private constraint"]}}
    )["passed"]


def test_worker_fixture_runs_coordinator_and_reports_partial_failure():
    from pathlib import Path

    from waku.ops.context_eval import capture_case

    cases = [json.loads(line) for line in Path("evals/context.jsonl").read_text().splitlines()]
    case = next(c for c in cases if c["case_id"] == "workers-001")
    artifact = capture_case(case, "optimized")
    assert len(artifact["subagent_results"]) == 3
    assert artifact["subagent_failure_rate"] == pytest.approx(1 / 3)
    assert artifact["aggregation_coverage"] == 1
    assert artifact["aggregation"]["conflicts"]
    assert "Storage A loses writes" in str(artifact["main_inputs"])


def test_all_variants_capture_without_gold_injection():
    from pathlib import Path

    from waku.context_engineering.eval import VARIANTS
    from waku.ops.context_eval import capture_case

    case = json.loads(Path("evals/context.jsonl").read_text().splitlines()[0])
    case["gold"]["must_preserve"].append("GOLD_ONLY_SENTINEL")
    for variant in VARIANTS:
        artifact = capture_case(case, variant)
        assert "GOLD_ONLY_SENTINEL" not in str(artifact["main_inputs"])


def test_optimized_fixtures_meet_release_hard_gate():
    """Included by the existing deterministic release suite; no judge/key needed."""
    from pathlib import Path

    from waku.ops.context_eval import capture_case

    for line in Path("evals/context.jsonl").read_text().splitlines():
        case = json.loads(line)
        checks = deterministic_checks(case, capture_case(case, "optimized"))
        assert checks["passed"], (case["case_id"], checks["critical_failures"])


def test_conflict_constraint_alone_is_not_synthesis():
    case = {"subagents": [{"id": "a"}], "gold": {"conflicts": ["disagree"]}}
    artifact = {
        "main_inputs": [{"messages": [{"role": "user", "content": "Preserve disagreements"}]}]
    }
    result = deterministic_checks(case, artifact)
    assert result["conflict_visibility"] == 0
    assert "hidden_conflict" in result["critical_failures"]


def test_cli_missing_judge_key_still_runs_deterministic(tmp_path, monkeypatch):
    import sys

    from waku.loop import models
    from waku.ops.context_eval import main

    def unavailable(settings):
        raise SystemExit("No key")

    monkeypatch.setattr(models, "get_client", unavailable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context_eval",
            "--case-id",
            "chat-001",
            "--variants",
            "optimized",
            "--judge",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    main()
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["records"][0]["deterministic"]["passed"]
    assert len(report["records"][0]["judge_runs"]) == 2
