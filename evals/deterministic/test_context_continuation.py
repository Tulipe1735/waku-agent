from dataclasses import replace

import pytest

from waku.context_engineering.compaction import compact_history, compile_continuation
from waku.context_engineering.continuation import Continuation, ContinuationStore


def test_checkpoint_chain_and_failure_preserves_valid_state(tmp_path):
    store = ContinuationStore(tmp_path)
    first = Continuation(
        task_id="task",
        objective="Debug deployment",
        constraints=["Do not deploy"],
        open_questions=["Why timeout?"],
        next_actions=["Inspect logs"],
    )
    store.save(first)
    second = replace(first, checkpoint_id="second", parent_checkpoint_id=first.checkpoint_id)
    store.save(second)
    assert store.latest("task") == second
    with pytest.raises(ValueError):
        store.save(replace(second, checkpoint_id="third", parent_checkpoint_id="missing"))
    assert store.latest("task") == second


def test_compiler_rejects_invented_confirmation_and_keeps_previous():
    first = Continuation(
        task_id="task",
        objective="Research",
        facts=[{"statement": "Cache issue", "status": "hypothesis", "source_refs": ["message:1"]}],
        constraints=["No deployment"],
    )
    state = {"task_id": "task", "objective": "Research"}

    def corrupt(payload):
        candidate = first.to_dict()
        candidate["facts"][0]["status"] = "confirmed"
        return candidate

    result = compile_continuation(
        state, [{"role": "user", "content": "Continue"}], [], first, summarizer=corrupt
    )
    assert result.continuation == first
    assert result.fallback and result.warnings


def test_history_keeps_complete_tool_pair_and_latest_request():
    history = [
        {"role": "user", "content": "old" * 9000},
        {"role": "assistant", "content": "old response"},
        {"role": "user", "content": "Inspect"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "evidence"}],
        },
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Continue without deployment"},
    ]
    retained, omitted = compact_history(history, max_tokens=500)
    assert retained[-1]["content"] == "Continue without deployment"
    assert "t1" in str(retained)
    assert omitted


def test_compilation_retains_critical_fields_over_twenty_turns():
    history = []
    for i in range(25):
        history.extend(
            [
                {"role": "user", "content": f"Inspect step {i}"},
                {"role": "assistant", "content": f"Observed step {i}"},
            ]
        )
    state = {
        "task_id": "long",
        "objective": "Resolve outage",
        "constraints": ["No restart"],
        "open_questions": ["Why lock?"],
        "next_actions": ["Inspect owner"],
        "done": ["Collected logs"],
    }
    result = compile_continuation(state, history, [], None)
    assert result.continuation.objective == "Resolve outage"
    assert result.continuation.constraints == ["No restart"]
    assert result.continuation.open_questions == ["Why lock?"]
    assert result.continuation.next_actions == ["Inspect owner"]


@pytest.mark.parametrize("field", ["done", "artifacts", "risks", "status", "current_phase"])
def test_summarizer_cannot_drop_task_state(field):
    original = Continuation(
        task_id="task", objective="Fix", done=["Read logs"], artifacts=["log.txt"], risks=["Race"]
    )

    def summarize(payload):
        candidate = payload["state"]
        candidate[field] = [] if isinstance(candidate[field], list) else "complete"
        return candidate

    result = compile_continuation({}, [], [], original, summarize)
    assert result.fallback and result.continuation == original


def test_summarizer_cannot_swap_real_source_for_another_real_source():
    original = Continuation(
        task_id="task",
        objective="Fix",
        facts=[{"statement": "Race", "status": "hypothesis", "source_refs": ["message:0"]}],
    )

    def summarize(payload):
        candidate = payload["state"]
        candidate["facts"][0]["source_refs"] = ["message:1"]
        return candidate

    result = compile_continuation(
        {},
        [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
        [],
        original,
        summarize,
    )
    assert result.fallback


def test_latest_corrupt_checkpoint_recovers_committed_parent_not_orphan(tmp_path):
    store = ContinuationStore(tmp_path)
    first = Continuation(task_id="task", objective="Fix")
    store.save(first)
    second = replace(first, checkpoint_id="second", parent_checkpoint_id=first.checkpoint_id)
    store.save(second)
    (tmp_path / "task/second.json").write_text("broken")
    orphan = replace(second, checkpoint_id="orphan")
    import json

    (tmp_path / "task/orphan.json").write_text(json.dumps(orphan.to_dict()))
    assert store.latest("task") == first


def test_latest_corrupt_manifest_recovers_previous_committed_head(tmp_path):
    store = ContinuationStore(tmp_path)
    first = Continuation(task_id="task", objective="Fix")
    store.save(first)
    store.save(replace(first, checkpoint_id="second", parent_checkpoint_id=first.checkpoint_id))
    (tmp_path / "task/latest.json").write_text("broken")
    assert store.latest("task") == first


def test_deterministic_digest_preserves_user_request_and_tool_failures():
    history = [
        {"id": "msg-1", "role": "user", "content": "Inspect without changing files"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "run-1", "name": "inspect", "input": {"path": "log"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "run-1",
                    "is_error": True,
                    "content": "Permission denied",
                }
            ],
        },
    ]
    result = compile_continuation({"task_id": "task", "objective": "Inspect"}, history, [])
    digest = result.continuation.recent_turn_digest
    assert "msg-1" in digest and "without changing files" in digest
    assert "run-1" in digest and "Permission denied" in digest


def test_large_optional_digest_does_not_force_fallback():
    result = compile_continuation(
        {"task_id": "task", "objective": "Inspect"},
        [{"role": "user", "content": "要求" * 9000}],
        [],
    )
    from waku.context_engineering.continuation import token_length

    assert not result.fallback
    assert token_length(result.continuation.to_dict()) <= 3000
    assert result.retained_history[0]["content"] == "要求" * 9000
