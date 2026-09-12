"""Shared packet behavior across serialization, storage and context assembly."""

import json
from datetime import UTC, datetime

import pytest

from evals.helpers import ScriptedClient, make_waku
from waku.context_engineering import ContextPacket
from waku.context_engineering.notebook import NotebookStore, read_checkpoint
from waku.context_engineering.packet import token_length


def test_packet_json_roundtrip_and_independent_defaults():
    packet = ContextPacket(
        content="保留任务约束",
        relevance_score=2.0,
        task_id="task",
        source_refs=["message:1"],
        metadata=None,
    )
    other = ContextPacket(content="Another observation", relevance_score=-1.0)
    packet.metadata["status"] = "proposed"
    assert not other.metadata and not other.source_refs
    assert packet.relevance_score == 1.0 and other.relevance_score == 0.0
    assert packet.token_count == token_length(packet.content)
    assert ContextPacket.from_dict(json.loads(json.dumps(packet.to_dict()))) == packet
    assert isinstance(packet.timestamp, datetime) and packet.timestamp.tzinfo is not None


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_packet_rejects_nonfinite_relevance(score):
    with pytest.raises(ValueError):
        ContextPacket(content="Observation", relevance_score=score)


@pytest.mark.parametrize("tokens", [-1, 1.5, True])
def test_packet_rejects_invalid_token_counts(tokens):
    with pytest.raises(ValueError):
        ContextPacket(content="Observation", token_count=tokens)


def test_legacy_checkpoint_is_read_and_extended_as_packet():
    legacy = {
        "schema_version": 1,
        "task_id": "task",
        "checkpoint_id": "old",
        "parent_checkpoint_id": None,
        "generated_at": datetime.now(UTC).isoformat(),
        "objective": "Resolve outage",
        "constraints": ["No deployment"],
        "recent_turn_digest": "Observed a timeout",
    }
    old = read_checkpoint(legacy)
    assert isinstance(old, ContextPacket)
    assert old.content == "Observed a timeout"
    from waku.context_engineering.compaction import compile_continuation

    compilation = compile_continuation({}, [], [], old)
    assert not compilation.fallback
    assert compilation.packet.metadata["constraints"] == ["No deployment"]
    assert compilation.packet.metadata["parent_checkpoint_id"] == "old"


def test_legacy_notebook_entry_is_returned_as_packet(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    folder = store.root / "task" / "entries"
    folder.mkdir()
    legacy = {
        "id": "F-001",
        "kind": "finding",
        "title": "Latency",
        "body": "14ms",
        "status": "confirmed",
        "phase": "init",
        "author": "user",
        "confidence": "high",
        "tags": [],
        "source_refs": [{"type": "tool", "ref": "run:1"}],
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    (folder / "F-001.json").write_text(json.dumps(legacy))
    packet = store.read("task", "F-001")
    assert isinstance(packet, ContextPacket)
    assert packet.content == "14ms" and packet.source_refs == ["run:1"]
    assert packet.metadata["author"] == "user"
    assert store.append("task", {"kind": "finding", "content": "New observation"}).id == "F-002"


def test_notebook_relevance_cannot_displace_open_work_or_hide_metadata_cost(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    question = store.append(
        "task",
        {
            "kind": "question",
            "content": "Who owns the lock?",
            "relevance_score": 0.0,
            "metadata": {"status": "open"},
        },
    )
    low = store.append("task", {"kind": "finding", "content": "Cache low", "relevance_score": 0.1})
    high = store.append(
        "task", {"kind": "finding", "content": "Cache high", "relevance_score": 0.9}
    )
    huge = store.append(
        "task",
        {
            "kind": "finding",
            "content": "Cache",
            "relevance_score": 1.0,
            "metadata": {"details": "noise" * 1000},
        },
    )
    packets = store.resume_context("task", query="Cache", max_tokens=1100)
    assert [packet.id for packet in packets] == [question.id, high.id, low.id]
    assert huge.id not in {packet.id for packet in packets}
    assert token_length([packet.to_dict() for packet in packets]) <= 1100


def test_runtime_assembles_all_sources_as_packets(tmp_path):
    app = make_waku(
        tmp_path / "home",
        client=ScriptedClient([]),
        context_notebook=True,
    )
    task_id = app.context.task_id(app.session.session_id)
    try:
        saved = app.context.checkpoint(app.session, {"objective": "Investigate cache"})
        note = app.context.notebook.append(task_id, {"kind": "finding", "content": "Cache latency"})
        app.context.aggregate(
            app.session,
            ContextPacket(
                content="Inspect cache",
                kind="aggregation",
                task_id=task_id,
                metadata={"findings": [], "uncertainties": [], "failures": []},
            ),
        )
        messages = app.context.prepare(app.session, "Continue cache investigation")
        payload = json.loads(messages[0]["content"].split("\n", 1)[1])
        packets = [ContextPacket.from_dict(item) for item in payload["packets"]]
        assert isinstance(saved, ContextPacket) and isinstance(note, ContextPacket)
        assert {packet.kind for packet in packets} == {"continuation", "finding", "aggregation"}
        assert all(packet.task_id == task_id for packet in packets)
        assert messages[0]["role"] == "user"
    finally:
        app.close()
        app.conn.close()
