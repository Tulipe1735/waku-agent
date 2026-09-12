"""Structure layer: deterministic packet order, dedup and whole-packet budgets."""

import json

import pytest

from evals.helpers import ScriptedClient, make_waku
from waku.context_engineering.assembly import DEFAULT_TOKEN_BUDGET, assemble
from waku.context_engineering.notebook import read_checkpoint
from waku.context_engineering.packet import ContextPacket, token_length


def _packet(content, *, id, kind="finding", score=0.5, status=None):
    metadata = {"status": status} if status else {}
    return ContextPacket(
        content=content,
        kind=kind,
        id=id,
        task_id="task",
        relevance_score=score,
        metadata=metadata,
        source_refs=[],
    )


def test_assemble_orders_protected_work_then_evidence():
    continuation = _packet("digest", kind="continuation", id="c1", score=1.0)
    open_question = _packet("why?", kind="question", id="q1", score=0.5, status="open")
    open_action = _packet("fix it", kind="action", id="a1", score=0.4, status="open")
    aggregation = _packet("worker report", kind="aggregation", id="g1", score=0.6)
    high = _packet("high evidence", id="f1", score=0.9)
    low = _packet("low evidence", id="f2", score=0.1)
    result = assemble([low, high, aggregation, open_action, open_question, continuation])
    assert [packet.id for packet in result.packets] == ["c1", "q1", "a1", "g1", "f1", "f2"]
    assert result.dropped == []
    assert result.over_budget is False


def test_assemble_treats_closed_work_as_optional_evidence():
    open_question = _packet("open?", kind="question", id="q1", score=0.0, status="open")
    closed = _packet("closed", kind="question", id="q2", score=1.0, status="resolved")
    result = assemble([closed, open_question])
    assert [packet.id for packet in result.packets] == ["q1", "q2"]


def test_assemble_deduplicates_ids_and_normalized_content():
    keep = _packet("Keep this", kind="continuation", id="dup", score=1.0)
    same_id = _packet("Different body", id="dup")
    echo = _packet("  keep   THIS ", id="echo")
    result = assemble([same_id, echo, keep])
    assert [packet.id for packet in result.packets] == ["dup"]
    assert sorted(entry["reason"] for entry in result.dropped) == [
        "duplicate_content",
        "duplicate_id",
    ]


def test_assemble_budget_keeps_protected_and_drops_optional():
    continuation = _packet("state", kind="continuation", id="c1", score=1.0)
    question = _packet("open?", kind="question", id="q1", score=0.5, status="open")
    small = _packet("small evidence", id="f1", score=0.9)
    huge = _packet("x" * 4000, id="f2", score=0.1)
    budget = token_length([p.to_dict() for p in (continuation, question, small)]) + 10
    result = assemble([huge, small, question, continuation], max_tokens=budget)
    assert [packet.id for packet in result.packets] == ["c1", "q1", "f1"]
    assert result.dropped == [{"id": "f2", "kind": "finding", "reason": "budget"}]
    assert result.token_length <= budget
    assert result.over_budget is False


def test_assemble_reports_over_budget_instead_of_truncating_pinned():
    continuation = _packet("x" * 1000, kind="continuation", id="c1", score=1.0)
    note = _packet("evidence", id="f1")
    result = assemble([continuation, note], max_tokens=100)
    assert [packet.id for packet in result.packets] == ["c1"]
    assert result.over_budget is True
    assert result.token_length > 100
    assert result.dropped == [{"id": "f1", "kind": "finding", "reason": "budget"}]


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_assemble_rejects_invalid_budget(budget):
    with pytest.raises(ValueError):
        assemble([], max_tokens=budget)


def test_assemble_rejects_non_packets():
    with pytest.raises(TypeError):
        assemble(["not a packet"])


def test_assemble_empty_input_is_empty():
    result = assemble([])
    assert result.packets == [] and result.dropped == []
    assert result.token_length == token_length([])


def test_runtime_prepare_orders_and_budgets_packets(tmp_path):
    app = make_waku(
        tmp_path / "home",
        client=ScriptedClient([]),
        context_notebook=True,
    )
    task = app.context.task_id(app.session.session_id)
    try:
        app.context.checkpoint(app.session, {"objective": "Investigate cache"})
        question = app.context.notebook.append(
            task,
            {
                "kind": "question",
                "content": "Who owns the lock?",
                "metadata": {"status": "open"},
            },
        )
        aggregation = ContextPacket(
            content="x" * DEFAULT_TOKEN_BUDGET,
            kind="aggregation",
            task_id=task,
            metadata={"findings": [], "uncertainties": [], "failures": []},
        )
        app.context.aggregate(app.session, aggregation)
        finding = app.context.notebook.append(
            task, {"kind": "finding", "content": "Cache latency", "relevance_score": 0.9}
        )
        events = []
        messages = app.context.prepare(
            app.session, "What next?", notify=lambda kind, event: events.append((kind, event))
        )
        payload = json.loads(messages[0]["content"].split("\n", 1)[1])
        packets = [ContextPacket.from_dict(item) for item in payload["packets"]]
        latest = read_checkpoint(app.context.notebook.read(task)["checkpoint"]["continuation"])
        assert [packet.kind for packet in packets] == ["continuation", "question", "finding"]
        assert [packet.id for packet in packets] == [latest.id, question.id, finding.id]
        restore = next(event for kind, event in events if kind == "context_restore")
        assert restore["over_budget"] is False
        assert restore["packet_tokens"] <= DEFAULT_TOKEN_BUDGET
        assert restore["dropped_packets"] == [
            {"id": aggregation.id, "kind": "aggregation", "reason": "budget"}
        ]
    finally:
        app.close()
        app.conn.close()
