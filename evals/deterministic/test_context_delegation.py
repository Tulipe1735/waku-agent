"""Deterministic contracts for bounded, isolated exploratory workers."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from waku.context_engineering import ContextPacket
from waku.context_engineering.delegation import (
    DelegationCoordinator,
    DelegationPlan,
    Subtask,
    SubtaskBudget,
)
from waku.tools.registry import Tool, ToolRegistry


def worker_packet(subtask_id, **metadata):
    return ContextPacket(
        content="", id=subtask_id, task_id="parent", kind="subagent_result", metadata=metadata
    )


def task(name, **kwargs):
    return Subtask(id=name, parent_task_id="parent", objective="Investigate", **kwargs)


def test_partial_failure_dedup_and_conflicts():
    def worker(t, tools, control):
        if t.id == "bad":
            raise RuntimeError("worker failed")
        return worker_packet(
            subtask_id=t.id,
            findings=[
                {"statement": "same fact", "source_refs": [t.id]},
                {"statement": t.id, "key": "answer", "source_refs": [t.id]},
            ],
        )

    result = DelegationCoordinator(worker=worker).run(
        DelegationPlan("parent", [task("yes"), task("no"), task("bad")])
    )
    assert len(result.metadata["results"]) == 3
    assert len(result.metadata["findings"]) == 3
    assert result.metadata["conflicts"][0]["key"] == "answer"
    assert result.metadata["failures"]
    json.dumps(result.to_dict())


def test_timeout_cancel_and_concurrency():
    active = 0
    peak = 0
    lock = threading.Lock()

    def worker(t, tools, control):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            while not control.cancelled:
                time.sleep(0.002)
            control.check()
        finally:
            with lock:
                active -= 1

    start = time.monotonic()
    result = DelegationCoordinator(worker=worker).run(
        DelegationPlan(
            "parent",
            [task(str(i), budget=SubtaskBudget(timeout_seconds=0.03)) for i in range(6)],
            max_concurrency=2,
            timeout_seconds=0.2,
        )
    )
    assert time.monotonic() - start < 1
    assert peak <= 2
    assert all(r["metadata"]["status"] == "timeout" for r in result.metadata["results"])
    event = threading.Event()
    event.set()
    cancelled = DelegationCoordinator(worker=worker).run(
        DelegationPlan("parent", [task("one")]), cancel_event=event
    )
    assert cancelled.metadata["results"][0]["metadata"]["status"] == "cancelled"


def test_scoped_tools_deny_write_and_recursive_delegation():
    registry = ToolRegistry()
    for name in ("read", "write", "delegate_task"):
        registry.register(Tool(name, name, {}, lambda: "ok"))
    seen = []

    def worker(t, tools, control):
        seen.extend(s["name"] for s in tools.schemas())
        assert "unknown tool" in tools.execute("write", {})
        return worker_packet(subtask_id=t.id)

    coordinator = DelegationCoordinator(tools=registry, worker=worker, read_only_tools={"read"})
    result = coordinator.run(DelegationPlan("parent", [task("one", allowed_tools=["read"])]))
    assert seen == ["read"]
    assert result.metadata["results"][0]["metadata"]["status"] == "success"
    for name in ("write", "delegate_task"):
        with pytest.raises(ValueError):
            coordinator.run(DelegationPlan("parent", [task("one", allowed_tools=[name])]))


def test_invalid_plan_and_iteration_budget():
    with pytest.raises(ValueError):
        DelegationPlan("parent", [task("one")], max_concurrency=5)
    with pytest.raises(ValueError):
        SubtaskBudget(max_iterations=9)
    with pytest.raises(ValueError):
        DelegationPlan("other", [task("one")])


def test_loop_worker_budget_prevents_provider_call():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=10),
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(
                        {
                            "findings": [],
                            "evidence": [],
                            "uncertainties": [],
                            "failures": [],
                            "recommendation": "done",
                        }
                    ),
                )
            ],
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    coordinator = DelegationCoordinator(client=client, model="fake")
    result = coordinator.run(DelegationPlan("parent", [task("one")], max_tokens=1))
    assert not calls
    assert result.metadata["results"][0]["metadata"]["status"] == "budget_exceeded"
    result = coordinator.run(DelegationPlan("parent", [task("two")]))
    assert len(calls) == 1
    assert result.metadata["results"][0]["metadata"]["input_tokens"] == 10
    assert result.metadata["results"][0]["metadata"]["trace_id"]


def test_recursion_and_malformed_result_are_visible_failures():
    def worker(t, tools, control):
        if t.id == "recursive":
            DelegationCoordinator(worker=worker).run(DelegationPlan("parent", [task("child")]))
        return worker_packet(t.id, findings=[{"statement": "bad", "source_refs": [{}]}])

    result = DelegationCoordinator(worker=worker).run(
        DelegationPlan("parent", [task("recursive"), task("malformed")])
    )
    assert [r["metadata"]["status"] for r in result.metadata["results"]] == ["failed", "failed"]
    assert "recursive delegation" in result.metadata["failures"][0]


def test_global_budget_is_reserved_before_concurrent_calls():
    entered = []

    def worker(t, tools, control):
        control.reserve(60)
        entered.append(t.id)
        return worker_packet(t.id)

    result = DelegationCoordinator(worker=worker, token_cost=0.001).run(
        DelegationPlan("parent", [task("one"), task("two")], max_tokens=100, max_cost=1)
    )
    assert len(entered) == 1
    assert sorted(r["metadata"]["status"] for r in result.metadata["results"]) == [
        "budget_exceeded",
        "success",
    ]
    entered.clear()
    result = DelegationCoordinator(worker=worker, token_cost=0.001).run(
        DelegationPlan("parent", [task("one"), task("two")], max_tokens=1000, max_cost=0.1)
    )
    assert len(entered) == 1
    assert sorted(r["metadata"]["status"] for r in result.metadata["results"]) == [
        "budget_exceeded",
        "success",
    ]


def test_fake_provider_reuses_loop_and_stops_at_iteration_limit():
    count = []

    class Block:
        type, name, id, input = "tool_use", "read", "call", {}

        def model_dump(self):
            return {"type": self.type, "name": self.name, "id": self.id, "input": self.input}

    def create(**kwargs):
        count.append(kwargs)
        return SimpleNamespace(
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            content=[Block()],
        )

    registry = ToolRegistry()
    registry.register(Tool("read", "read", {}, lambda: "data"))
    coordinator = DelegationCoordinator(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="fake",
        tools=registry,
        read_only_tools={"read"},
        token_cost=0.000001,
    )
    result = coordinator.run(
        DelegationPlan(
            "parent",
            [task("one", allowed_tools=["read"], budget=SubtaskBudget(max_tokens=100000))],
            max_tokens=100000,
        )
    )
    assert len(count) == 8
    assert result.metadata["results"][0]["metadata"]["status"] == "failed"
    assert result.metadata["results"][0]["metadata"]["iterations"] == 8


def test_expired_worker_cannot_execute_tools_after_return():
    executed = []
    release = threading.Event()
    finished = threading.Event()
    registry = ToolRegistry()
    registry.register(Tool("read", "read", {}, lambda: executed.append(True) or "read"))

    def worker(t, tools, control):
        release.wait(1)
        try:
            assert "cancelled" in tools.execute("read", {})
            return worker_packet(t.id)
        finally:
            finished.set()

    coordinator = DelegationCoordinator(worker=worker, tools=registry, read_only_tools={"read"})
    result = coordinator.run(
        DelegationPlan(
            "parent",
            [task("one", allowed_tools=["read"], budget=SubtaskBudget(timeout_seconds=0.02))],
        )
    )
    assert result.metadata["results"][0]["metadata"]["status"] == "timeout"
    release.set()
    assert finished.wait(1)
    assert not executed
