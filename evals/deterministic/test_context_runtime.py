import json

from evals.helpers import ScriptedClient, make_waku, response, text_block
from waku.context_engineering.notebook import read_checkpoint
from waku.context_engineering.runtime import ContextRuntime


def latest_state(app):
    task = app.context.task_id(app.session.session_id)
    return read_checkpoint(app.context.notebook.read(task)["checkpoint"]["continuation"])


def test_resume_is_user_data_and_retrieval_still_runs(tmp_path):
    class Recorder(ScriptedClient):
        def _create(self, **kwargs):
            self.sent = json.loads(json.dumps(kwargs, default=str))
            return super()._create(**kwargs)

    client = Recorder(
        [
            response([text_block('{"retrieve":false,"query":"","reason":"test"}')]),
            response([text_block("Continue investigation")]),
        ]
    )
    app = make_waku(
        tmp_path / "home",
        client=client,
        context_notebook=True,
        consolidate_every=100,
    )
    app.context.checkpoint(
        app.session,
        {
            "objective": "Resolve outage",
            "constraints": ["Never deploy"],
            "open_questions": ["Why lock?"],
            "next_actions": ["Inspect owner"],
        },
        event="milestone",
    )
    app.respond("Continue")
    assert "Never deploy" not in client.sent["system"]
    assert "Never deploy" in str(client.sent["messages"])
    assert latest_state(app).metadata["objective"] == "Resolve outage"


def test_session_switch_cannot_leak_task_checkpoint(tmp_path):
    app = make_waku(tmp_path / "home", client=ScriptedClient([]), context_notebook=True)
    app.context.checkpoint(app.session, {"objective": "Private project A"}, event="milestone")
    app.session.start_new("B")
    messages = app.context.prepare(app.session, "Hello")
    assert "Private project A" not in str(messages)


def test_checkpoint_failure_keeps_answer_and_is_visible(tmp_path, monkeypatch):
    script = [
        response([text_block('{"retrieve":false,"query":"","reason":"test"}')]),
        response([text_block("The answer")]),
    ]
    app = make_waku(tmp_path / "home", client=ScriptedClient(script), context_notebook=True)
    monkeypatch.setattr(
        ContextRuntime, "checkpoint", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk"))
    )
    events = []
    result = app.respond(
        "Please summarize and handoff", observer=lambda kind, ev: events.append((kind, ev))
    )
    assert "The answer" in result.reply
    assert any(kind == "context_warning" for kind, ev in events)


def test_long_tool_result_retains_final_error_and_ids(tmp_path):
    from evals.helpers import tool_block

    class Recorder(ScriptedClient):
        def _create(self, **kwargs):
            self.sent = json.loads(json.dumps(kwargs, default=lambda obj: vars(obj)))
            return super()._create(**kwargs)

    client = Recorder(
        [
            response([text_block('{"retrieve":false,"query":"","reason":"test"}')]),
            response([tool_block("large", {})]),
            response([text_block("Observed")]),
        ]
    )
    app = make_waku(tmp_path / "home", client=client, context_notebook=True)
    from waku.tools.registry import Tool

    app.tools.register(
        Tool("large", "large", {"type": "object"}, lambda: "noise" * 10000 + "FINAL ERROR")
    )
    result = app.respond("Read result")
    observation = client.sent["messages"][-1]["content"][0]
    assert len(observation["content"].encode()) <= 2000
    assert "FINAL ERROR" in observation["content"] and observation["tool_use_id"] == "tu_1"
    assert len(result.tool_calls[0]["output"]) > 40000


def test_twenty_five_turn_handoff_preserves_task_state(tmp_path):
    client = ScriptedClient([])
    app = make_waku(
        tmp_path / "home",
        client=client,
        context_notebook=True,
        consolidate_every=100000,
    )
    app.context.checkpoint(
        app.session,
        {
            "objective": "Fix outage",
            "constraints": ["No deployment"],
            "open_questions": ["Why lock?"],
            "next_actions": ["Inspect owner"],
            "done": ["Collected logs"],
        },
    )
    for i in range(25):
        client._script.extend(
            [
                response([text_block('{"retrieve":false,"query":"","reason":"test"}')]),
                response([text_block(f"Observed step {i}")]),
            ]
        )
        app.respond(f"Investigate step {i}")
    messages = app.context.prepare(app.session, "Continue investigation")
    for text in ("Fix outage", "No deployment", "Why lock?", "Inspect owner", "Collected logs"):
        assert text in str(messages)
    current = latest_state(app)
    assert current.metadata["parent_checkpoint_id"]


def test_project_tool_bodies_are_not_copied_into_traces_or_memory(tmp_path):
    app = make_waku(tmp_path / "home", client=ScriptedClient([]), context_notebook=True)
    event = {
        "tool": "notebook_read",
        "args": {"task_id": "private"},
        "output": "PRIVATE_NOTE_SENTINEL",
    }
    app.tracer.event("tool", event)
    app.session.add_exchange("Read notes", "Read complete", tool_calls=[event])
    assert "PRIVATE_NOTE_SENTINEL" not in app.tracer.path.read_text()
    assert "PRIVATE_NOTE_SENTINEL" not in str(app.session.history)
    assert event["output"] == "PRIVATE_NOTE_SENTINEL"
