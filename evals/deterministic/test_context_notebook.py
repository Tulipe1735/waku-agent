"""Notebook regression checks exercise storage, recovery and the tool boundary."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from waku.context_engineering.notebook import NotebookStore
from waku.tools.notebook import make_tools
from waku.tools.registry import ToolRegistry


def finding(**overrides):
    entry = {
        "kind": "finding",
        "title": "Measured",
        "body": "Latency 12ms",
        "status": "confirmed",
        "source_refs": ["run:1"],
        **overrides,
    }
    return {
        "kind": entry.pop("kind"),
        "content": entry.pop("body"),
        "source_refs": entry.pop("source_refs"),
        **{key: entry.pop(key) for key in ("id",) if key in entry},
        "metadata": entry,
    }


def test_append_search_human_edit_and_diff(tmp_path):
    store = NotebookStore(tmp_path, author="user")
    store.create("research", phase="measure")
    entry = store.append("research", finding(tags=["perf"]))
    assert entry.id == "F-001" and entry.metadata["author"] == "user"
    first = store.checkpoint("research", source_refs=[{"type": "conversation", "ref": "msg:1"}])
    path = tmp_path / ".waku/notebooks/research/entries/F-001.json"
    edited = json.loads(path.read_text())
    edited["content"] = "Human correction 14ms"
    path.write_text(json.dumps(edited))
    second = store.checkpoint("research", phase="review")
    assert second["parent_checkpoint_id"] == first["checkpoint_id"]
    assert "Human correction 14ms" in store.diff(
        "research", first["checkpoint_id"], second["checkpoint_id"]
    )
    assert (
        store.search(
            task_id="research",
            phase="measure",
            kind="finding",
            status="confirmed",
            tag="perf",
            source="run:1",
        )[0].id
        == "F-001"
    )
    assert "Human correction" in str(store.resume_context("research", query="correction"))


def test_checkpoint_failure_retains_previous(tmp_path, monkeypatch):
    store = NotebookStore(tmp_path)
    store.create("task")
    old = store.checkpoint("task")
    monkeypatch.setattr(
        store, "_atomic_write", lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        store.checkpoint("task")
    assert store.read("task")["checkpoint"]["checkpoint_id"] == old["checkpoint_id"]


@pytest.mark.parametrize("task_id", ["../outside", "/tmp/escape", "a/b", "..", "a\\b"])
def test_path_escape_rejected(tmp_path, task_id):
    with pytest.raises(ValueError):
        NotebookStore(tmp_path).create(task_id)


def test_symlink_and_readonly_rejected(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".waku/notebooks/escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        store.create("escape")
    ro = NotebookStore(tmp_path, read_only=True)
    assert ro.read("task")["task_id"] == "task"
    with pytest.raises(PermissionError):
        ro.append("task", finding())
    assert {tool.name for tool in make_tools(ro)} == {
        "notebook_read",
        "notebook_search",
        "notebook_diff",
    }


def test_invalid_entries_and_spoofed_author_rejected(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    for invalid in [
        finding(kind="instruction"),
        finding(source_refs=[]),
        finding(confidence="certain"),
        finding(author="user"),
        finding(id="F-999"),
    ]:
        with pytest.raises(ValueError):
            store.append("task", invalid)


def test_concurrent_append_and_tools(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(lambda _: store.append("task", finding()), range(12)))
    assert len({entry.id for entry in entries}) == 12
    registry = ToolRegistry()
    for tool in make_tools(store):
        registry.register(tool)
    assert len(registry.schemas()) == 6
    assert "Error" in registry.execute("notebook_append", {"task_id": "../bad", "entry": finding()})


def test_recovery_is_bounded_and_prioritizes_open_work(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    store.append("task", finding(body="old noise " * 10000))
    store.append(
        "task",
        {
            "kind": "question",
            "content": "Need owner",
            "metadata": {"title": "Unresolved", "status": "open"},
        },
    )
    store.checkpoint("task")
    context = store.resume_context("task", max_tokens=500)
    assert "Need owner" in str(context)
    assert len(json.dumps([packet.to_dict() for packet in context]).encode()) <= 500


def test_corrupt_checkpoint_falls_back_and_failed_temp_is_ignored(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    first = store.checkpoint("task")
    second = store.checkpoint("task")
    folder = tmp_path / ".waku/notebooks/task/checkpoints"
    (folder / f"{second['checkpoint_id']}.md").write_text("corrupt")
    (folder / ".pending-failure").write_text("half written")
    assert store.read("task")["checkpoint"]["checkpoint_id"] == first["checkpoint_id"]


def test_internal_symlink_does_not_read_or_write_outside(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".waku/notebooks/task/entries").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        store.append("task", finding())
    assert not list(outside.iterdir())


def test_custom_storage_and_current_phase(tmp_path):
    store = NotebookStore(tmp_path, storage_root=tmp_path / "instance-b/notebooks")
    store.create("task")
    store.checkpoint("task", phase="review")
    assert store.append("task", finding()).metadata["phase"] == "review"
    assert not (tmp_path / ".waku").exists()


def test_stable_id_not_reused_after_human_deletion(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    first = store.append("task", finding())
    store.checkpoint("task")
    (tmp_path / f".waku/notebooks/task/entries/{first.id}.json").unlink()
    assert store.append("task", finding()).id == "F-002"


def test_other_process_lock_prevents_overwriting(tmp_path):
    store = NotebookStore(tmp_path)
    store.create("task")
    lock = store.root / "task" / ".write-lock"
    lock.write_text("another writer")
    with pytest.raises(FileExistsError):
        store.append("task", {"kind": "finding", "content": "race"})
    assert not store.read("task")["entries"]
