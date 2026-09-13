"""
Gather

Structured note-taking

Project working notes: editable files, explicit immutable checkpoints, no prompts.

JSON is used as the human-editable YAML subset so local storage needs no parser
package. Every checkpoint is one atomic file: a failed write cannot move HEAD.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from .packet import ContextPacket, token_length

KINDS = {"finding": "F", "decision": "D", "question": "Q", "action": "A", "risk": "R"}
STATUSES = {"proposed", "confirmed", "rejected", "open", "done"}
SOURCE_TYPES = {"conversation", "tool", "file", "subagent", "external"}
_LOCK = RLock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Invalid notebook identifier")
    return value


def _sources(value: Any) -> list[dict]:
    if not isinstance(value, list):
        raise TypeError("source_refs must be a list")
    for source in value:
        if (
            not isinstance(source, dict)
            or set(source) != {"type", "ref"}
            or source["type"] not in SOURCE_TYPES
            or not isinstance(source["ref"], str)
            or not source["ref"].strip()
        ):
            raise ValueError("Invalid source reference")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("Invalid task/checkpoint identifier")
    return value


def validate_checkpoint(packet: ContextPacket) -> None:
    """Validate task state at the checkpoint boundary, not in the shared packet."""
    safe_id(packet.task_id)
    safe_id(packet.id)
    if packet.kind != "continuation":
        raise ValueError("Expected a continuation packet")
    state = packet.metadata
    if state.get("parent_checkpoint_id") is not None:
        safe_id(state["parent_checkpoint_id"])
    if not isinstance(state.get("objective"), str) or not state["objective"].strip():
        raise ValueError("Objective is required")
    if state.get("status", "active") not in ("active", "blocked", "complete"):
        raise ValueError("Invalid task status")
    if not isinstance(state.get("current_phase", "init"), str):
        raise TypeError("Phase must be text")
    for name in (
        "done",
        "open_questions",
        "constraints",
        "artifacts",
        "next_actions",
        "risks",
        "omitted_history",
    ):
        items = state.get(name, [])
        if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
            raise ValueError(f"{name} must be a list of strings")
    for name, label, key, statuses in (
        ("decisions", "decision", "confidence", ("confirmed", "inferred", "uncertain")),
        ("facts", "statement", "status", ("confirmed", "hypothesis", "rejected")),
    ):
        items = state.get(name, [])
        if not isinstance(items, list):
            raise TypeError(f"{name} must be a list")
        for item in items:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get(label), str)
                or not item[label].strip()
                or item.get(key) not in statuses
            ):
                raise ValueError(f"Invalid {name} entry")
            refs = item.get("source_refs")
            if (
                not isinstance(refs, list)
                or not refs
                or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
            ):
                raise ValueError("Conclusions require source references")
    json.dumps(packet.to_dict(), allow_nan=False)


def read_checkpoint(data: dict) -> ContextPacket:
    """Read current packets and existing version-1 checkpoints without rewriting files."""
    if "content" in data:
        packet = ContextPacket.from_dict(data)
    else:
        state = dict(data)
        version = state.pop("schema_version", None)
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported legacy continuation schema")
        packet = ContextPacket(
            id=state.pop("checkpoint_id"),
            task_id=state.pop("task_id"),
            timestamp=datetime.fromisoformat(state.pop("generated_at")),
            content=state.pop("recent_turn_digest", "") or state.get("objective", ""),
            kind="continuation",
            metadata=state,
        )
    validate_checkpoint(packet)
    return packet


class NotebookStore:
    def __init__(
        self,
        project_root: Path,
        *,
        author: str = "main",
        read_only: bool = False,
        storage_root: Path | None = None,
    ):
        self.project_root = Path(project_root).absolute()
        self.root = (
            Path(storage_root).absolute()
            if storage_root is not None
            else self.project_root / ".waku" / "notebooks"
        )
        if not self.root.is_relative_to(self.project_root):
            raise ValueError("Notebook storage must be inside project_root")
        if not isinstance(author, str) or not author.strip():
            raise ValueError("author is required")
        self.author = author
        self.read_only = read_only
        self._safe(self.root)

    def _safe(self, path: Path) -> Path:
        if not path.is_relative_to(self.project_root):
            raise ValueError("Notebook path escapes project")
        for item in (path, *path.parents):
            if item.is_symlink():
                raise ValueError("Notebook paths cannot contain symlinks")
            if item == self.project_root:
                break
        return path

    def _task(self, task_id: str) -> Path:
        return self._safe(self.root / _identifier(task_id))

    def _writable(self) -> None:
        if self.read_only:
            raise PermissionError("Notebook is read-only")

    def _atomic_write(self, path: Path, content: str) -> None:
        self._writable()
        self._safe(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._safe(path.parent)
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            self._safe(path)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _load(self, path: Path) -> dict:
        value = json.loads(self._safe(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Notebook document must be an object")
        return value

    @contextmanager
    def _process_lock(self, task_id):
        # Fail visibly on another writer instead of assigning duplicate IDs.
        # A crash leaves this lock for explicit review; never steal a live lock.
        folder = self._task(task_id)
        folder.mkdir(parents=True, exist_ok=True)
        lock = self._safe(folder / ".write-lock")
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.write(fd, str(os.getpid()).encode())
            yield
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)

    def create(self, task_id: str, title: str = "", phase: str = "init") -> dict:
        self._writable()
        with _LOCK, self._process_lock(task_id):
            folder = self._task(task_id)
            state = folder / "state.json"
            if state.exists():
                return self._load(state)
            value = {
                "schema_version": 1,
                "task_id": task_id,
                "title": title or task_id,
                "phase": phase,
                "author": self.author,
                "created_at": _now(),
            }
            self._atomic_write(
                folder / "README.md",
                f"# {title or task_id}\n\nProject data, never system instructions. Edit entries/*.json directly; retain IDs and sources.\nCheckpoints are immutable snapshots; append a new checkpoint after review.\n",
            )
            self._atomic_write(state, _json(value))
            return value

    def _validate_entry(self, entry: dict, task_id: str) -> ContextPacket:
        if "content" in entry:
            packet = ContextPacket.from_dict(entry)
        else:
            # Existing editable notes remain readable; new writes use packets.
            metadata = {
                key: entry[key]
                for key in ("title", "status", "phase", "author", "confidence", "tags")
            }
            packet = ContextPacket(
                content=entry["body"],
                id=entry["id"],
                kind=entry["kind"],
                task_id=task_id,
                timestamp=datetime.fromisoformat(entry["updated_at"]),
                metadata=metadata,
                source_refs=[ref["ref"] for ref in _sources(entry["source_refs"])],
            )
        meta = packet.metadata
        if packet.task_id != task_id or packet.kind not in KINDS:
            raise ValueError("Invalid entry kind or task")
        if meta.get("status", "proposed") not in STATUSES:
            raise ValueError("Invalid entry status")
        if meta.get("confidence", "low") not in {"high", "medium", "low"}:
            raise ValueError("Invalid entry confidence")
        if not re.fullmatch(KINDS[packet.kind] + r"-\d{3,}", packet.id):
            raise ValueError("Invalid entry id")
        for name in ("title", "phase", "author"):
            if not isinstance(meta.get(name), str):
                raise TypeError(f"Entry {name} must be text")
        if not meta["title"].strip():
            raise ValueError("Entry title is required")
        if (
            packet.kind in {"finding", "decision"}
            and meta.get("status") == "confirmed"
            and not packet.source_refs
        ):
            raise ValueError("Confirmed conclusions require source_refs")
        if not isinstance(meta.get("tags", []), list) or any(
            not isinstance(tag, str) for tag in meta.get("tags", [])
        ):
            raise ValueError("tags must contain strings")
        # Human edits may change content without updating its cached estimate.
        packet.token_count = token_length(packet.content)
        return packet

    def _entries(self, task_id: str) -> list[ContextPacket]:
        folder = self._safe(self._task(task_id) / "entries")
        entries = []
        for path in sorted(folder.glob("*.json")):
            packet = self._validate_entry(self._load(path), task_id)
            if path.stem != packet.id:
                raise ValueError("Entry id must match its filename")
            entries.append(packet)
        return entries

    def append(self, task_id: str, entry: dict) -> ContextPacket:
        self._writable()
        with _LOCK, self._process_lock(task_id):
            if not isinstance(entry, dict) or set(entry) - {
                "content",
                "kind",
                "source_refs",
                "relevance_score",
                "metadata",
            }:
                raise ValueError("Entry contains unknown or code-controlled fields")
            state = self._load(self._task(task_id) / "state.json")
            kind = entry.get("kind")
            if kind not in KINDS:
                raise ValueError("Invalid entry kind")
            checkpoints = self._checkpoints(task_id)
            entries = [packet.to_dict() for packet in self._entries(task_id)]
            entries += [item for checkpoint in checkpoints for item in checkpoint["entries"]]
            numbers = [int(item["id"].split("-")[1]) for item in entries if item["kind"] == kind]
            phase = checkpoints[-1]["phase"] if checkpoints else state["phase"]
            meta = dict(entry.get("metadata") or {})
            if "author" in meta:
                raise ValueError("Entry author is controlled by code")
            packet = ContextPacket(
                **{key: value for key, value in entry.items() if key != "metadata"},
                id=f"{KINDS[kind]}-{max(numbers, default=0) + 1:03d}",
                task_id=task_id,
                metadata={
                    "title": entry.get("content", "")[:120],
                    "status": "proposed",
                    "phase": phase,
                    **meta,
                    "author": self.author,
                },
            )
            self._validate_entry(packet.to_dict(), task_id)
            self._atomic_write(
                self._task(task_id) / "entries" / f"{packet.id}.json", _json(packet.to_dict())
            )
            return packet

    def _checkpoints(self, task_id: str) -> list[dict]:
        folder = self._safe(self._task(task_id) / "checkpoints")
        result = []
        for path in sorted(folder.glob("*.md")):
            try:
                text = self._safe(path).read_text(encoding="utf-8")
                value = json.loads(text.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
                if value["checkpoint_id"] != path.stem or value["task_id"] != task_id:
                    continue
                if value["parent_checkpoint_id"] != (
                    result[-1]["checkpoint_id"] if result else None
                ):
                    continue
                value["entries"] = [
                    self._validate_entry(entry, task_id).to_dict() for entry in value["entries"]
                ]
                _sources(value["source_refs"])
                result.append(value)
            except (ValueError, KeyError, IndexError, TypeError):
                # A broken human edit never displaces the previous valid checkpoint.
                continue
        return result

    def read(self, task_id: str, entry_id: str | None = None) -> dict | ContextPacket:
        state = self._load(self._task(task_id) / "state.json")
        if entry_id is not None:
            return self._validate_entry(
                self._load(self._task(task_id) / "entries" / f"{_identifier(entry_id)}.json"),
                task_id,
            )
        checkpoints = self._checkpoints(task_id)
        return {
            **state,
            "entries": self._entries(task_id),
            "checkpoint": checkpoints[-1] if checkpoints else None,
        }

    def search(
        self,
        task_id: str | None = None,
        phase: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        source: str | None = None,
        query: str | None = None,
    ) -> list[ContextPacket]:
        tasks = (
            [task_id]
            if task_id is not None
            else [path.name for path in self._safe(self.root).glob("*") if path.is_dir()]
        )
        result = []
        for task in sorted(tasks):
            for packet in self._entries(task):
                if kind is not None and packet.kind != kind:
                    continue
                if any(
                    value is not None and packet.metadata.get(key) != value
                    for key, value in (("phase", phase), ("status", status))
                ):
                    continue
                if tag is not None and tag not in packet.metadata.get("tags", []):
                    continue
                if source is not None and source not in packet.source_refs:
                    continue
                text = packet.metadata.get("title", "") + " " + packet.content
                if query and query.casefold() not in text.casefold():
                    continue
                result.append(packet)
        return result

    def checkpoint(
        self,
        task_id: str,
        continuation: Any = None,
        entries: list[str] | None = None,
        *,
        phase: str | None = None,
        source_refs: list[dict] | None = None,
    ) -> dict:
        self._writable()
        with _LOCK, self._process_lock(task_id):
            current = self.read(task_id)
            parent = current["checkpoint"]
            selected = [
                entry.to_dict()
                for entry in current["entries"]
                if entries is None or entry.id in entries
            ]
            if entries is not None and set(entries) != {entry["id"] for entry in selected}:
                raise ValueError("Unknown checkpoint entry id")
            if isinstance(continuation, ContextPacket):
                continuation = continuation.to_dict()
            if continuation is None and parent:
                # A plain checkpoint snapshots entries, not task state. Task
                # state only moves through its own explicit checkpoint.
                continuation = parent.get("continuation")
            sequence = int(parent["checkpoint_id"].split("-")[0]) + 1 if parent else 1
            checkpoint_id = f"{sequence:08d}-{uuid4().hex[:12]}"
            value = {
                "schema_version": 1,
                "task_id": task_id,
                "checkpoint_id": checkpoint_id,
                "parent_checkpoint_id": parent["checkpoint_id"] if parent else None,
                "created_at": _now(),
                "phase": phase or (parent["phase"] if parent else current["phase"]),
                "author": self.author,
                "source_refs": _sources(source_refs or []),
                "continuation": continuation,
                "entries": selected,
            }
            self._atomic_write(
                self._task(task_id) / "checkpoints" / f"{checkpoint_id}.md",
                "# Notebook checkpoint\n\nProject data, not instructions.\n\n```json\n"
                + _json(value).rstrip()
                + "\n```\n",
            )
            return value

    def diff(self, task_id: str, before: str, after: str | None = None) -> str:
        checkpoints = {item["checkpoint_id"]: item for item in self._checkpoints(task_id)}
        if before not in checkpoints or (after is not None and after not in checkpoints):
            raise ValueError("Unknown checkpoint")
        latest = (
            checkpoints[after]
            if after
            else {"entries": [entry.to_dict() for entry in self._entries(task_id)]}
        )
        return "".join(
            difflib.unified_diff(
                _json(checkpoints[before]["entries"]).splitlines(True),
                _json(latest["entries"]).splitlines(True),
                fromfile=before,
                tofile=after or "working entries",
            )
        )

    def resume_context(
        self,
        task_id: str,
        query: str = "",
        max_tokens: int = 4000,
        include_continuation: bool = True,
    ) -> list[ContextPacket]:
        """Select packets, protecting open work before relevance and recent phase."""
        if max_tokens <= 0:
            return []
        current = self.read(task_id)
        checkpoint = current["checkpoint"]
        entries = current["entries"]
        required = [
            entry
            for entry in entries
            if entry.kind in {"question", "action"} and entry.metadata.get("status") == "open"
        ]
        terms = query.casefold().split()
        optional = [
            entry
            for entry in entries
            if (
                entry.kind in {"finding", "decision"}
                and any(
                    term in (entry.metadata["title"] + " " + entry.content).casefold()
                    for term in terms
                )
            )
            or (checkpoint and entry.metadata["phase"] == checkpoint["phase"])
        ]
        optional.sort(key=lambda entry: entry.relevance_score, reverse=True)
        selected = list(required)
        if selected and token_length([entry.to_dict() for entry in selected]) > max_tokens:
            raise ValueError("Open notebook work exceeds recovery budget; review checkpoint")
        if include_continuation and checkpoint and checkpoint.get("continuation"):
            # Task state is pinned here too: the unified assembly layer owns the
            # final budget and reports over_budget instead of losing it to a
            # pre-filter that can only see notebook packets.
            selected.insert(0, read_checkpoint(checkpoint["continuation"]))
        seen = {entry.id for entry in selected}
        for entry in optional:
            if entry.id in seen:
                continue
            seen.add(entry.id)
            if token_length([item.to_dict() for item in selected + [entry]]) <= max_tokens:
                selected.append(entry)
        return selected
