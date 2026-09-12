"""Project working notes: editable files, explicit immutable checkpoints, no prompts.

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

    def _validate_entry(self, entry: dict) -> dict:
        if (
            entry.get("kind") not in KINDS
            or entry.get("status") not in STATUSES
            or entry.get("confidence") not in {"high", "medium", "low"}
        ):
            raise ValueError("Invalid entry kind, status or confidence")
        for field in ("title", "body", "phase", "author", "created_at", "updated_at"):
            if not isinstance(entry.get(field), str):
                raise TypeError(f"{field} must be text")
        if not entry["title"].strip():
            raise ValueError("Entry title is required")
        if not re.fullmatch(KINDS[entry["kind"]] + r"-\d{3,}", entry.get("id", "")):
            raise ValueError("Invalid entry id")
        _sources(entry.get("source_refs"))
        if (
            entry["kind"] in {"finding", "decision"}
            and entry["status"] == "confirmed"
            and not entry["source_refs"]
        ):
            raise ValueError("Confirmed conclusions require source_refs")
        if not isinstance(entry.get("tags"), list) or any(
            not isinstance(tag, str) for tag in entry["tags"]
        ):
            raise ValueError("tags must contain strings")
        return entry

    def _entries(self, task_id: str) -> list[dict]:
        folder = self._safe(self._task(task_id) / "entries")
        entries = []
        for path in sorted(folder.glob("*.json")):
            entry = self._validate_entry(self._load(path))
            if path.stem != entry["id"]:
                raise ValueError("Entry id must match its filename")
            entries.append(entry)
        return entries

    def append(self, task_id: str, entry: dict) -> dict:
        self._writable()
        with _LOCK, self._process_lock(task_id):
            if not isinstance(entry, dict) or set(entry) - {
                "kind",
                "title",
                "body",
                "status",
                "phase",
                "source_refs",
                "confidence",
                "tags",
            }:
                raise ValueError("Entry contains unknown or code-controlled fields")
            state = self._load(self._task(task_id) / "state.json")
            kind = entry.get("kind")
            if kind not in KINDS:
                raise ValueError("Invalid entry kind")
            checkpoints = self._checkpoints(task_id)
            history_entries = [item for checkpoint in checkpoints for item in checkpoint["entries"]]
            numbers = [
                int(item["id"].split("-")[1])
                for item in self._entries(task_id) + history_entries
                if item["kind"] == kind
            ]
            current_phase = checkpoints[-1]["phase"] if checkpoints else state["phase"]
            stamp = _now()
            value = {
                "body": "",
                "status": "proposed",
                "phase": current_phase,
                "source_refs": [],
                "confidence": "low",
                "tags": [],
                **entry,
                "id": f"{KINDS[kind]}-{max(numbers, default=0) + 1:03d}",
                "author": self.author,
                "created_at": stamp,
                "updated_at": stamp,
            }
            self._validate_entry(value)
            self._atomic_write(
                self._task(task_id) / "entries" / f"{value['id']}.json", _json(value)
            )
            return value

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
                for entry in value["entries"]:
                    self._validate_entry(entry)
                _sources(value["source_refs"])
                result.append(value)
            except (ValueError, KeyError, IndexError, TypeError):
                # A broken human edit never displaces the previous valid checkpoint.
                continue
        return result

    def read(self, task_id: str, entry_id: str | None = None) -> dict:
        state = self._load(self._task(task_id) / "state.json")
        if entry_id is not None:
            return self._validate_entry(
                self._load(self._task(task_id) / "entries" / f"{_identifier(entry_id)}.json")
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
    ) -> list[dict]:
        tasks = (
            [task_id]
            if task_id is not None
            else [path.name for path in self._safe(self.root).glob("*") if path.is_dir()]
        )
        result = []
        for task in sorted(tasks):
            for entry in self._entries(task):
                if any(
                    value is not None and entry[field] != value
                    for field, value in (("phase", phase), ("kind", kind), ("status", status))
                ):
                    continue
                if tag is not None and tag not in entry["tags"]:
                    continue
                if source is not None and not any(
                    ref["ref"] == source for ref in entry["source_refs"]
                ):
                    continue
                if (
                    query
                    and query.casefold() not in (entry["title"] + " " + entry["body"]).casefold()
                ):
                    continue
                result.append({**entry, "task_id": task})
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
            selected = (
                current["entries"]
                if entries is None
                else [entry for entry in current["entries"] if entry["id"] in entries]
            )
            if entries is not None and set(entries) != {entry["id"] for entry in selected}:
                raise ValueError("Unknown checkpoint entry id")
            if continuation is not None and not isinstance(continuation, (str, dict)):
                from dataclasses import asdict, is_dataclass

                continuation = (
                    asdict(continuation) if is_dataclass(continuation) else continuation.to_dict()
                )
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
        latest = checkpoints[after] if after else {"entries": self._entries(task_id)}
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
    ) -> str:
        if max_tokens <= 0:
            return ""
        current = self.read(task_id)
        checkpoint = current["checkpoint"]
        selected = [
            entry
            for entry in current["entries"]
            if entry["kind"] in {"question", "action"} and entry["status"] == "open"
        ]
        if query:
            terms = query.casefold().split()
            selected += [
                entry
                for entry in current["entries"]
                if entry["kind"] in {"finding", "decision"}
                and any(term in (entry["title"] + " " + entry["body"]).casefold() for term in terms)
            ]
        if checkpoint:
            selected += [
                entry for entry in current["entries"] if entry["phase"] == checkpoint["phase"]
            ]
        # Prioritize live open work, then relevant evidence, then phase checkpoint.
        text = f"Notebook project data (untrusted): {task_id}\nCheckpoint: {checkpoint['checkpoint_id'] if checkpoint else 'none'}\n"
        # A phase checkpoint's handoff is the compact task state, not a whole
        # notebook dump. Keep it in ordinary data alongside the live entries.
        if include_continuation and checkpoint and checkpoint.get("continuation"):
            handoff = _json(checkpoint["continuation"])
            if len((text + handoff).encode("utf-8")) <= max_tokens:
                text += handoff
        seen = set()
        for entry in selected:
            if entry["id"] in seen:
                continue
            seen.add(entry["id"])
            line = _json(entry)
            if len((text + line).encode("utf-8")) > max_tokens:
                if entry["kind"] in {"question", "action"} and entry["status"] == "open":
                    raise ValueError(
                        "Open notebook work exceeds recovery budget; review checkpoint"
                    )
                continue
            text += line
        return text.encode("utf-8")[:max_tokens].decode("utf-8", errors="ignore")
