"""Versioned task handoffs. Checkpoints are untrusted project data, never instructions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now() -> str:
    return datetime.now(UTC).isoformat()


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("Invalid task/checkpoint identifier")
    return value


def token_length(value) -> int:
    # UTF-8 bytes is deliberately conservative across scripts; no tokenizer dependency.
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return len(text.encode("utf-8"))


@dataclass
class Continuation:
    task_id: str
    objective: str
    schema_version: int = 1
    checkpoint_id: str = field(default_factory=lambda: uuid4().hex)
    parent_checkpoint_id: str | None = None
    generated_at: str = field(default_factory=now)
    status: str = "active"
    current_phase: str = "init"
    done: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recent_turn_digest: str = ""
    omitted_history: list[str] = field(default_factory=list)

    def validate(self) -> None:
        safe_id(self.task_id)
        safe_id(self.checkpoint_id)
        if self.parent_checkpoint_id is not None:
            safe_id(self.parent_checkpoint_id)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported continuation schema")
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("Objective is required")
        if self.status not in ("active", "blocked", "complete"):
            raise ValueError("Invalid task status")
        if not isinstance(self.current_phase, str) or not self.current_phase:
            raise ValueError("Phase is required")
        datetime.fromisoformat(self.generated_at)
        for name in (
            "done",
            "open_questions",
            "constraints",
            "artifacts",
            "next_actions",
            "risks",
            "omitted_history",
        ):
            items = getattr(self, name)
            if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
                raise ValueError(f"{name} must be a list of strings")
        if not isinstance(self.recent_turn_digest, str):
            raise TypeError("Invalid recent_turn_digest")
        for name, label, statuses in (
            ("decisions", "decision", ("confirmed", "inferred", "uncertain")),
            ("facts", "statement", ("confirmed", "hypothesis", "rejected")),
        ):
            items = getattr(self, name)
            if not isinstance(items, list):
                raise TypeError(f"{name} must be a list")
            for item in items:
                key = "confidence" if name == "decisions" else "status"
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get(label), str)
                    or not item[label].strip()
                    or item.get(key) not in statuses
                ):
                    raise ValueError(f"Invalid {name} entry")
                if name == "decisions" and not isinstance(item.get("rationale"), str):
                    raise ValueError("Decision rationale required")
                refs = item.get("source_refs")
                if (
                    not isinstance(refs, list)
                    or not refs
                    or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                ):
                    raise ValueError("Conclusions require source references")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Continuation:
        result = cls(**data)
        result.validate()
        return result


class ContinuationStore:
    """Immutable JSON checkpoints with an atomic head; failed writes never replace head."""

    def __init__(self, root: Path):
        self.root = Path(root).absolute()

    def _directory(self, task_id: str) -> Path:
        path = self.root / safe_id(task_id)
        for part in (self.root, *self.root.parents, path):
            if part.is_symlink():
                raise ValueError("Symlink checkpoint paths are forbidden")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def latest(self, task_id: str) -> Continuation | None:
        path = self._directory(task_id)
        head = path / "latest.json"
        if not head.exists():
            return None
        if head.is_symlink():
            raise ValueError("Symlink head forbidden")
        # Consult only committed IDs; uncommitted orphan files cannot win.
        for manifest in (head, path / "previous.json"):
            if manifest.is_symlink():
                raise ValueError("Symlink manifest forbidden")
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                for identifier in [data["checkpoint_id"]] + data.get("ancestors", []):
                    try:
                        return self.read(task_id, identifier)
                    except (ValueError, TypeError, OSError):
                        continue
            except (ValueError, KeyError, TypeError, OSError):
                continue
        raise ValueError("No valid committed checkpoint remains")

    def read(self, task_id: str, checkpoint_id: str) -> Continuation:
        path = self._directory(task_id) / f"{safe_id(checkpoint_id)}.json"
        if path.is_symlink():
            raise ValueError("Symlink checkpoint forbidden")
        result = Continuation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if result.task_id != task_id or result.checkpoint_id != checkpoint_id:
            raise ValueError("Checkpoint identity mismatch")
        return result

    def save(self, continuation: Continuation) -> Path:
        continuation.validate()
        directory = self._directory(continuation.task_id)
        # Serialize writers across processes as well as threads. The lock contains no data.
        lock = directory / ".write-lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        temporary = directory / f".{uuid4().hex}.tmp"
        try:
            previous = self.latest(continuation.task_id)
            expected = previous.checkpoint_id if previous else None
            if continuation.parent_checkpoint_id != expected:
                raise ValueError("Parent checkpoint does not match current head")
            target = directory / f"{continuation.checkpoint_id}.json"
            with target.open("x", encoding="utf-8") as out:
                json.dump(continuation.to_dict(), out, ensure_ascii=False, indent=2)
                out.flush()
                os.fsync(out.fileno())
            ancestors = []
            parent = previous
            while parent:
                ancestors.append(parent.checkpoint_id)
                parent = (
                    self.read(parent.task_id, parent.parent_checkpoint_id)
                    if parent.parent_checkpoint_id
                    else None
                )
            if previous:
                backup = directory / "previous.json"
                if backup.is_symlink():
                    raise ValueError("Symlink backup forbidden")
                with temporary.open("x", encoding="utf-8") as out:
                    json.dump(
                        {"checkpoint_id": previous.checkpoint_id, "ancestors": ancestors[1:]}, out
                    )
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, backup)
            with temporary.open("x", encoding="utf-8") as out:
                json.dump(
                    {"checkpoint_id": continuation.checkpoint_id, "ancestors": ancestors}, out
                )
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, directory / "latest.json")
            return target
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
