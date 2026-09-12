"""
Structure

One small information packet shared by context producers and consumers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def token_length(value: Any) -> int:
    """Conservative UTF-8 byte estimate, not a provider tokenizer count."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return len(text.encode("utf-8"))


@dataclass
class ContextPacket:
    """Candidate information; relevance is a ranking hint, never authority.

    A zero token_count is estimated from content. Consumers budget the complete
    serialized packet as well, so metadata cannot bypass a context limit.
    Producer-specific details belong in metadata, without packet subclasses.
    """

    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    token_count: int = 0
    relevance_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    kind: str = "context"
    task_id: str = ""
    source_refs: list[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化后处理"""
        if not isinstance(self.content, str) or not isinstance(self.timestamp, datetime):
            raise TypeError("content must be text and timestamp must be a datetime")
        if type(self.token_count) is not int or self.token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        if not math.isfinite(self.relevance_score):
            raise ValueError("relevance_score must be finite")
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))
        if self.metadata is None:
            self.metadata = {}
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if any(not isinstance(value, str) for value in (self.id, self.kind, self.task_id)):
            raise TypeError("id, kind and task_id must be text")
        if not isinstance(self.source_refs, list) or any(
            not isinstance(ref, str) or not ref.strip() for ref in self.source_refs
        ):
            raise ValueError("source_refs must contain non-empty strings")
        if self.token_count == 0:
            self.token_count = token_length(self.content)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ContextPacket:
        data = dict(data)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
