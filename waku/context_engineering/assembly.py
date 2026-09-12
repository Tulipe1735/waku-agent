"""
Select

Assembly
├── packets
│   最终送给模型的 context
│
├── dropped
│   哪些内容没进去 + 为什么
│
├── over_budget
│   强制保护内容是否已经超过预算
│
└── token_length
    最终实际 token 数
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from .packet import ContextPacket, token_length

DEFAULT_TOKEN_BUDGET = 6000
_PINNED_KINDS = frozenset({"continuation"})
_OPEN_WORK_KINDS = frozenset({"question", "action"})
_OPEN_STATUS = "open"


@dataclass
class Assembly:
    packets: list[ContextPacket] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    over_budget: bool = False
    token_length: int = 0


def _priority(packet: ContextPacket) -> int:
    if packet.kind in _PINNED_KINDS:
        return 0
    if packet.kind in _OPEN_WORK_KINDS and packet.metadata.get("status") == _OPEN_STATUS:
        return 1
    if packet.kind == "aggregation":
        return 2
    return 3


def _epoch(packet: ContextPacket) -> float:
    timestamp = packet.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.timestamp()


def _identity(packet: ContextPacket) -> str:
    return " ".join(packet.content.split()).casefold()


def _drop(packet: ContextPacket, reason: str) -> dict[str, Any]:
    return {"id": packet.id, "kind": packet.kind, "reason": reason}


def _serialized(packets: list[ContextPacket]) -> int:
    return token_length([packet.to_dict() for packet in packets])


def assemble(packets: Iterable[ContextPacket], max_tokens: int = DEFAULT_TOKEN_BUDGET) -> Assembly:
    """Order packets and fit them to a whole-packet budget.

    Task state and open work are never dropped; optional evidence is taken by
    relevance and recency while the serialized list fits. Every omission is
    reported, and pinned packets alone exceeding the budget set over_budget
    instead of being silently truncated.
    """
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    candidates = list(packets)
    if any(not isinstance(packet, ContextPacket) for packet in candidates):
        raise TypeError("assemble expects ContextPacket instances")
    ordered = sorted(
        candidates,
        key=lambda packet: (
            _priority(packet),
            -packet.relevance_score,
            -_epoch(packet),
            packet.id,
        ),
    )
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    unique: list[ContextPacket] = []
    dropped: list[dict[str, Any]] = []
    for packet in ordered:
        identity = _identity(packet)
        if packet.id in seen_ids:
            dropped.append(_drop(packet, "duplicate_id"))
            continue
        if identity and identity in seen_content:
            dropped.append(_drop(packet, "duplicate_content"))
            continue
        seen_ids.add(packet.id)
        if identity:
            seen_content.add(identity)
        unique.append(packet)
    selected = [packet for packet in unique if _priority(packet) <= 1]
    for packet in (packet for packet in unique if _priority(packet) > 1):
        if _serialized(selected + [packet]) <= max_tokens:
            selected.append(packet)
        else:
            dropped.append(_drop(packet, "budget"))
    tokens = _serialized(selected)
    return Assembly(
        packets=selected,
        dropped=dropped,
        over_budget=tokens > max_tokens,
        token_length=tokens,
    )
