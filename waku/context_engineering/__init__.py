"""Local context engineering with a shared information packet."""

from .assembly import Assembly, assemble
from .packet import ContextPacket

__all__ = ["Assembly", "ContextPacket", "assemble"]
