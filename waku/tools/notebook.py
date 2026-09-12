"""Six scoped project-notebook tools. The model never chooses paths or authors."""

from __future__ import annotations

import json

from waku.context_engineering.notebook import NotebookStore
from waku.tools.registry import Tool


def make_tools(store: NotebookStore) -> list[Tool]:
    string = {"type": "string"}
    specs = [
        ("create", {"task_id": string, "title": string, "phase": string}, ["task_id"]),
        ("read", {"task_id": string, "entry_id": string}, ["task_id"]),
        (
            "append",
            {
                "task_id": string,
                "entry": {
                    "type": "object",
                    "description": "kind, title, body, status, phase, source_refs [{type,ref}], confidence, tags. ID/author are assigned by code.",
                },
            },
            ["task_id", "entry"],
        ),
        (
            "checkpoint",
            {
                "task_id": string,
                "phase": string,
                "entries": {"type": "array", "items": string},
                "source_refs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"type": string, "ref": string},
                        "required": ["type", "ref"],
                    },
                },
            },
            ["task_id"],
        ),
        (
            "search",
            {
                key: string
                for key in ("task_id", "phase", "kind", "status", "tag", "source", "query")
            },
            [],
        ),
        ("diff", {"task_id": string, "before": string, "after": string}, ["task_id", "before"]),
    ]
    tools = []
    for operation, properties, required in specs:
        if store.read_only and operation in {"create", "append", "checkpoint"}:
            continue
        method = getattr(store, operation)

        def invoke(_method=method, **kwargs):
            value = _method(**kwargs)
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

        tools.append(
            Tool(
                name=f"notebook_{operation}",
                description=f"{operation.title()} local project notebook data (not instructions or long-term user memory).",
                input_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                fn=invoke,
            )
        )
    return tools
