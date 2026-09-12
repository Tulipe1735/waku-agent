"""THE LOOP — observe → reason → act → repeat. This file is the whole trick.

Every agent framework is ultimately this while-loop with more indirection:

    while not done:
        response = llm(messages, tools)          # reason
        if response asks for tools:
            results = run(tool_calls)            # act
            messages += results                  # observe
        else:
            done                                 # reply to the human

End-loop guardrails (the orange box's exit conditions):
  1. the model stops asking for tools  → natural end of turn
  2. max_iterations reached            → hard stop, never spin forever
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from waku.tools.registry import ToolRegistry

# Observers let the gateway show tool calls live and let ops/tracing record
# them — without either being wired into the loop's logic.
LoopEvent = dict[str, Any]
Observer = Callable[[str, LoopEvent], None]


class ModelClient(Protocol):
    """The shared Messages seam supports native, compatible and test clients."""

    @property
    def messages(self) -> Any: ...


@dataclass
class LoopResult:
    reply: str
    tool_calls: list[LoopEvent] = field(default_factory=list)
    iterations: int = 0


def run_loop(
    client: ModelClient,
    model: str,
    system: str,
    messages: list[dict],
    tools: ToolRegistry,
    max_iterations: int = 10,
    max_tokens: int = 2048,
    observer: Observer | None = None,
    stream: bool = False,
    tool_result_limit: int | None = None,
) -> LoopResult:
    """Run one agent turn. `messages` is mutated in place — after the call it
    contains the full working memory of the turn (assistant thoughts, tool
    calls, tool results), which is exactly what gets traced.

    stream=True emits the assistant's text as it's generated (notify("text",
    {"delta": ...})) so a gateway can show it appear token by token — used by
    the dashboard. Falls back to a single call for clients without streaming."""
    notify = observer or (lambda kind, ev: None)
    result = LoopResult(reply="")
    can_stream = stream and hasattr(client.messages, "stream")
    observed_results = []

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration

        # ---- reason: one LLM call with the current working memory
        response = None
        if can_stream:
            try:
                with client.messages.stream(
                    model=model,
                    system=system,
                    messages=messages,
                    tools=tools.schemas(),
                    max_tokens=max_tokens,
                ) as s:
                    for delta in s.text_stream:
                        notify("text", {"delta": delta})
                    response = s.get_final_message()
            except Exception:
                response = None  # any streaming hiccup → fall back to one call
        if response is None:
            response = client.messages.create(
                model=model,
                system=system,
                messages=messages,
                tools=tools.schemas(),
                max_tokens=max_tokens,
            )
        notify(
            "llm",
            {
                "iteration": iteration,
                "stop_reason": response.stop_reason,
                "usage": {"in": response.usage.input_tokens, "out": response.usage.output_tokens},
            },
        )

        # the assistant's turn (text and/or tool requests) joins working memory
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        # ---- guardrail 1: no tool calls → the model is talking to the human
        if not tool_uses:
            result.reply = "".join(b.text for b in response.content if b.type == "text")
            return result

        # ---- act: execute each requested tool; observe: feed results back
        tool_results = []
        for call in tool_uses:
            output = tools.execute(call.name, call.input, notify=notify)
            event = {"tool": call.name, "args": call.input, "output": output}
            result.tool_calls.append(event)
            notify("tool", event)
            block = {"type": "tool_result", "tool_use_id": call.id, "content": output}
            tool_results.append(block)
            if tool_result_limit is not None:
                from waku.context_engineering.compaction import bounded_tool_result
                observed_results.append((block, output))
                # Rebalance observations without splitting any call/result pair.
                # Full raw outputs remain in LoopResult and the normal tool trace.
                allowance = max(1, min(tool_result_limit, 6000 // len(observed_results)))
                for observed, raw in observed_results:
                    observed["content"] = bounded_tool_result(raw, allowance)
        messages.append({"role": "user", "content": tool_results})

    # ---- guardrail 2: ran out of iterations
    result.reply = (
        "(I hit my iteration limit before finishing — try breaking the request into smaller steps.)"
    )
    return result
