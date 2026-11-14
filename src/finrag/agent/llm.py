"""A minimal chat client for any OpenAI-compatible endpoint (Groq by default).

Talking to the OpenAI-compatible API, rather than a vendor SDK, keeps the agent
provider-agnostic: switching to another host or model is a config change. The agent
only depends on the ``LLM`` protocol, so tests drive it with a scripted fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, exactly as the model produced it


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLM(Protocol):
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = GROQ_BASE_URL,
        temperature: float = 0.0,
        max_retries: int = 8,
    ) -> None:
        from openai import OpenAI

        # The SDK retries 429/5xx with exponential backoff and honours Retry-After,
        # which matters on Groq's free-tier tokens-per-minute limits.
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)
        self.model = model
        self.temperature = temperature  # 0 for reproducible evals

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs |= {"tools": tools, "tool_choice": tool_choice}
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            ToolCall(tc.id, tc.function.name, tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage
        return LLMResponse(
            content=msg.content or "",
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
