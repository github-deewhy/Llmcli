"""Ollama native API provider (``/api/chat``, newline-delimited JSON stream)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from llmcli.config import Config
from llmcli.providers.base import BaseProvider, StreamState, ToolCall, parse_tool_calls
from llmcli.providers.events import Capabilities, Done, Event, Usage
from llmcli.providers.http import check_status, protocol_error, translate_errors

MAX_SERVER_ERROR_CHARS = 200


class OllamaProvider(BaseProvider):
    def __init__(self, config: Config):
        super().__init__(getattr(config, "request_timeout", 300))
        self.endpoint = config.endpoint
        self.model = config.model
        self.context_size = config.context_size
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p
        # Top-level request fields such as {"think": false} (spec 14, section 2.2).
        self.extra_body = dict(getattr(config, "extra_body", {}) or {})

    # --- health and models ---------------------------------------------------

    def check_health(self) -> tuple[bool, str]:
        try:
            with translate_errors(self.endpoint, self.request_timeout):
                resp = self.client.get(f"{self.endpoint}/api/tags")
                check_status(resp, self.endpoint)
                models = [m["name"] for m in resp.json().get("models", [])]
        except Exception as error:  # noqa: BLE001 - health check reports, never raises
            return False, str(error) or error.__class__.__name__
        if not self._model_installed(models):
            return False, f"Model '{self.model}' not found locally. Run: `ollama pull {self.model}`"
        return True, f"Connected to Ollama. Model '{self.model}' is ready."

    def _model_installed(self, installed: list[str]) -> bool:
        """Ollama lists untagged pulls as ``name:latest``; accept either spelling."""
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return self.model in installed or wanted in installed

    def list_models(self) -> list[str]:
        try:
            resp = self.client.get(f"{self.endpoint}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:  # noqa: BLE001
            return []

    # --- probing ---------------------------------------------------------------

    def probe(self) -> Capabilities:
        """``POST /api/show``: capability list and context length (no generation)."""
        try:
            resp = self.client.post(f"{self.endpoint}/api/show", json={"model": self.model})
            if resp.status_code >= 400:
                return self.capabilities
            body = resp.json()
        except Exception:  # noqa: BLE001 - probing must never break start-up
            return self.capabilities
        if not isinstance(body, dict):
            return self.capabilities
        capabilities = body.get("capabilities")
        if isinstance(capabilities, list):
            self.capabilities.set("native_tools", "yes" if "tools" in capabilities else "no", "server")
            self.capabilities.set("reasoning", "field" if "thinking" in capabilities else "none", "server")
        model_info = body.get("model_info")
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                if key.endswith(".context_length") and isinstance(value, int):
                    self.capabilities.set("context_length", value, "server")
                    break
        return self.capabilities

    # --- streaming -------------------------------------------------------------

    def _payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        default_temperature = 0.2 if tools else 0.7
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": self.context_size,
                "num_predict": self.max_tokens,
                "temperature": self.temperature if self.temperature is not None else default_temperature,
                **({"top_p": self.top_p} if self.top_p is not None else {}),
            },
        }
        if tools:
            payload["tools"] = tools
        for key, value in self.extra_body.items():
            payload.setdefault(key, value)
        return payload

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[Event]:
        state = StreamState(self.capabilities)
        raw_calls: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        with translate_errors(self.endpoint, self.request_timeout):
            with self.client.stream(
                "POST", f"{self.endpoint}/api/chat", json=self._payload(messages, tools)
            ) as resp:
                check_status(resp, self.endpoint)
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    chunk = self._decode(line)
                    message = chunk.get("message") or {}
                    if not isinstance(message, dict):
                        raise protocol_error(self.endpoint, "message is not an object")
                    thinking = message.get("thinking")
                    if isinstance(thinking, str):
                        yield from state.reasoning(thinking)
                    content = message.get("content")
                    if isinstance(content, str):
                        yield from state.content(content)
                    calls = message.get("tool_calls")
                    if calls:
                        if not isinstance(calls, list):
                            raise protocol_error(self.endpoint, "tool_calls is not a list")
                        for call in self._parse_calls(calls):
                            raw_calls.append(call[1])
                            yield state.tool_call(call[0])
                    if chunk.get("done"):
                        final = chunk
                        break
        yield from state.finish()
        if raw_calls:
            self.capabilities.set("streams_tool_calls", "yes", "observed")
        yield self.record_usage(self._usage(final, state))
        assistant: dict[str, Any] = {"role": "assistant", "content": state.text}
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        yield Done(finish_reason=final.get("done_reason"), assistant_message=assistant)

    def _decode(self, line: str) -> dict[str, Any]:
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            raise protocol_error(self.endpoint, "invalid JSON line") from None
        if not isinstance(chunk, dict):
            raise protocol_error(self.endpoint, "line is not a JSON object")
        if "error" in chunk:
            reason = str(chunk["error"])[:MAX_SERVER_ERROR_CHARS]
            raise protocol_error(self.endpoint, f"server error: {reason}")
        return chunk

    def _parse_calls(self, calls: list[Any]) -> list[tuple[ToolCall, dict[str, Any]]]:
        try:
            parsed = parse_tool_calls(calls)
        except ValueError as error:
            raise protocol_error(self.endpoint, str(error)) from None
        return list(zip(parsed, calls, strict=True))

    @staticmethod
    def _usage(final: dict[str, Any], state: StreamState) -> Usage:
        prompt = final.get("prompt_eval_count")
        completion = final.get("eval_count")
        duration_ns = final.get("eval_duration")
        speed = None
        if isinstance(completion, int) and isinstance(duration_ns, int) and duration_ns > 0:
            speed = round(completion / (duration_ns / 1e9), 2)
        return Usage(
            prompt_tokens=prompt if isinstance(prompt, int) else None,
            completion_tokens=completion if isinstance(completion, int) else None,
            cached_tokens=None,
            ttft_s=state.ttft_s,
            gen_tokens_per_s=speed,
        )

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_name": call.name, "content": content}
