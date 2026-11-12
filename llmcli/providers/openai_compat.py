"""OpenAI-compatible provider (llama.cpp llama-server, vLLM, LM Studio, or a remote URL).

Only Ollama is tested against a live server (spec 14, section 3); this provider is
covered by unit tests with recorded streams.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from llmcli.config import Config
from llmcli.providers.base import BaseProvider, StreamState, parse_tool_calls
from llmcli.providers.events import Capabilities, Done, Event, Usage
from llmcli.providers.http import check_status, protocol_error, translate_errors

MAX_SERVER_ERROR_CHARS = 200


class OpenAICompatProvider(BaseProvider):
    def __init__(self, config: Config):
        super().__init__(getattr(config, "request_timeout", 300))
        self.endpoint = config.endpoint.rstrip("/")
        # Use the base URL exactly as the server documents it (.../v1 or any
        # other path). Only a bare host such as LM Studio's
        # http://localhost:1234 gets the conventional /v1 appended.
        self.base_url = self.endpoint if urlparse(self.endpoint).path.strip("/") else f"{self.endpoint}/v1"
        self.model = config.model
        self.api_key = config.api_key
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.extra_body = dict(config.extra_body)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # --- health and models -----------------------------------------------------

    def check_health(self) -> tuple[bool, str]:
        try:
            with translate_errors(self.endpoint, self.request_timeout):
                resp = self.client.get(f"{self.base_url}/models", headers=self._headers())
        except Exception as error:  # noqa: BLE001 - health check reports, never raises
            return False, str(error) or error.__class__.__name__
        if resp.status_code == 200:
            return True, f"Connected to OpenAI-compatible endpoint. Using model: {self.model}"
        return False, f"Endpoint returned {resp.status_code}. Check API key and URL."

    def list_models(self) -> list[str]:
        try:
            resp = self.client.get(f"{self.base_url}/models", headers=self._headers())
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:  # noqa: BLE001
            return []

    # --- probing ---------------------------------------------------------------

    def probe(self) -> Capabilities:
        """``GET /models`` and, for llama.cpp, ``GET /props``. Field names are [VERIFY]."""
        try:
            resp = self.client.get(f"{self.base_url}/models", headers=self._headers())
            entries = resp.json().get("data", []) if resp.status_code == 200 else []
        except Exception:  # noqa: BLE001 - probing must never break start-up
            entries = []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict) or entry.get("id") != self.model:
                continue
            length = entry.get("max_model_len")  # vLLM
            meta = entry.get("meta")
            if not isinstance(length, int) and isinstance(meta, dict):
                length = meta.get("n_ctx_train")  # llama.cpp
            if isinstance(length, int):
                self.capabilities.set("context_length", length, "server")
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        try:
            resp = self.client.get(f"{root}/props", headers=self._headers())
            props = resp.json() if resp.status_code == 200 else {}
        except Exception:  # noqa: BLE001
            props = {}
        if isinstance(props, dict):
            settings = props.get("default_generation_settings")
            if isinstance(settings, dict) and isinstance(settings.get("n_ctx"), int):
                self.capabilities.set("context_length", settings["n_ctx"], "server")
            if isinstance(props.get("total_slots"), int):
                self.capabilities.set("parallel_slots", props["total_slots"], "server")
        return self.capabilities

    # --- streaming -------------------------------------------------------------

    def _payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
        }
        temperature = self.temperature if self.temperature is not None else (0.2 if tools else None)
        if temperature is not None:
            payload["temperature"] = temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        for key, value in self.extra_body.items():
            payload.setdefault(key, value)
        return payload

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[Event]:
        state = StreamState(self.capabilities)
        fragments: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage_body: dict[str, Any] = {}
        with translate_errors(self.endpoint, self.request_timeout):
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=self._payload(messages, tools),
                headers=self._headers(),
            ) as resp:
                check_status(resp, self.endpoint)
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    chunk = self._decode(data)
                    if isinstance(chunk.get("usage"), dict):
                        usage_body = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        raise protocol_error(self.endpoint, "delta is not an object")
                    for key in ("reasoning_content", "reasoning"):
                        if isinstance(delta.get(key), str):
                            yield from state.reasoning(delta[key])
                    if isinstance(delta.get("content"), str):
                        yield from state.content(delta["content"])
                    if delta.get("tool_calls"):
                        self._merge_fragments(fragments, delta["tool_calls"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        yield from state.finish()
        raw_calls = self._assemble(fragments)
        if raw_calls:
            for call in self._parse(raw_calls):
                yield state.tool_call(call)
            if len(fragments) and any(f["chunks"] > 1 for f in fragments.values()):
                self.capabilities.set("streams_tool_calls", "yes", "observed")
        yield self.record_usage(self._usage(usage_body, state))
        assistant: dict[str, Any] = {"role": "assistant", "content": state.text}
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        yield Done(finish_reason=finish_reason, assistant_message=assistant)

    def _decode(self, data: str) -> dict[str, Any]:
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            raise protocol_error(self.endpoint, "invalid JSON in stream") from None
        if not isinstance(chunk, dict):
            raise protocol_error(self.endpoint, "stream event is not a JSON object")
        if "error" in chunk:
            error = chunk["error"]
            reason = error.get("message", "") if isinstance(error, dict) else str(error)
            raise protocol_error(self.endpoint, f"server error: {str(reason)[:MAX_SERVER_ERROR_CHARS]}")
        return chunk

    def _merge_fragments(self, fragments: dict[int, dict[str, Any]], deltas: Any) -> None:
        """Merge streamed tool-call pieces by ``index``; arguments arrive as string fragments."""
        if not isinstance(deltas, list):
            raise protocol_error(self.endpoint, "tool_calls is not a list")
        for position, delta in enumerate(deltas):
            if not isinstance(delta, dict):
                raise protocol_error(self.endpoint, "tool call fragment is not an object")
            index = delta.get("index", position)
            if not isinstance(index, int):
                raise protocol_error(self.endpoint, "tool call index is not a number")
            entry = fragments.setdefault(index, {"id": None, "name": "", "arguments": "", "chunks": 0})
            entry["chunks"] += 1
            if delta.get("id"):
                entry["id"] = delta["id"]
            function = delta.get("function") or {}
            if not isinstance(function, dict):
                raise protocol_error(self.endpoint, "tool call function is not an object")
            if isinstance(function.get("name"), str):
                entry["name"] += function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                entry["arguments"] += arguments
            elif isinstance(arguments, dict):  # some servers send whole objects
                entry["arguments"] = json.dumps(arguments)

    @staticmethod
    def _assemble(fragments: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        calls = []
        for index in sorted(fragments):
            entry = fragments[index]
            calls.append(
                {
                    "id": entry["id"] or f"call_{index}",
                    "type": "function",
                    "function": {"name": entry["name"], "arguments": entry["arguments"] or "{}"},
                }
            )
        return calls

    def _parse(self, raw_calls: list[dict[str, Any]]):
        try:
            return parse_tool_calls(raw_calls)
        except ValueError as error:
            raise protocol_error(self.endpoint, str(error)) from None

    @staticmethod
    def _usage(body: dict[str, Any], state: StreamState) -> Usage:
        prompt = body.get("prompt_tokens")
        completion = body.get("completion_tokens")
        details = body.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        completion_int = completion if isinstance(completion, int) else None
        return Usage(
            prompt_tokens=prompt if isinstance(prompt, int) else None,
            completion_tokens=completion_int,
            cached_tokens=cached if isinstance(cached, int) else None,
            ttft_s=state.ttft_s,
            gen_tokens_per_s=state.wall_tokens_per_s(completion_int),
        )
