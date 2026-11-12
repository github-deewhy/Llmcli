"""Ollama native API provider."""

import json
from collections.abc import Iterator
from typing import Any

import httpx

from llmcli.config import Config
from llmcli.providers.base import BaseProvider, ModelResponse, ToolCall, parse_tool_calls


class OllamaProvider(BaseProvider):
    def __init__(self, config: Config):
        self.endpoint = config.endpoint
        self.model = config.model
        self.context_size = config.context_size
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p
        # Long read timeout for cold model loads on CPU
        self.timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)

    def check_health(self) -> tuple[bool, str]:
        try:
            resp = httpx.get(f"{self.endpoint}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not self._model_installed(models):
                return False, f"Model '{self.model}' not found locally. Run: `ollama pull {self.model}`"
            return True, f"Connected to Ollama. Model '{self.model}' is ready."
        except httpx.ConnectError:
            return False, f"Cannot connect to Ollama at {self.endpoint}. Is it running?"
        except Exception as e:
            return False, str(e)

    def _model_installed(self, installed: list[str]) -> bool:
        """Ollama lists untagged pulls as ``name:latest``; accept either spelling."""
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return self.model in installed or wanted in installed

    def chat(self, messages: list[dict], stream: bool = True) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "num_ctx": self.context_size,
                "num_predict": self.max_tokens,
                "temperature": self.temperature if self.temperature is not None else 0.7,
                **({"top_p": self.top_p} if self.top_p is not None else {}),
            },
        }

        with httpx.stream("POST", f"{self.endpoint}/api/chat", json=payload, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                except json.JSONDecodeError:
                    continue

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.context_size,
                "num_predict": self.max_tokens,
                "temperature": self.temperature if self.temperature is not None else 0.2,
                **({"top_p": self.top_p} if self.top_p is not None else {}),
            },
        }
        if tools:
            payload["tools"] = tools

        response = httpx.post(
            f"{self.endpoint}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Ollama returned an invalid response.")
        message = body.get("message", {})
        if not isinstance(message, dict):
            raise ValueError("Ollama returned an invalid assistant message.")
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("Ollama returned an invalid tool call list.")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Ollama returned non-text assistant content.")
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            assistant_message["tool_calls"] = calls
        return ModelResponse(
            content=content,
            tool_calls=parse_tool_calls(calls),
            assistant_message=assistant_message,
        )

    def tool_result_message(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_name": call.name,
            "content": content,
        }

    def list_models(self) -> list[str]:
        try:
            resp = httpx.get(f"{self.endpoint}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
