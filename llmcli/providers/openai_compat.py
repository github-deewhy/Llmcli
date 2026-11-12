"""OpenAI-compatible API provider (LM Studio, vLLM, OpenRouter, Groq, etc.)."""

import json
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx

from llmcli.config import Config
from llmcli.providers.base import BaseProvider, ModelResponse, parse_tool_calls


class OpenAICompatProvider(BaseProvider):
    def __init__(self, config: Config):
        self.endpoint = config.endpoint.rstrip("/")
        # Use the base URL exactly as the service documents it (.../v1,
        # .../v1beta/openai, .../api/v1). Only a bare host such as LM Studio's
        # http://localhost:1234 gets the conventional /v1 appended.
        self.base_url = self.endpoint if urlparse(self.endpoint).path.strip("/") else f"{self.endpoint}/v1"
        self.model = config.model
        self.api_key = config.api_key
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.extra_body = dict(config.extra_body)
        self.timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)

    def check_health(self) -> tuple[bool, str]:
        # Generic health check: just verify we can reach the endpoint root or /v1/models
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = httpx.get(f"{self.base_url}/models", headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return True, f"Connected to OpenAI-compatible endpoint. Using model: {self.model}"
            return False, f"Endpoint returned {resp.status_code}. Check API key and URL."
        except httpx.ConnectError:
            return False, f"Cannot connect to endpoint at {self.endpoint}."
        except Exception as e:
            return False, str(e)

    def _apply_sampling(self, payload: dict[str, Any], default_temperature: float | None) -> None:
        """Add sampling options and provider-specific extras to a request body."""
        temperature = self.temperature if self.temperature is not None else default_temperature
        if temperature is not None:
            payload["temperature"] = temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        for key, value in self.extra_body.items():
            payload.setdefault(key, value)

    def chat(self, messages: list[dict], stream: bool = True) -> Iterator[str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
        }
        self._apply_sampling(payload, default_temperature=None)

        url = f"{self.base_url}/chat/completions"
        with httpx.stream("POST", url, json=payload, headers=headers, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices and "delta" in choices[0] and "content" in choices[0]["delta"]:
                            content = choices[0]["delta"]["content"]
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.max_tokens,
        }
        self._apply_sampling(payload, default_temperature=0.2)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("OpenAI-compatible endpoint returned an invalid response.")
        choices = body.get("choices", [])
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("message"), dict)
        ):
            raise ValueError("OpenAI-compatible endpoint returned no assistant message.")
        message = choices[0]["message"]
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("OpenAI-compatible endpoint returned an invalid tool call list.")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("OpenAI-compatible endpoint returned non-text assistant content.")
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            assistant_message["tool_calls"] = calls
        return ModelResponse(
            content=content,
            tool_calls=parse_tool_calls(calls),
            assistant_message=assistant_message,
        )

    def list_models(self) -> list[str]:
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = httpx.get(f"{self.base_url}/models", headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []
