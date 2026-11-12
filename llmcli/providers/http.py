"""HTTP client set-up and error translation shared by providers (spec 14, sections 2.3, 2.4)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx

from llmcli.providers.events import (
    ProviderConnectError,
    ProviderError,
    ProviderHTTPError,
    ProviderProtocolError,
    ProviderTimeout,
)

CONNECT_TIMEOUT_S = 5.0
WRITE_TIMEOUT_S = 30.0
POOL_TIMEOUT_S = 5.0
DEFAULT_REQUEST_TIMEOUT_S = 300


def make_client(request_timeout: int | float = DEFAULT_REQUEST_TIMEOUT_S) -> httpx.Client:
    """A client with finite timeouts that ignores proxy variables and ``.netrc`` (P8).

    ``request_timeout`` is the longest gap allowed with no bytes from the server,
    including model load and prompt processing before the first token.
    """
    return httpx.Client(
        trust_env=False,
        timeout=httpx.Timeout(
            connect=CONNECT_TIMEOUT_S,
            read=float(request_timeout),
            write=WRITE_TIMEOUT_S,
            pool=POOL_TIMEOUT_S,
        ),
    )


def status_message(endpoint: str, status_code: int) -> str:
    """One-line message for an HTTP error. The server's body is never included."""
    if status_code in (401, 403):
        hint = "Check the api_key for this profile."
    elif status_code == 404:
        hint = "Check the model name and the endpoint URL (is the model installed?)."
    elif status_code >= 500:
        hint = "The server failed; check its log."
    else:
        hint = "Check the model name, endpoint and request settings."
    return f"{endpoint} returned HTTP {status_code}. {hint}"


def check_status(response: httpx.Response, endpoint: str) -> None:
    if response.status_code >= 400:
        raise ProviderHTTPError(status_message(endpoint, response.status_code), response.status_code)


@contextmanager
def translate_errors(endpoint: str, request_timeout: int | float) -> Iterator[None]:
    """Turn httpx exceptions into one-line ``ProviderError``s."""
    try:
        yield
    except ProviderError:
        raise
    except httpx.TimeoutException as error:
        if isinstance(error, httpx.ConnectTimeout):
            raise ProviderConnectError(
                f"Cannot reach {endpoint} (connection timed out). Is the server running?"
            ) from None
        raise ProviderTimeout(
            f"No data from {endpoint} for {int(request_timeout)} s. "
            "Raise request_timeout in config.toml if the model is slow to load."
        ) from None
    except httpx.ConnectError:
        raise ProviderConnectError(f"Cannot reach {endpoint}. Is the server running?") from None
    except httpx.HTTPError as error:
        raise ProviderProtocolError(
            f"Connection to {endpoint} failed ({error.__class__.__name__})."
        ) from None


def protocol_error(endpoint: str, reason: str) -> ProviderProtocolError:
    return ProviderProtocolError(f"{endpoint} sent a response llmcli could not read ({reason}).")
