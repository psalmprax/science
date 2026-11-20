"""Clients for the language-model proposer: a callable ``prompt -> reply`` over an OpenAI-compatible API.

Status: experimental

:class:`apedp_dyn.proposers.llm_proposer.LlmEquationProposer` takes any
``Callable[[str], str]``. ``nvidia_client`` is one for NVIDIA's hosted models
(``https://integrate.api.nvidia.com/v1``, chat completions, bearer key).

The key is read from the environment (``NVIDIA_API_KEY``) at call time. It is
never written to a file, a log line or an error message by this module: errors
carry the HTTP status and the service's own message only. Keep it in a secret
store (a GitHub Actions secret, a Colab secret), not in the repository.

Which models an account may call changes, and listing a model does not mean
the account may use it. ``pick_model`` therefore goes down a list of preferred
identifiers and returns the first one that actually answers.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
KEY_VARIABLE = "NVIDIA_API_KEY"
#: tried in order; the first that answers is used. These three answered within the time limit in both surveys
#: of 2026-11-19; six others the catalogue lists timed out in at least one (kimi-k3.1-pro, gemma-4-31b-it,
#: glm-5.3-flash, muse-glimmer-30b, step-5-preview, mistral-large-4-1t-a52b).
PREFERRED_MODELS = (
    "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3-ultra-550b-a55b", "minimaxai/minimax-m3.1-flash",
)
#: listed models whose names say they are not general chat models; never tried as a fallback
_NOT_CHAT = ("embed", "rerank", "guard", "safety", "detect", "vision", "-vl", "whisper", "parakeet", "tts", "ocr", "clip",
             "retriev", "reward", "translate", "asr", "audio", "image", "video", "diffusion", "pii", "topic-control")
_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def reply_text(message: dict[str, Any]) -> tuple[str, str]:
    """The answer in a chat message and where it was found (``content``, a reasoning field, or ``nothing``).

    Hosted reasoning models put their working in a separate field and the
    answer in ``content``. When the token budget runs out while they are still
    working, ``content`` comes back empty and the only text is the working;
    that text is returned then (its end is where proposals would be), marked
    as such, because an empty reply teaches nothing. ``content`` may also be a
    list of parts, and working may be wrapped in ``<think>`` tags, closed or not.
    """
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    text = _THINKING.sub("", str(content or ""))
    if "<think>" in text:  # an unclosed tag: the budget ran out inside the working
        text = text.split("<think>", 1)[0] or text.split("<think>", 1)[1]
    if text.strip():
        return text.strip(), "content"
    for field in _REASONING_FIELDS:
        working = message.get(field)
        if isinstance(working, str) and working.strip():
            return working.strip()[-4000:], field
    return "", "nothing"


class LlmUnavailableError(RuntimeError):
    """The model could not be reached or refused the request."""


def _key(api_key: str | None) -> str:
    key = api_key or os.environ.get(KEY_VARIABLE, "")
    if not key:
        raise LlmUnavailableError(f"no API key: set {KEY_VARIABLE} in the environment (a secret, not a file in the repository)")
    return key


def _request(path: str, payload: dict[str, Any] | None, api_key: str | None, base_url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 - fixed https service
        base_url.rstrip("/") + path, data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {_key(api_key)}", "Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if payload is None else "POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https service
        parsed: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return parsed


def available_models(api_key: str | None = None, base_url: str = NVIDIA_BASE_URL, timeout: float = 30.0) -> list[str]:
    """Identifiers the service lists. Being listed does not mean the account may call the model."""
    try:
        listing = _request("/models", None, api_key, base_url, timeout)
    except urllib.error.HTTPError as exc:
        raise LlmUnavailableError(f"listing models failed with HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise LlmUnavailableError(f"the service could not be reached ({type(exc).__name__})") from None
    return sorted(str(item["id"]) for item in listing.get("data", []))


def nvidia_client(model: str, *, api_key: str | None = None, base_url: str = NVIDIA_BASE_URL, temperature: float = 0.0,
                  max_tokens: int = 4096, min_interval: float = 1.6, retries: int = 4, timeout: float = 120.0,
                  thinking: bool | None = False, reasoning_budget: int | None = None,
                  sleep: Callable[[float], None] = time.sleep) -> Callable[[str], str]:
    """A ``prompt -> reply`` callable for one model. ``client.calls`` counts requests; ``client.model`` names the model.

    Requests are spaced ``min_interval`` seconds apart (the free tier is rate
    limited) and retried with a growing wait on HTTP 429 and 5xx. Reasoning that
    a model wraps in ``<think>`` tags is removed from the reply (:func:`reply_text`).
    ``client.last`` describes the latest answer (where the text was found, why
    the model stopped, token counts) and never contains the key or the prompt.

    ``thinking`` is sent as ``chat_template_kwargs.enable_thinking``, the switch
    NVIDIA documents for its hosted reasoning models; ``None`` leaves the model's
    default. It is off by default because a proposer needs the answer, and with
    thinking on the models surveyed on 2026-11-19 spent 4,096 tokens working
    through the rows by hand and never reached one. ``reasoning_budget`` caps the
    working when thinking is on.
    """
    state = {"last": 0.0}

    def client(prompt: str) -> str:
        wait = min_interval
        for attempt in range(retries + 1):
            pause = state["last"] + min_interval - time.monotonic()
            if pause > 0:
                sleep(pause)
            state["last"] = time.monotonic()
            client.calls += 1  # type: ignore[attr-defined]
            try:
                body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}],
                                        "temperature": temperature, "max_tokens": max_tokens}
                if thinking is not None:
                    body["chat_template_kwargs"] = {"enable_thinking": thinking}
                    if thinking and reasoning_budget is not None:
                        body["reasoning_budget"] = reasoning_budget
                reply = _request("/chat/completions", body, api_key, base_url, timeout)
                choice = reply["choices"][0]
                text, source = reply_text(choice["message"])
                client.last = {"source": source, "finish_reason": choice.get("finish_reason"), "characters": len(text),  # type: ignore[attr-defined]
                               "fields": sorted(k for k, v in choice["message"].items() if v),
                               "usage": {k: v for k, v in (reply.get("usage") or {}).items() if isinstance(v, int)}}
                return text
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:200] if exc.fp else ""
                if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                    raise LlmUnavailableError(f"{model}: HTTP {exc.code} {detail}") from None
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
                if attempt == retries:
                    raise LlmUnavailableError(f"{model}: {type(exc).__name__}") from None
            sleep(wait)
            wait *= 2
        raise LlmUnavailableError(f"{model}: no reply")  # pragma: no cover - the loop returns or raises

    client.calls = 0  # type: ignore[attr-defined]
    client.last = {}  # type: ignore[attr-defined]
    client.model = model  # type: ignore[attr-defined]
    return client


def pick_model(preferences: Sequence[str] = PREFERRED_MODELS, *, api_key: str | None = None, base_url: str = NVIDIA_BASE_URL,
               sleep: Callable[[float], None] = time.sleep, fall_back_to_listed: bool = True) -> str:
    """The first preferred model that is listed and answers a one-line prompt.

    If none of the preferences is listed (the catalogue changes), every listed
    model whose name does not mark it as something other than a chat model is
    tried in the order listed, unless ``fall_back_to_listed`` is off.
    """
    listing = available_models(api_key, base_url)
    listed = set(listing)
    failures = []
    candidates = list(preferences)
    if fall_back_to_listed:
        candidates += [m for m in listing if m not in preferences and not any(word in m.lower() for word in _NOT_CHAT)]
    for model in candidates:
        if model not in listed:
            failures.append(f"{model}: not listed")
            continue
        try:
            nvidia_client(model, api_key=api_key, base_url=base_url, max_tokens=64, retries=1, sleep=sleep)("Reply with the word ok.")
            return model
        except LlmUnavailableError as exc:
            failures.append(str(exc)[:120])
    raise LlmUnavailableError("no preferred model answered: " + "; ".join(failures))
