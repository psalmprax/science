"""The NVIDIA client: the request it sends, what it does with the reply, and that the key never leaks."""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from apedp_dyn.proposers import llm_clients

KEY = "test-key-not-a-real-one"


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _serve(monkeypatch, handler):
    seen = []

    def fake(request, timeout=None):
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(llm_clients.urllib.request, "urlopen", fake)
    return seen


def _chat(text):
    return _Reply(json.dumps({"choices": [{"message": {"content": text}}]}).encode())


def test_request_carries_the_key_as_a_bearer_and_the_reply_loses_its_thinking(monkeypatch):
    monkeypatch.setenv(llm_clients.KEY_VARIABLE, KEY)
    seen = _serve(monkeypatch, lambda request: _chat("<think>long reasoning</think>\nx0*x1\nx0/x1"))
    client = llm_clients.nvidia_client("some/model", min_interval=0.0, sleep=lambda s: None)
    assert client("propose") == "x0*x1\nx0/x1" and client.calls == 1 and client.model == "some/model"
    request = seen[0]
    assert request.full_url == "https://integrate.api.nvidia.com/v1/chat/completions" and request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {KEY}"
    body = json.loads(request.data)
    assert body["model"] == "some/model" and body["messages"] == [{"role": "user", "content": "propose"}] and body["temperature"] == 0.0


def test_rate_limit_is_retried_and_other_errors_are_reported_without_the_key(monkeypatch):
    monkeypatch.setenv(llm_clients.KEY_VARIABLE, KEY)
    answers = iter([429, 429, 200])

    def flaky(request):
        code = next(answers)
        if code != 200:
            raise urllib.error.HTTPError(request.full_url, code, "Too Many Requests", {}, io.BytesIO(b"slow down"))
        return _chat("x0")

    _serve(monkeypatch, flaky)
    waits = []
    client = llm_clients.nvidia_client("some/model", min_interval=0.0, sleep=waits.append)
    assert client("p") == "x0" and client.calls == 3 and waits == [0.0, 0.0]

    def forbidden(request):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"detail":"Authorization failed"}'))

    _serve(monkeypatch, forbidden)
    with pytest.raises(llm_clients.LlmUnavailableError) as caught:
        llm_clients.nvidia_client("some/model", min_interval=0.0, sleep=lambda s: None)("p")
    assert "403" in str(caught.value) and KEY not in str(caught.value) and caught.value.__cause__ is None


def test_no_key_is_an_explicit_error(monkeypatch):
    monkeypatch.delenv(llm_clients.KEY_VARIABLE, raising=False)
    with pytest.raises(llm_clients.LlmUnavailableError, match="NVIDIA_API_KEY"):
        llm_clients.nvidia_client("some/model", sleep=lambda s: None)("p")


def test_pick_model_takes_the_first_preference_that_is_listed_and_answers(monkeypatch):
    monkeypatch.setenv(llm_clients.KEY_VARIABLE, KEY)

    def handler(request):
        if request.full_url.endswith("/models"):
            return _Reply(json.dumps({"data": [{"id": "b/listed-but-refused"}, {"id": "c/works"}]}).encode())
        if json.loads(request.data)["model"] == "b/listed-but-refused":
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b"Function not found for account"))
        return _chat("ok")

    _serve(monkeypatch, handler)
    assert llm_clients.pick_model(("a/not-listed", "b/listed-but-refused", "c/works"), sleep=lambda s: None) == "c/works"
    with pytest.raises(llm_clients.LlmUnavailableError, match="no preferred model answered"):
        llm_clients.pick_model(("a/not-listed", "b/listed-but-refused"), sleep=lambda s: None)
