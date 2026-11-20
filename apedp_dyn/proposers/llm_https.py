"""The language-model client, refusing to send the key anywhere but an https service.

Status: experimental

``proposers/llm_clients.py`` takes the service address as a parameter and hands
it to ``urllib.request.urlopen`` with the API key in a header. The default is
NVIDIA's https endpoint and no caller in the repository passes another, so
nothing has gone wrong; but a caller could pass ``http://`` (the key would
travel in clear text) or ``file://`` (a local file would be read), and the CI
security scan flags the call for exactly that (bandit B310).

That file is frozen by the digests of PREREG-LLM-PROPOSER and
PREREG-LLM-CONTROLLER and is not edited. The finding is recorded as accepted in
``scripts/bandit_baseline.json`` (audit M-21), and the check that closes it is
here: new code calls these wrappers, which refuse any address that is not https
before the frozen client sees it.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from apedp_dyn.proposers import llm_clients


def require_https(base_url: str) -> str:
    """``base_url`` if it is an https address with a host; ``ValueError`` otherwise."""
    parts = urlsplit(str(base_url))
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("the language-model service address must be https with a host; "
                         f"refusing to send an API key to a {parts.scheme or 'schemeless'} address")
    return str(base_url)


def nvidia_client(model: str, *, base_url: str = llm_clients.NVIDIA_BASE_URL, **options: Any) -> Any:
    """``llm_clients.nvidia_client`` for an https service only."""
    return llm_clients.nvidia_client(model, base_url=require_https(base_url), **options)


def available_models(api_key: str | None = None, base_url: str = llm_clients.NVIDIA_BASE_URL, timeout: float = 30.0) -> list[str]:
    """``llm_clients.available_models`` for an https service only."""
    listed: list[str] = llm_clients.available_models(api_key, require_https(base_url), timeout)
    return listed
