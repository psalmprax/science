"""The CI security scan's accepted findings, and the check that closes the one there is (audit M-21)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from apedp_dyn.proposers import llm_clients, llm_https

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "bandit_baseline.json"


def test_the_baseline_holds_one_finding_and_it_is_the_documented_one():
    """A baseline that grows quietly is a gate switched off; each entry needs an audit entry."""
    results = json.loads(BASELINE.read_text(encoding="utf-8"))["results"]
    assert [(r["filename"], r["test_id"]) for r in results] == [("apedp_dyn/proposers/llm_clients.py", "B310")]
    audit = (ROOT / "docs" / "SYSTEM_FLAWS_AUDIT.md").read_text(encoding="utf-8")
    assert "M-21" in audit and "bandit_baseline.json" in audit


def test_the_scan_uses_the_baseline_and_still_fails_on_anything_new():
    workflow = yaml.load((ROOT / ".github" / "workflows" / "test.yml").read_text(), Loader=yaml.BaseLoader)  # noqa: S506
    script = "\n".join(step.get("run", "") for step in workflow["jobs"]["test"]["steps"])
    scan = next(line for line in script.splitlines() if line.strip().startswith("bandit "))
    following = script[script.index(scan):]
    assert "-ll" in scan and "-b scripts/bandit_baseline.json" in following
    assert "|| true" not in following.split("\n\n")[0] and "--exit-zero" not in following


def test_the_frozen_client_is_unchanged_and_still_has_the_open_call():
    digest = json.loads((ROOT / "preregistration" / "DIGEST-LLM-CONTROLLER.json").read_text())["sha256"]
    path = "apedp_dyn/proposers/llm_clients.py"
    import hashlib

    assert digest[path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("address", ["http://integrate.api.nvidia.com/v1", "file:///etc/passwd", "ftp://example.org", "",
                                     "integrate.api.nvidia.com/v1", "https://"])
def test_an_address_that_is_not_https_is_refused_before_any_request(address, monkeypatch):
    called = []
    monkeypatch.setattr(llm_clients, "_request", lambda *a, **k: called.append(a) or {})
    with pytest.raises(ValueError, match="https"):
        llm_https.nvidia_client("some-model", base_url=address, api_key="k")
    with pytest.raises(ValueError, match="https"):
        llm_https.available_models("k", address)
    assert not called


def test_the_default_address_is_https_and_passes():
    assert llm_https.require_https(llm_clients.NVIDIA_BASE_URL) == llm_clients.NVIDIA_BASE_URL
