"""The CI security scan's accepted findings, and the check that closes the one there is (audit M-21)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from apedp_dyn.proposers import llm_clients, llm_https

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "bandit_baseline.json"


def test_the_baseline_holds_three_findings_and_each_is_a_documented_one():
    """A baseline that grows quietly is a gate switched off; each entry needs an audit entry.

    The second and third came in on 27 November 2026, when the scan was widened to the referee core and the
    language-model package (M-39): two downloads from fixed https addresses whose bytes are checked against a pinned
    SHA-256 before use, in files frozen by a preregistration and therefore not open to an inline marker.
    """
    results = json.loads(BASELINE.read_text(encoding="utf-8"))["results"]
    assert [(r["filename"], r["test_id"]) for r in results] == [
        ("apedp_dyn/proposers/llm_clients.py", "B310"),
        ("apedp_mmlm/law_corpora.py", "B310"),
        ("apedp_mmlm/text_corpus.py", "B310"),
    ]
    audit = (ROOT / "docs" / "SYSTEM_FLAWS_AUDIT.md").read_text(encoding="utf-8")
    assert "M-21" in audit and "bandit_baseline.json" in audit
    assert "M-39" in audit and "apedp_mmlm/law_corpora.py" in audit and "apedp_mmlm/text_corpus.py" in audit


def test_the_two_downloads_in_the_baseline_are_https_and_checked_against_a_pinned_hash():
    from apedp_mmlm import law_corpora, text_corpus

    assert law_corpora.ODEBENCH_URL.startswith("https://") and text_corpus.SOURCE.startswith("https://")
    assert len(law_corpora.ODEBENCH_SHA256) == 64 and all(len(digest) == 64 for digest in text_corpus.SHA256.values())
    for module in (law_corpora, text_corpus):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert source.count("urllib.request.urlopen(") == 1 and "does not match its pinned SHA-256" in source


def test_the_scan_covers_the_four_packages():
    workflow = yaml.load((ROOT / ".github" / "workflows" / "test.yml").read_text(), Loader=yaml.BaseLoader)  # noqa: S506
    script = "\n".join(step.get("run", "") for step in workflow["jobs"]["test"]["steps"])
    scan = next(line for line in script.splitlines() if line.strip().startswith("bandit "))
    for package in ("apedp/", "apedp_dyn/", "apedp_referee/", "apedp_mmlm/"):
        assert f" {package}" in scan


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
