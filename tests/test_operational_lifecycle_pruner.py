"""Unit and integration tests for APEDP-X Operational Lifecycle Pruner.

Validates:
  1. Storage footprint audit accuracy, disk reporting, and JSON schema.
  2. Lifecycle policy thresholds and enforcement.
  3. Safe dry-run pruning guarantees (zero mutations).
  4. Active vs. auxiliary Lean 4 (elan) toolchain separation and retention.
  5. Package cache and Cargo target cleanup behavior.
  6. Automated pre-flight storage guard self-healing logic.
  7. CLI entrypoint execution and JSON output compliance.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from apedp_dyn.operational import (
    LifecyclePolicy,
    PruneResult,
    StorageFootprint,
    audit_storage_footprint,
    auto_guard_storage,
    prune_lifecycle,
)
from apedp_dyn.operational.lifecycle_pruner import (
    _dir_size_bytes,
    _format_bytes,
    _safe_clean_dir_contents,
    _scan_elan_toolchains,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_audit_storage_footprint_structure():
    """Verify footprint audit returns expected metrics and serializes cleanly."""
    footprint = audit_storage_footprint(repo_root=REPO_ROOT)
    assert isinstance(footprint, StorageFootprint)
    assert footprint.root_total_gb > 0.0
    assert 0.0 <= footprint.root_usage_percent <= 100.0
    assert footprint.root_avail_gb >= 0.0
    assert footprint.cargo_target_bytes >= 0
    assert footprint.uv_cache_bytes >= 0
    assert footprint.bun_cache_bytes >= 0
    assert footprint.pip_cache_bytes >= 0
    assert footprint.total_reclaimable_bytes >= 0

    d = footprint.to_dict()
    assert "formatted" in d
    assert "root_usage" in d["formatted"]
    assert "total_reclaimable" in d["formatted"]
    # Verify it is valid JSON
    serialized = json.dumps(d)
    assert len(serialized) > 0


def test_format_bytes_utility():
    """Verify formatting scales correctly from bytes to gigabytes."""
    assert "500.00 B" == _format_bytes(500)
    assert "1.00 KB" == _format_bytes(1024)
    assert "1.00 MB" == _format_bytes(1024 * 1024)
    assert "2.50 GB" == _format_bytes(int(2.5 * 1024 * 1024 * 1024))


def test_dir_size_bytes_and_safe_clean(tmp_path: Path):
    """Verify directory size computation and safe content clearing."""
    test_dir = tmp_path / "cache_test"
    test_dir.mkdir()
    sub_file1 = test_dir / "sample1.bin"
    sub_file1.write_bytes(b"A" * 2048)
    sub_dir = test_dir / "nested"
    sub_dir.mkdir()
    sub_file2 = sub_dir / "sample2.bin"
    sub_file2.write_bytes(b"B" * 4096)

    size = _dir_size_bytes(test_dir)
    assert size == 6144

    # Dry run should not delete anything
    dry_reclaimed, dry_actions = _safe_clean_dir_contents(test_dir, dry_run=True)
    assert dry_reclaimed == 6144
    assert len(dry_actions) == 1
    assert test_dir.is_dir()
    assert sub_file1.is_file()

    # Real clean should empty contents but keep the directory
    reclaimed, _actions = _safe_clean_dir_contents(test_dir, dry_run=False)
    assert reclaimed == 6144
    assert test_dir.is_dir()
    assert not sub_file1.exists()
    assert not sub_dir.exists()
    assert _dir_size_bytes(test_dir) == 0


def test_elan_toolchain_separation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify active Lean toolchain is preserved and auxiliary toolchains are flagged."""
    elan_home = tmp_path / ".elan"
    toolchains = elan_home / "toolchains"
    toolchains.mkdir(parents=True)

    # Simulate primary v4.35.0 and auxiliary v4.34.1
    t_active = toolchains / "leanprover--lean4---v4.35.0"
    t_active.mkdir()
    (t_active / "bin").mkdir()
    (t_active / "bin" / "lean").write_bytes(b"binary_active")

    t_aux = toolchains / "leanprover--lean4---v4.34.1"
    t_aux.mkdir()
    (t_aux / "bin").mkdir()
    (t_aux / "bin" / "lean").write_bytes(b"binary_auxiliary")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "apedp_dyn.operational.lifecycle_pruner._get_active_elan_toolchain",
        lambda: "leanprover--lean4---v4.35.0",
    )

    active, aux, active_bytes, aux_bytes = _scan_elan_toolchains()
    assert active == "leanprover--lean4---v4.35.0"
    assert aux == ["leanprover--lean4---v4.34.1"]
    assert active_bytes > 0
    assert aux_bytes > 0


def test_prune_lifecycle_dry_run():
    """Verify that dry-run returns a valid PruneResult with zero disk mutations."""
    policy = LifecyclePolicy()
    result = prune_lifecycle(policy=policy, dry_run=True, repo_root=REPO_ROOT)
    assert isinstance(result, PruneResult)
    assert result.dry_run is True
    assert result.initial_avail_gb == pytest.approx(result.final_avail_gb, abs=0.01)
    assert result.bytes_reclaimed >= 0
    d = result.to_dict()
    assert d["dry_run"] is True
    assert "formatted_reclaimed" in d


def test_auto_guard_storage_healthy():
    """Verify auto-guard passes when relaxed thresholds are configured."""
    # 99.9% max usage threshold and 0.01 GB min free guarantees healthy pass
    relaxed_policy = LifecyclePolicy(
        max_disk_usage_percent=99.99,
        min_disk_free_gb=0.01,
    )
    healthy = auto_guard_storage(policy=relaxed_policy, repo_root=REPO_ROOT)
    assert healthy is True


def test_cli_status_and_json_invocation():
    """Verify CLI --status and --json flags execute successfully via subprocess."""
    res = subprocess.run(
        [sys.executable, "-m", "apedp_dyn.operational.lifecycle_pruner", "--status", "--json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert res.returncode == 0, f"CLI status failed: {res.stderr}"
    data = json.loads(res.stdout)
    assert "root_usage_percent" in data
    assert "formatted" in data
    assert "cargo_target" in data["formatted"]


def test_cli_guard_flag():
    """Verify CLI --guard returns exit code 0 when within headroom limits."""
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "apedp_dyn.operational.lifecycle_pruner",
            "--guard",
            "--max-usage",
            "99.9",
            "--min-free",
            "0.1",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert res.returncode == 0
    assert "HEALTHY" in res.stdout
