"""Operational Footprint and Automated Lifecycle Pruner for APEDP-X.

Addresses operational audit defect (-0.8):
  "The system accumulates massive local build and toolchain bloat (Cargo target
  caches exceeding 4.7 GB, dual Lean 4 toolchains consuming 6 GB, and
  multi-gigabyte package caches across uv, bun, and npx), quickly driving the
  host filesystem to capacity without automated lifecycle pruning."

Provides:
  1. StorageFootprint: Deterministic audit of filesystem usage, Cargo build targets,
     active vs. auxiliary Lean 4 (elan) toolchains, and package manager caches.
  2. LifecyclePolicy: Preregistered and configurable storage budgets and pruning rules.
  3. prune_lifecycle(): Safe, idempotent pruning of bloat with dry-run support.
  4. auto_guard_storage(): Zero-configuration pre-flight headroom check and self-healing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("apedp_dyn.operational.lifecycle_pruner")

# Pre-registered default operational limits
DEFAULT_MAX_DISK_USAGE_PERCENT = 85.0
DEFAULT_MIN_DISK_FREE_GB = 10.0
DEFAULT_MAX_CARGO_TARGET_GB = 2.0
DEFAULT_MAX_PACKAGE_CACHE_GB = 1.0


def _dir_size_bytes(path: Path) -> int:
    """Calculate the total size in bytes of a directory tree safely."""
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_symlink():
                    continue
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += _dir_size_bytes(Path(entry.path))
            except OSError:
                continue
    except OSError:
        pass
    return total


def _format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0 or unit == "TB":
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    return f"{num_bytes:.2f} B"


@dataclass
class StorageFootprint:
    """Audited local storage footprint across filesystem, targets, and caches."""
    root_total_gb: float
    root_used_gb: float
    root_avail_gb: float
    root_usage_percent: float

    cargo_target_bytes: int
    cargo_target_dirs: list[str]

    elan_active_toolchain: str
    elan_auxiliary_toolchains: list[str]
    elan_active_bytes: int
    elan_auxiliary_bytes: int

    uv_cache_bytes: int
    bun_cache_bytes: int
    npm_npx_cache_bytes: int
    pip_cache_bytes: int

    browser_snap_cache_bytes: int
    playwright_cache_bytes: int
    ide_cache_bytes: int
    test_cache_bytes: int

    total_reclaimable_bytes: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["formatted"] = {
            "root_usage": f"{self.root_usage_percent:.1f}% ({self.root_avail_gb:.2f} GB free of {self.root_total_gb:.2f} GB)",
            "cargo_target": _format_bytes(self.cargo_target_bytes),
            "elan_auxiliary": _format_bytes(self.elan_auxiliary_bytes),
            "uv_cache": _format_bytes(self.uv_cache_bytes),
            "bun_cache": _format_bytes(self.bun_cache_bytes),
            "npm_npx_cache": _format_bytes(self.npm_npx_cache_bytes),
            "pip_cache": _format_bytes(self.pip_cache_bytes),
            "browser_snap_cache": _format_bytes(self.browser_snap_cache_bytes),
            "playwright_cache": _format_bytes(self.playwright_cache_bytes),
            "ide_cache": _format_bytes(self.ide_cache_bytes),
            "test_cache": _format_bytes(self.test_cache_bytes),
            "total_reclaimable": _format_bytes(self.total_reclaimable_bytes),
        }
        return data


@dataclass
class LifecyclePolicy:
    """Storage thresholds and pruning directives."""
    max_disk_usage_percent: float = DEFAULT_MAX_DISK_USAGE_PERCENT
    min_disk_free_gb: float = DEFAULT_MIN_DISK_FREE_GB
    max_cargo_target_gb: float = DEFAULT_MAX_CARGO_TARGET_GB
    max_package_cache_gb: float = DEFAULT_MAX_PACKAGE_CACHE_GB

    prune_cargo_targets: bool = True
    prune_auxiliary_lean: bool = True
    prune_package_caches: bool = True
    prune_browser_caches: bool = True
    prune_ide_caches: bool = True
    prune_test_caches: bool = False


@dataclass
class PruneResult:
    """Summary of operations performed by the lifecycle pruner."""
    dry_run: bool
    initial_avail_gb: float
    final_avail_gb: float
    bytes_reclaimed: int
    actions_taken: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "initial_avail_gb": round(self.initial_avail_gb, 2),
            "final_avail_gb": round(self.final_avail_gb, 2),
            "bytes_reclaimed": self.bytes_reclaimed,
            "formatted_reclaimed": _format_bytes(self.bytes_reclaimed),
            "actions_taken": self.actions_taken,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _get_active_elan_toolchain() -> str:
    """Determine the currently active/default Lean 4 elan toolchain."""
    elan_bin = shutil.which("elan")
    if elan_bin:
        try:
            res = subprocess.run(  # noqa: S603
                [elan_bin, "which", "lean"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0 and res.stdout.strip():
                lean_path = Path(res.stdout.strip())
                for part in lean_path.parts:
                    if "lean4" in part or "leanprover" in part:
                        return part
        except (OSError, subprocess.SubprocessError):
            pass

        try:
            res = subprocess.run(  # noqa: S603
                [elan_bin, "default"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip().split()[0]
        except (OSError, subprocess.SubprocessError):
            pass

    return ""


def _scan_elan_toolchains() -> tuple[str, list[str], int, int]:
    """Scan ~/.elan/toolchains and categorize active vs. auxiliary toolchains."""
    toolchains_dir = Path.home() / ".elan" / "toolchains"
    if not toolchains_dir.is_dir():
        return "", [], 0, 0

    active_name = _get_active_elan_toolchain()
    installed = [d.name for d in toolchains_dir.iterdir() if d.is_dir()]

    # If active_name is empty or not in installed, pick the newest or first
    if active_name not in installed and installed:
        active_name = sorted(installed)[-1]

    auxiliary: list[str] = [t for t in installed if t != active_name]

    active_bytes = _dir_size_bytes(toolchains_dir / active_name) if active_name else 0
    aux_bytes = sum(_dir_size_bytes(toolchains_dir / t) for t in auxiliary)

    return active_name, auxiliary, active_bytes, aux_bytes


def _find_cargo_targets(repo_root: Path) -> list[Path]:
    """Find Cargo target directories within the workspace."""
    targets: list[Path] = []
    # Root target
    root_target = repo_root / "target"
    if root_target.is_dir():
        targets.append(root_target)

    # Subproject targets (e.g. rust/apedp-kernel/target)
    rust_dir = repo_root / "rust"
    if rust_dir.is_dir():
        for sub in rust_dir.iterdir():
            t = sub / "target"
            if t.is_dir() and t not in targets:
                targets.append(t)

    return targets


def audit_storage_footprint(repo_root: Path | None = None) -> StorageFootprint:
    """Perform a deterministic non-destructive scan of operational storage footprint."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    # 1. Filesystem root usage
    try:
        usage = shutil.disk_usage(repo_root)
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        avail_gb = usage.free / (1024**3)
        usage_pct = (usage.used / usage.total) * 100.0 if usage.total > 0 else 0.0
    except OSError:
        total_gb, used_gb, avail_gb, usage_pct = 0.0, 0.0, 0.0, 0.0

    # 2. Cargo target directories
    cargo_targets = _find_cargo_targets(repo_root)
    cargo_target_bytes = sum(_dir_size_bytes(t) for t in cargo_targets)
    cargo_target_dirs = [str(t.relative_to(repo_root) if t.is_relative_to(repo_root) else t) for t in cargo_targets]

    # 3. Elan toolchains
    active_elan, aux_elan, active_elan_bytes, aux_elan_bytes = _scan_elan_toolchains()

    # 4. Package manager caches
    home = Path.home()
    uv_cache = _dir_size_bytes(home / ".cache" / "uv") + _dir_size_bytes(home / ".local" / "share" / "uv" / "cache")
    bun_cache = _dir_size_bytes(home / ".bun" / "install" / "cache")
    npm_npx_cache = _dir_size_bytes(home / ".npm" / "_npx") + _dir_size_bytes(home / ".npm" / "_logs")
    pip_cache = _dir_size_bytes(home / ".cache" / "pip")

    # 5. Browser & Playwright caches
    snap_firefox = _dir_size_bytes(home / "snap" / "firefox" / "common" / ".cache")
    snap_chromium = _dir_size_bytes(home / "snap" / "chromium" / "common" / ".cache")
    chrome_cache = _dir_size_bytes(home / ".cache" / "google-chrome")
    browser_snap_cache = snap_firefox + snap_chromium + chrome_cache
    playwright_cache = _dir_size_bytes(home / ".cache" / "ms-playwright")

    # 6. IDE caches (VS Code & Antigravity)
    ide_dirs = [
        home / ".config" / "Code" / "Cache",
        home / ".config" / "Code" / "CachedData",
        home / ".config" / "Code" / "CachedExtensionVSIXs",
        home / ".config" / "Antigravity" / "Cache",
        home / ".config" / "Antigravity" / "CachedData",
        home / ".config" / "Antigravity" / "CachedExtensionVSIXs",
        home / ".config" / "Antigravity" / "Crashpad",
    ]
    ide_cache = sum(_dir_size_bytes(d) for d in ide_dirs)

    # 7. Stale in-repo test caches
    test_dirs = [
        repo_root / ".pytest_cache",
        repo_root / ".mypy_cache",
        repo_root / ".ruff_cache",
    ]
    test_cache = sum(_dir_size_bytes(d) for d in test_dirs)

    # Total reclaimable: auxiliary Lean toolchains + package caches + browser caches + playwright + ide caches + cargo targets
    total_reclaimable = (
        cargo_target_bytes
        + aux_elan_bytes
        + uv_cache
        + bun_cache
        + npm_npx_cache
        + pip_cache
        + browser_snap_cache
        + playwright_cache
        + ide_cache
    )

    return StorageFootprint(
        root_total_gb=total_gb,
        root_used_gb=used_gb,
        root_avail_gb=avail_gb,
        root_usage_percent=usage_pct,
        cargo_target_bytes=cargo_target_bytes,
        cargo_target_dirs=cargo_target_dirs,
        elan_active_toolchain=active_elan,
        elan_auxiliary_toolchains=aux_elan,
        elan_active_bytes=active_elan_bytes,
        elan_auxiliary_bytes=aux_elan_bytes,
        uv_cache_bytes=uv_cache,
        bun_cache_bytes=bun_cache,
        npm_npx_cache_bytes=npm_npx_cache,
        pip_cache_bytes=pip_cache,
        browser_snap_cache_bytes=browser_snap_cache,
        playwright_cache_bytes=playwright_cache,
        ide_cache_bytes=ide_cache,
        test_cache_bytes=test_cache,
        total_reclaimable_bytes=total_reclaimable,
    )


def _safe_clean_dir_contents(directory: Path, dry_run: bool) -> tuple[int, list[str]]:
    """Clean the contents of a directory without removing the directory itself."""
    if not directory.is_dir():
        return 0, []
    reclaimed = _dir_size_bytes(directory)
    actions: list[str] = []
    if dry_run:
        return reclaimed, [f"[dry-run] Would empty {directory} ({_format_bytes(reclaimed)})"]

    for item in directory.iterdir():
        try:
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Could not remove {item}: {e}")
    actions.append(f"Emptied {directory} (freed {_format_bytes(reclaimed)})")
    return reclaimed, actions


def prune_lifecycle(
    policy: LifecyclePolicy | None = None,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> PruneResult:
    """Execute lifecycle pruning of bloated build caches, toolchains, and package managers."""
    if policy is None:
        policy = LifecyclePolicy()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    initial_audit = audit_storage_footprint(repo_root)
    initial_avail = initial_audit.root_avail_gb

    total_reclaimed = 0
    actions: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    home = Path.home()

    # 1. Prune Cargo targets if requested
    if policy.prune_cargo_targets:
        targets = _find_cargo_targets(repo_root)
        for t in targets:
            size = _dir_size_bytes(t)
            # Prune if target size exceeds threshold or force clean
            if size > 0 and (size >= policy.max_cargo_target_gb * (1024**3) or not dry_run):
                if dry_run:
                    total_reclaimed += size
                    actions.append(f"[dry-run] Would remove Cargo target {t} ({_format_bytes(size)})")
                else:
                    # Attempt cargo clean first if cargo is available in target dir
                    cargo_bin = shutil.which("cargo")
                    if cargo_bin:
                        try:
                            workspace_root = t.parent
                            subprocess.run(  # noqa: S603
                                [cargo_bin, "clean"],
                                cwd=workspace_root,
                                capture_output=True,
                                timeout=60,
                            )
                        except (OSError, subprocess.SubprocessError):
                            pass

                    # Ensure target directory is fully deleted
                    if t.exists():
                        try:
                            shutil.rmtree(t)
                            total_reclaimed += size
                            actions.append(f"Removed Cargo target {t} (freed {_format_bytes(size)})")
                        except OSError as e:
                            errors.append(f"Failed to delete {t}: {e}")
                    else:
                        total_reclaimed += size
                        actions.append(f"Cleaned Cargo target {t} via cargo clean (freed {_format_bytes(size)})")

    # 2. Prune Auxiliary Lean 4 toolchains
    if policy.prune_auxiliary_lean and initial_audit.elan_auxiliary_toolchains:
        for toolchain in initial_audit.elan_auxiliary_toolchains:
            t_dir = home / ".elan" / "toolchains" / toolchain
            size = _dir_size_bytes(t_dir)
            if dry_run:
                total_reclaimed += size
                actions.append(f"[dry-run] Would uninstall auxiliary Lean toolchain {toolchain} ({_format_bytes(size)})")
            else:
                uninstalled = False
                elan_bin = shutil.which("elan")
                if elan_bin:
                    try:
                        res = subprocess.run(  # noqa: S603
                            [elan_bin, "toolchain", "uninstall", toolchain],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if res.returncode == 0:
                            uninstalled = True
                    except (OSError, subprocess.SubprocessError):
                        pass

                if not uninstalled and t_dir.exists():
                    try:
                        shutil.rmtree(t_dir)
                        uninstalled = True
                    except OSError as e:
                        errors.append(f"Failed to remove toolchain {toolchain}: {e}")

                if uninstalled:
                    total_reclaimed += size
                    actions.append(f"Uninstalled auxiliary Lean toolchain {toolchain} (freed {_format_bytes(size)})")

    # 3. Prune package manager caches
    if policy.prune_package_caches:
        # UV cache
        uv_dirs = [home / ".cache" / "uv", home / ".local" / "share" / "uv" / "cache"]
        for d in uv_dirs:
            if d.exists():
                size = _dir_size_bytes(d)
                if size > 0:
                    if dry_run:
                        total_reclaimed += size
                        actions.append(f"[dry-run] Would clear UV cache at {d} ({_format_bytes(size)})")
                    else:
                        uv_bin = shutil.which("uv")
                        if uv_bin:
                            try:
                                subprocess.run([uv_bin, "cache", "clean"], capture_output=True, timeout=30)  # noqa: S603
                            except (OSError, subprocess.SubprocessError):
                                pass
                        rec, acts = _safe_clean_dir_contents(d, dry_run=False)
                        total_reclaimed += rec
                        actions.extend(acts)

        # Bun cache
        bun_cache_dir = home / ".bun" / "install" / "cache"
        if bun_cache_dir.exists():
            size = _dir_size_bytes(bun_cache_dir)
            if size > 0:
                if dry_run:
                    total_reclaimed += size
                    actions.append(f"[dry-run] Would clear Bun cache at {bun_cache_dir} ({_format_bytes(size)})")
                else:
                    bun_bin = shutil.which("bun")
                    if bun_bin:
                        try:
                            subprocess.run([bun_bin, "pm", "cache", "rm"], capture_output=True, timeout=30)  # noqa: S603
                        except (OSError, subprocess.SubprocessError):
                            pass
                    rec, acts = _safe_clean_dir_contents(bun_cache_dir, dry_run=False)
                    total_reclaimed += rec
                    actions.extend(acts)

        # NPM / NPX cache
        npm_npx = home / ".npm" / "_npx"
        if npm_npx.exists():
            size = _dir_size_bytes(npm_npx)
            if size > 0:
                if dry_run:
                    total_reclaimed += size
                    actions.append(f"[dry-run] Would clear NPX cache at {npm_npx} ({_format_bytes(size)})")
                else:
                    try:
                        shutil.rmtree(npm_npx)
                        total_reclaimed += size
                        actions.append(f"Removed NPX cache {npm_npx} (freed {_format_bytes(size)})")
                    except OSError as e:
                        errors.append(f"Failed to remove {npm_npx}: {e}")

        npm_logs = home / ".npm" / "_logs"
        if npm_logs.exists():
            size = _dir_size_bytes(npm_logs)
            if size > 0:
                if dry_run:
                    total_reclaimed += size
                    actions.append(f"[dry-run] Would clear NPM logs at {npm_logs} ({_format_bytes(size)})")
                else:
                    try:
                        shutil.rmtree(npm_logs)
                        total_reclaimed += size
                        actions.append(f"Removed NPM logs {npm_logs} (freed {_format_bytes(size)})")
                    except OSError as e:
                        errors.append(f"Failed to remove {npm_logs}: {e}")

        # Pip cache
        pip_cache_dir = home / ".cache" / "pip"
        if pip_cache_dir.exists():
            size = _dir_size_bytes(pip_cache_dir)
            if size > 0:
                if dry_run:
                    total_reclaimed += size
                    actions.append(f"[dry-run] Would clear Pip cache at {pip_cache_dir} ({_format_bytes(size)})")
                else:
                    pip_bin = shutil.which("pip") or shutil.which("pip3")
                    if pip_bin:
                        try:
                            subprocess.run([pip_bin, "cache", "purge"], capture_output=True, timeout=30)  # noqa: S603
                        except (OSError, subprocess.SubprocessError):
                            pass
                    rec, acts = _safe_clean_dir_contents(pip_cache_dir, dry_run=False)
                    total_reclaimed += rec
                    actions.extend(acts)

    # 4. Prune browser & playwright caches
    if policy.prune_browser_caches:
        browser_dirs = [
            home / "snap" / "firefox" / "common" / ".cache",
            home / "snap" / "chromium" / "common" / ".cache",
            home / ".cache" / "google-chrome",
            home / ".cache" / "ms-playwright",
        ]
        for b_dir in browser_dirs:
            if b_dir.exists():
                size = _dir_size_bytes(b_dir)
                if size > 0:
                    rec, acts = _safe_clean_dir_contents(b_dir, dry_run)
                    total_reclaimed += rec
                    actions.extend(acts)

    # 5. Prune IDE caches (VS Code & Antigravity)
    if policy.prune_ide_caches:
        ide_dirs = [
            home / ".config" / "Code" / "Cache",
            home / ".config" / "Code" / "CachedData",
            home / ".config" / "Code" / "CachedExtensionVSIXs",
            home / ".config" / "Antigravity" / "Cache",
            home / ".config" / "Antigravity" / "CachedData",
            home / ".config" / "Antigravity" / "CachedExtensionVSIXs",
            home / ".config" / "Antigravity" / "Crashpad",
        ]
        for ide_dir in ide_dirs:
            if ide_dir.exists():
                size = _dir_size_bytes(ide_dir)
                if size > 0:
                    rec, acts = _safe_clean_dir_contents(ide_dir, dry_run)
                    total_reclaimed += rec
                    actions.extend(acts)

    # 6. Optional test caches
    if policy.prune_test_caches:
        test_dirs = [
            repo_root / ".pytest_cache",
            repo_root / ".mypy_cache",
            repo_root / ".ruff_cache",
        ]
        for t_dir in test_dirs:
            if t_dir.exists():
                size = _dir_size_bytes(t_dir)
                if size > 0:
                    rec, acts = _safe_clean_dir_contents(t_dir, dry_run)
                    total_reclaimed += rec
                    actions.extend(acts)

    if dry_run:
        final_avail = initial_avail
    else:
        final_audit = audit_storage_footprint(repo_root)
        final_avail = final_audit.root_avail_gb

    return PruneResult(
        dry_run=dry_run,
        initial_avail_gb=initial_avail,
        final_avail_gb=final_avail,
        bytes_reclaimed=total_reclaimed,
        actions_taken=actions,
        warnings=warnings,
        errors=errors,
    )


def auto_guard_storage(
    policy: LifecyclePolicy | None = None,
    repo_root: Path | None = None,
) -> bool:
    """Pre-flight automated headroom guard.

    Evaluates host filesystem capacity. If disk usage exceeds `max_disk_usage_percent`
    or free space falls below `min_disk_free_gb`, automatically invokes lifecycle
    pruning to restore operational headroom.

    Returns:
        True if the filesystem is within safe operational limits, False if critical
        thresholds are breached and cannot be remedied.
    """
    if policy is None:
        policy = LifecyclePolicy()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    audit = audit_storage_footprint(repo_root)

    is_breached = (
        audit.root_usage_percent >= policy.max_disk_usage_percent
        or audit.root_avail_gb <= policy.min_disk_free_gb
    )

    if not is_breached:
        logger.info(
            f"Storage guard OK: {audit.root_usage_percent:.1f}% used ({audit.root_avail_gb:.2f} GB free)"
        )
        return True

    logger.warning(
        f"Storage guard TRIGGERED: {audit.root_usage_percent:.1f}% used ({audit.root_avail_gb:.2f} GB free). "
        f"Thresholds: max {policy.max_disk_usage_percent}%, min {policy.min_disk_free_gb} GB. Initiating pruning..."
    )

    res = prune_lifecycle(policy=policy, dry_run=False, repo_root=repo_root)
    logger.info(
        f"Automated pruning complete: reclaimed {_format_bytes(res.bytes_reclaimed)}. "
        f"New free space: {res.final_avail_gb:.2f} GB."
    )

    post_audit = audit_storage_footprint(repo_root)
    return post_audit.root_usage_percent < policy.max_disk_usage_percent


def main() -> int:
    """CLI entrypoint for operational storage auditing and lifecycle pruning."""
    parser = argparse.ArgumentParser(
        description="APEDP-X Operational Storage Footprint & Lifecycle Pruner"
    )
    parser.add_argument(
        "--status", action="store_true", help="Print storage footprint audit report"
    )
    parser.add_argument(
        "--prune", action="store_true", help="Execute lifecycle pruning of bloated caches"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate pruning without deleting files"
    )
    parser.add_argument(
        "--guard",
        action="store_true",
        help="Check operational headroom and auto-prune only if thresholds breached",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results formatted as JSON"
    )
    parser.add_argument(
        "--max-usage",
        type=float,
        default=DEFAULT_MAX_DISK_USAGE_PERCENT,
        help="Max allowable filesystem usage percentage (default: 85.0)",
    )
    parser.add_argument(
        "--min-free",
        type=float,
        default=DEFAULT_MIN_DISK_FREE_GB,
        help="Min allowable free disk in GB (default: 10.0)",
    )

    args = parser.parse_args()

    policy = LifecyclePolicy(
        max_disk_usage_percent=args.max_usage,
        min_disk_free_gb=args.min_free,
    )

    if args.guard:
        healthy = auto_guard_storage(policy=policy)
        if args.json:
            print(json.dumps({"healthy": healthy, "policy": asdict(policy)}, indent=2))
        else:
            status = "HEALTHY" if healthy else "CRITICAL"
            print(f"Operational Storage Guard: {status}")
        return 0 if healthy else 1

    if args.prune or args.dry_run:
        res = prune_lifecycle(policy=policy, dry_run=args.dry_run)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            prefix = "[DRY-RUN] " if res.dry_run else ""
            print(f"{prefix}Lifecycle Pruning Results:")
            print(f"  Initial Free: {res.initial_avail_gb:.2f} GB")
            print(f"  Final Free:   {res.final_avail_gb:.2f} GB")
            print(f"  Reclaimed:    {_format_bytes(res.bytes_reclaimed)}")
            print("\nActions Taken:")
            for a in res.actions_taken:
                print(f"  - {a}")
            if res.errors:
                print("\nErrors:")
                for e in res.errors:
                    print(f"  ! {e}")
        return 0

    # Default action: status report
    footprint = audit_storage_footprint()
    if args.json:
        print(json.dumps(footprint.to_dict(), indent=2))
    else:
        fmt = footprint.to_dict()["formatted"]
        print("=" * 68)
        print("  APEDP-X Operational Storage Footprint & Lifecycle Audit")
        print("=" * 68)
        print(f"Filesystem Usage:          {fmt['root_usage']}")
        print(f"Cargo Build Targets:       {fmt['cargo_target']}")
        print(f"Active Lean Toolchain:     {footprint.elan_active_toolchain or 'None'} ({_format_bytes(footprint.elan_active_bytes)})")
        print(f"Auxiliary Lean Toolchains: {', '.join(footprint.elan_auxiliary_toolchains) or 'None'} ({fmt['elan_auxiliary']})")
        print(f"UV Package Cache:          {fmt['uv_cache']}")
        print(f"Bun Package Cache:         {fmt['bun_cache']}")
        print(f"NPM / NPX Cache:           {fmt['npm_npx_cache']}")
        print(f"Pip Package Cache:         {fmt['pip_cache']}")
        print(f"Browser Snap Caches:       {fmt['browser_snap_cache']}")
        print(f"Playwright Cache:          {fmt['playwright_cache']}")
        print(f"IDE Engine / VSIX Cache:   {fmt['ide_cache']}")
        print(f"In-Repo Test Cache:        {fmt['test_cache']}")
        print("-" * 68)
        print(f"Total Reclaimable Bloat:   {fmt['total_reclaimable']}")
        print("=" * 68)
        if footprint.root_usage_percent >= policy.max_disk_usage_percent:
            print("  ⚠️  WARNING: Filesystem usage exceeds operational budget threshold!")
            print("  Run 'python -m apedp_dyn.operational.lifecycle_pruner --prune' to clean.")
        else:
            print("  ✅ Filesystem headroom is currently within operational budget.")
        print("=" * 68)

    return 0


if __name__ == "__main__":
    sys.exit(main())
