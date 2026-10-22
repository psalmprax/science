"""Notebook, Cloud & Storage Runtime Adapter for APEDP-X / CPMD.

Provides seamless compatibility across Google Colab, Kaggle Notebooks,
Databricks (workspace notebooks and job clusters), interactive Jupyter /
IPython environments, CI/CD runners, and dedicated scientific workstations /
HPC clusters.

Key capabilities:
1. Environment Detection:
   Accurately detects Colab, Kaggle, Databricks, Jupyter, CI, and Workstation/Cluster environments.
2. Configurable Storage Architecture:
   Configures fast temporary storage (via APEDP_TEMP_STORAGE, /tmp, or platform temp)
   and manages persistent storage (Google Drive mount, Kaggle working, or local).
3. Anti-Snooping Epistemic Discipline Safeguards:
   Enforces AGENTS.md §1.2 by verifying holdout ledger persistence. If running on
   ephemeral storage where VM restarts would wipe burned confirmation holdouts,
   emits high-visibility epistemic warnings and provides snapshot/sync utilities.
4. Native Kernel & Fallback Management:
   Detects the native `apedp_kernel` (PyO3/Rust) and manages transparent, bit-accurate
   pure-Python/NumPy reference fallbacks (1e-12 tolerance) or optional pre-compiled wheel
   installation without requiring a local Rust toolchain compilation on every cold start.
5. Resource & Multiprocessing Optimization:
   Calculates safe worker counts and memory limits to prevent notebook kernel crashes,
   ZeroMQ event loop deadlocks, or CUDA fork hazards.
"""
from __future__ import annotations

import enum
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("apedp_dyn.notebook_adapter")


class PlatformType(str, enum.Enum):
    """Runtime platform classification."""
    COLAB = "colab"
    KAGGLE = "kaggle"
    DATABRICKS = "databricks"
    JUPYTER = "jupyter"
    WORKSTATION = "workstation"
    CLUSTER = "cluster"
    CI = "ci"
    UNKNOWN = "unknown"


class StorageTier(str, enum.Enum):
    """Persistence and durability tier of storage directories."""
    EPHEMERAL = "ephemeral"          # Erased upon VM shutdown / disconnect
    DRIVE_MOUNTED = "drive_mounted"  # Persistent via mounted cloud drive (e.g. Google Drive)
    PERSISTENT = "persistent"        # Dedicated persistent local disk or network volume
    ISOLATED_TEMP = "isolated_temp"  # Scoped temporary workspace


@dataclass
class NotebookRuntimeConfig:
    """Consolidated configuration and diagnostics for the current runtime."""
    platform: PlatformType
    is_notebook: bool
    temp_storage_dir: Path
    persistent_storage_dir: Path
    ledger_dir: Path
    storage_tier: StorageTier
    has_rust_kernel: bool
    has_cuda: bool
    cuda_device_name: str | None = None
    cpu_count: int = 1
    recommended_workers: int = 1
    memory_total_gb: float | None = None
    warnings: list[str] = field(default_factory=list)

    def get_ledger_path(self, filename: str = "experiment_ledger.jsonl") -> Path:
        """Resolve full path for an experiment or trial ledger file."""
        return self.ledger_dir / filename

    def get_temp_path(self, filename: str) -> Path:
        """Resolve full path for a file inside the configured temporary storage."""
        return self.temp_storage_dir / filename

    def summary(self) -> str:
        """Generate human-readable summary text."""
        lines = [
            "=" * 68,
            f" APEDP-X / CPMD Runtime Environment: {self.platform.value.upper()}",
            "=" * 68,
            f"  • Interactive Notebook : {self.is_notebook}",
            f"  • Storage Tier         : {self.storage_tier.value.upper()}",
            f"  • Temporary Storage    : {self.temp_storage_dir}",
            f"  • Persistent Storage   : {self.persistent_storage_dir}",
            f"  • Holdout Ledger Dir   : {self.ledger_dir}",
            f"  • Rust Kernel Status   : {'Available (Native)' if self.has_rust_kernel else 'Pure-Python Fallback Active'}",
            f"  • CUDA Acceleration    : {'Available (' + (self.cuda_device_name or 'GPU') + ')' if self.has_cuda else 'Disabled / CPU-only'}",
            f"  • CPU Cores / Workers  : {self.cpu_count} cores -> {self.recommended_workers} safe workers",
        ]
        if self.memory_total_gb is not None:
            lines.append(f"  • System Memory        : ~{self.memory_total_gb:.1f} GB")
        if self.warnings:
            lines.append("-" * 68)
            lines.append("  [!] CONFIGURATION & EPISTEMIC WARNINGS:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append("=" * 68)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter / Colab notebooks."""
        tier_color = "#28a745" if self.storage_tier in (StorageTier.PERSISTENT, StorageTier.DRIVE_MOUNTED) else "#dc3545"
        kernel_badge = (
            "<span style='color: #28a745; font-weight: bold;'>Native Rust (Fast)</span>"
            if self.has_rust_kernel
            else "<span style='color: #fd7e14; font-weight: bold;'>Pure Python Fallback (Accurate)</span>"
        )
        cuda_badge = (
            f"<span style='color: #28a745; font-weight: bold;'>CUDA ({self.cuda_device_name})</span>"
            if self.has_cuda
            else "<span style='color: #6c757d;'>CPU Only</span>"
        )

        warning_html = ""
        if self.warnings:
            items = "".join(f"<li style='margin-bottom: 4px;'>{w}</li>" for w in self.warnings)
            warning_html = f"""
            <div style='background-color: #fff3cd; border: 1px solid #ffeeba; color: #856404; padding: 10px; margin-top: 10px; border-radius: 4px;'>
                <strong>Epistemic & Runtime Warnings:</strong>
                <ul style='margin: 5px 0 0 20px; padding: 0;'>{items}</ul>
            </div>
            """

        return f"""
        <div style='font-family: monospace, sans-serif; border: 1px solid #ced4da; border-radius: 6px; padding: 14px; background-color: #f8f9fa; max-width: 720px;'>
            <div style='font-size: 1.1em; font-weight: bold; border-bottom: 2px solid #007bff; padding-bottom: 6px; margin-bottom: 10px;'>
                APEDP-X / CPMD Runtime: <span style='color: #007bff;'>{self.platform.value.upper()}</span>
            </div>
            <table style='width: 100%; border-collapse: collapse; font-size: 0.95em;'>
                <tr><td style='padding: 4px 0;'><strong>Storage Tier:</strong></td><td><span style='color: {tier_color}; font-weight: bold;'>{self.storage_tier.value.upper()}</span></td></tr>
                <tr><td style='padding: 4px 0;'><strong>Temporary Storage:</strong></td><td><code>{self.temp_storage_dir}</code></td></tr>
                <tr><td style='padding: 4px 0;'><strong>Holdout Ledger Dir:</strong></td><td><code>{self.ledger_dir}</code></td></tr>
                <tr><td style='padding: 4px 0;'><strong>Rust Kernel:</strong></td><td>{kernel_badge}</td></tr>
                <tr><td style='padding: 4px 0;'><strong>Compute Hardware:</strong></td><td>{cuda_badge} | {self.cpu_count} CPUs ({self.recommended_workers} safe workers)</td></tr>
            </table>
            {warning_html}
        </div>
        """


def detect_platform() -> PlatformType:
    """Detect whether code is running in Colab, Kaggle, Databricks, Jupyter, CI, or Workstation."""
    # 1. Google Colab
    if "google.colab" in sys.modules:
        return PlatformType.COLAB
    if bool(os.environ.get("COLAB_GPU")) or bool(os.environ.get("GCS_READ_CACHE")):
        return PlatformType.COLAB
    if os.path.exists("/content") and os.path.isdir("/content/sample_data"):
        return PlatformType.COLAB

    # 2. Kaggle
    if bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
        return PlatformType.KAGGLE
    if os.path.exists("/kaggle/working") and os.path.isdir("/kaggle/working"):
        return PlatformType.KAGGLE

    # 3. Databricks (notebooks and clusters set DATABRICKS_RUNTIME_VERSION;
    #    job compute sets DATABRICKS_CLUSTER_ID)
    if bool(os.environ.get("DATABRICKS_RUNTIME_VERSION")) or bool(os.environ.get("DATABRICKS_CLUSTER_ID")):
        return PlatformType.DATABRICKS

    # 4. Continuous Integration
    if os.environ.get("CI") in ("true", "1") or bool(os.environ.get("GITHUB_ACTIONS")):
        return PlatformType.CI

    # 5. HPC / Slurm Cluster
    if bool(os.environ.get("SLURM_JOB_ID")) or bool(os.environ.get("PBS_JOBID")):
        return PlatformType.CLUSTER

    # 6. Interactive Jupyter / IPython notebook outside cloud
    if detect_notebook():
        return PlatformType.JUPYTER

    # 7. Default to local workstation
    return PlatformType.WORKSTATION


def detect_notebook() -> bool:
    """Return True if executing within an interactive Jupyter or IPython notebook kernel."""
    try:
        import IPython  # type: ignore[import-untyped]
        shell = IPython.get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name in ("ZMQInteractiveShell", "Shell"):
            return True
        if "google.colab" in str(shell.__class__):
            return True
        return False
    except Exception as exc:  # noqa: BLE001 — platform probing: any failure means "not a notebook platform"
        logger.debug("notebook platform detection failed: %r", exc)
        return False


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    return detect_platform() == PlatformType.COLAB


def is_kaggle() -> bool:
    """Return True if running inside Kaggle Notebooks."""
    return detect_platform() == PlatformType.KAGGLE


def is_databricks() -> bool:
    """Return True if running inside a Databricks notebook or job cluster."""
    return detect_platform() == PlatformType.DATABRICKS


def mount_google_drive(mount_point: str = "/content/drive", force_remount: bool = False) -> bool:
    """Safely mount Google Drive on Google Colab to preserve holdout ledgers.

    Returns:
        True if Google Drive was successfully mounted; False otherwise.
    """
    if not is_colab():
        logger.info("mount_google_drive called outside Google Colab; skipped.")
        return False

    drive_root = Path(mount_point) / "MyDrive"
    if drive_root.exists() and not force_remount:
        logger.info("Google Drive is already mounted at %s", mount_point)
        return True

    try:
        from google.colab import drive  # type: ignore[import-untyped]
        drive.mount(mount_point, force_remount=force_remount)
        return drive_root.exists()
    except Exception as exc:  # noqa: BLE001 — gdrive mounts raise heterogeneous colab/jupyter errors; bound and reported
        logger.warning("Failed to mount Google Drive: %s", exc)
        return False


def check_rust_kernel_available() -> bool:
    """Check if the native `apedp_kernel` PyO3 extension is importable."""
    try:
        import apedp_kernel  # type: ignore[import-not-found]
        return bool(apedp_kernel is not None)
    except (ImportError, ModuleNotFoundError):
        return False


def check_cuda_available() -> tuple[bool, str | None]:
    """Check if PyTorch with CUDA acceleration is available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, device_name
        return False, None
    except Exception as exc:  # noqa: BLE001 — probe of optional torch/CUDA; absence is the normal outcome
        logger.debug("GPU probe unavailable: %r", exc)
        return False, None


def get_recommended_worker_count(platform: PlatformType | None = None, task_type: str = "cpu") -> int:
    """Calculate safe worker count for multiprocessing, accounting for notebook fork constraints."""
    platform = platform or detect_platform()
    cpu_count = os.cpu_count() or 1

    if platform == PlatformType.COLAB:
        # Colab typically has 2 vCPUs; multiple forks can cause ZeroMQ or CUDA context locks
        return 1 if task_type in ("cuda", "gpu") else min(2, cpu_count)
    elif platform == PlatformType.KAGGLE:
        # Kaggle typically has 4 vCPUs
        return 2 if task_type in ("cuda", "gpu") else min(4, cpu_count)
    elif platform == PlatformType.DATABRICKS:
        # Driver-node forks can deadlock against Spark Connect / ZMQ, and GPU
        # clusters risk duplicated CUDA contexts on fork.
        return 1 if task_type in ("cuda", "gpu") else max(1, min(cpu_count, 16))
    elif platform in (PlatformType.JUPYTER, PlatformType.CI):
        return min(4, cpu_count)
    else:
        # Dedicated workstation or cluster
        return max(1, min(cpu_count, 16))


def _safe_mkdir_or_fallback(target: Path, fallback: Path) -> Path:
    """Attempt to create target directory; fall back safely if permissions or OS error occur."""
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def configure_storage(
    temp_dir: str | Path | None = None,
    persistent_dir: str | Path | None = None,
    allow_drive_mount: bool = False,
    mount_point: str = "/content/drive",
) -> tuple[Path, Path, StorageTier, list[str]]:
    """Configure and create temporary storage and persistent holdout directories.

    Order of precedence for temporary storage:
      1. Explicit `temp_dir` argument
      2. Environment variable `APEDP_TEMP_STORAGE`
      3. Colab: `/content/apedp_temp` (falling back to `<tempfile.gettempdir()>/apedp/temp` if unwritable)
      4. Kaggle: `/kaggle/temp` (falling back to `<tempfile.gettempdir()>/apedp/temp` if unwritable)
      5. Databricks: `/local_disk0/apedp_temp` driver-local scratch (falling back to system temp if absent)
      6. Standard system temp: `<tempfile.gettempdir()>/apedp`

    Order of precedence for persistent storage & holdout ledger:
      1. Explicit `persistent_dir` argument
      2. Environment variable `APEDP_STORAGE_DIR`
      3. Colab: `/content/drive/MyDrive/apedp_storage` (if Drive mounted) or `/content/apedp_storage`
      4. Kaggle: `/kaggle/working/apedp_storage`
      5. Databricks: `/dbfs/FileStore/apedp_storage` when the DBFS FUSE mount is live
      6. Workstation / Local: `<workspace>/artifacts/storage` or `~/.apedp/storage`

    Returns:
        (resolved_temp_dir, resolved_persistent_dir, storage_tier, warnings)
    """
    platform = detect_platform()
    warnings: list[str] = []

    # 1. Resolve Temporary Storage
    if temp_dir is not None:
        t_dir = Path(temp_dir).expanduser().resolve()
        t_dir.mkdir(parents=True, exist_ok=True)
    elif "APEDP_TEMP_STORAGE" in os.environ:
        t_dir = Path(os.environ["APEDP_TEMP_STORAGE"]).expanduser().resolve()
        t_dir.mkdir(parents=True, exist_ok=True)
    elif platform == PlatformType.COLAB:
        colab_temp = Path("/content/apedp_temp")
        system_temp = Path(tempfile.gettempdir()) / "apedp" / "temp"
        t_dir = _safe_mkdir_or_fallback(colab_temp, system_temp)
    elif platform == PlatformType.KAGGLE:
        kaggle_temp = Path("/kaggle/temp")
        system_temp = Path(tempfile.gettempdir()) / "apedp" / "temp"
        t_dir = _safe_mkdir_or_fallback(kaggle_temp, system_temp)
    elif platform == PlatformType.DATABRICKS:
        driver_scratch = Path("/local_disk0/apedp_temp")
        system_temp = Path(tempfile.gettempdir()) / "apedp" / "temp"
        t_dir = _safe_mkdir_or_fallback(driver_scratch, system_temp)
    else:
        system_temp = Path(tempfile.gettempdir()) / "apedp"
        system_temp.mkdir(parents=True, exist_ok=True)
        t_dir = system_temp

    os.environ["APEDP_TEMP_STORAGE"] = str(t_dir)
    # Point standard library tempfile to the configured directory
    tempfile.tempdir = str(t_dir)

    # 2. Resolve Persistent Storage & Holdout Ledger
    tier = StorageTier.PERSISTENT
    if persistent_dir is not None:
        p_dir = Path(persistent_dir).expanduser().resolve()
        p_dir.mkdir(parents=True, exist_ok=True)
    elif "APEDP_STORAGE_DIR" in os.environ:
        p_dir = Path(os.environ["APEDP_STORAGE_DIR"]).expanduser().resolve()
        p_dir.mkdir(parents=True, exist_ok=True)
    elif platform == PlatformType.COLAB:
        drive_path = Path(mount_point) / "MyDrive"
        if not drive_path.exists() and allow_drive_mount:
            mount_google_drive(mount_point)

        if drive_path.exists():
            drive_storage = drive_path / "apedp_storage"
            system_persist = Path(tempfile.gettempdir()) / "apedp" / "storage"
            p_dir = _safe_mkdir_or_fallback(drive_storage, system_persist)
            tier = StorageTier.DRIVE_MOUNTED
        else:
            colab_storage = Path("/content/apedp_storage")
            system_persist = Path(tempfile.gettempdir()) / "apedp" / "storage"
            p_dir = _safe_mkdir_or_fallback(colab_storage, system_persist)
            tier = StorageTier.EPHEMERAL
            warnings.append(
                "EPISTEMIC RISK: Running in Google Colab without mounted Google Drive! "
                "The holdout experiment ledger is located in ephemeral storage (/content/apedp_storage). "
                "Under AGENTS.md §1.2, confirmation holdouts burned during hypothesis falsification "
                "must remain permanently recorded to prevent data snooping. If this Colab runtime "
                "restarts, burned holdout slices will be lost. To safeguard epistemic integrity, "
                "call `mount_google_drive()` or set `APEDP_STORAGE_DIR` to a persistent path."
            )
    elif platform == PlatformType.KAGGLE:
        kaggle_storage = Path("/kaggle/working/apedp_storage")
        system_persist = Path(tempfile.gettempdir()) / "apedp" / "storage"
        p_dir = _safe_mkdir_or_fallback(kaggle_storage, system_persist)
        tier = StorageTier.EPHEMERAL
        warnings.append(
            "EPHEMERAL STORAGE NOTICE: Running on Kaggle Notebooks. Files written to /kaggle/working "
            "are preserved only if you 'Save & Run All' (commit). If this interactive session is closed, "
            "uncommitted holdout ledgers will be erased."
        )
    elif platform == PlatformType.DATABRICKS:
        dbfs_root = Path("/dbfs/FileStore")
        system_persist = Path(tempfile.gettempdir()) / "apedp" / "storage"
        if dbfs_root.is_dir():
            p_dir = _safe_mkdir_or_fallback(dbfs_root / "apedp_storage", system_persist)
            tier = StorageTier.PERSISTENT
        else:
            driver_storage = Path("/databricks/driver/apedp_storage")
            p_dir = _safe_mkdir_or_fallback(driver_storage, system_persist)
            tier = StorageTier.EPHEMERAL
            warnings.append(
                "EPISTEMIC RISK: Running on Databricks without a reachable DBFS mount "
                "(/dbfs/FileStore). The holdout experiment ledger is located in driver-local "
                "storage, which is erased when the cluster terminates. Under AGENTS.md §1.2, "
                "burned confirmation holdouts must remain permanently recorded to prevent data "
                "snooping. Set `APEDP_STORAGE_DIR` to a Unity Catalog Volume, a DBFS path, or "
                "a cloud object store (e.g. s3://, gs://, abfss://) to keep burned holdouts."
            )
    else:
        # Dedicated workstation, cluster, or CI
        repo_root = Path(__file__).resolve().parent.parent
        workstation_storage = repo_root / "artifacts" / "storage"
        system_persist = Path(tempfile.gettempdir()) / "apedp" / "storage"
        p_dir = _safe_mkdir_or_fallback(workstation_storage, system_persist)
        tier = StorageTier.PERSISTENT

    os.environ["APEDP_STORAGE_DIR"] = str(p_dir)

    # If APEDP_LEDGER_PATH is not already set, configure it to the persistent dir
    if "APEDP_LEDGER_PATH" not in os.environ:
        os.environ["APEDP_LEDGER_PATH"] = str(p_dir / "experiment_ledger.jsonl")

    return t_dir, p_dir, tier, warnings


def ensure_kernel(
    allow_pure_python_fallback: bool = True,
    precompiled_wheel_url: str | None = None,
    auto_compile: bool = False,
) -> dict[str, Any]:
    """Verify `apedp_kernel` availability or configure transparent pure-Python fallback.

    Parameters:
        allow_pure_python_fallback: If True, allow bit-identical Python/NumPy fallback.
        precompiled_wheel_url: Optional direct URL to pre-compiled binary wheel.
        auto_compile: If True and cargo is installed, attempt building via maturin.

    Returns:
        Dictionary detailing kernel status, fallback status, and actions taken.
    """
    has_rust = check_rust_kernel_available()
    result: dict[str, Any] = {
        "rust_kernel_available": has_rust,
        "pure_python_fallback_active": not has_rust and allow_pure_python_fallback,
        "wheel_installed": False,
        "compiled": False,
        "message": "",
    }

    if has_rust:
        result["message"] = "Native apedp_kernel extension is already installed and functional."
        return result

    # Try pre-compiled wheel if provided
    if precompiled_wheel_url:
        try:
            logger.info("Attempting to install pre-compiled wheel from %s", precompiled_wheel_url)
            subprocess.check_call(  # noqa: S603 — installs the operator-supplied wheel URL by design; argv list is fixed
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", precompiled_wheel_url]
            )
            if check_rust_kernel_available():
                result["rust_kernel_available"] = True
                result["pure_python_fallback_active"] = False
                result["wheel_installed"] = True
                result["message"] = "Successfully installed pre-compiled apedp_kernel wheel."
                return result
        except Exception as exc:  # noqa: BLE001 — wheel install: pip/packaging errors are heterogeneous by design
            logger.warning("Failed to install precompiled wheel: %s", exc)

    # Try auto-compiling if requested and cargo exists
    if auto_compile and shutil.which("cargo") is not None:
        try:
            logger.info("Attempting maturin compile for rust/apedp-kernel...")
            repo_root = Path(__file__).resolve().parent.parent
            rust_dir = repo_root / "rust" / "apedp-kernel"
            if not rust_dir.exists():
                rust_dir = Path.cwd() / "rust" / "apedp-kernel"
            if rust_dir.exists():
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "maturin"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                maturin_path = shutil.which("maturin")
                if maturin_path is None:
                    raise RuntimeError("maturin not found on PATH after pip install")
                subprocess.check_call(  # noqa: S603 — maturin resolved via shutil.which; argv is a fixed list
                    [maturin_path, "develop", "--release"],
                    cwd=str(rust_dir),
                )
                if check_rust_kernel_available():
                    result["rust_kernel_available"] = True
                    result["pure_python_fallback_active"] = False
                    result["compiled"] = True
                    result["message"] = "Successfully built apedp_kernel via maturin."
                    return result
        except Exception as exc:  # noqa: BLE001 — maturin/cargo/packaging errors are heterogeneous by design
            logger.warning("Auto-compilation of apedp_kernel failed: %s", exc)

    if allow_pure_python_fallback:
        result["message"] = (
            "apedp_kernel not found. Operating under transparent pure-Python / NumPy "
            "reference fallback with verified 1e-12 mathematical parity."
        )
        return result

    raise ImportError(
        "apedp_kernel is required but not installed, and allow_pure_python_fallback is False."
    )


@contextmanager
def temporary_storage_scope(
    custom_temp_dir: str | Path | None = None,
    clean_on_exit: bool = True,
) -> Generator[Path, None, None]:
    """Context manager for temporary, isolated storage scoped to a run or test.

    Restores previous environment variables and tempfile directories upon exit.
    """
    old_temp_env = os.environ.get("APEDP_TEMP_STORAGE")
    old_tempfile_dir = tempfile.tempdir

    if custom_temp_dir is not None:
        scoped_dir = Path(custom_temp_dir).expanduser().resolve()
        scoped_dir.mkdir(parents=True, exist_ok=True)
        created_temp = False
    else:
        scoped_dir = Path(tempfile.mkdtemp(prefix="apedp_scope_"))
        created_temp = True

    os.environ["APEDP_TEMP_STORAGE"] = str(scoped_dir)
    tempfile.tempdir = str(scoped_dir)

    try:
        yield scoped_dir
    finally:
        if old_temp_env is not None:
            os.environ["APEDP_TEMP_STORAGE"] = old_temp_env
        else:
            os.environ.pop("APEDP_TEMP_STORAGE", None)
        tempfile.tempdir = old_tempfile_dir

        if clean_on_exit and created_temp and scoped_dir.exists():
            shutil.rmtree(scoped_dir, ignore_errors=True)


def init_environment(
    temp_dir: str | Path | None = None,
    persistent_dir: str | Path | None = None,
    allow_drive_mount: bool = False,
    allow_pure_python_fallback: bool = True,
    mount_gdrive: bool | None = None,
    auto_compile_kernel: bool = False,
    quiet: bool = False,
    **kwargs: Any,
) -> NotebookRuntimeConfig:
    """Universal initialization function for notebooks, cloud runners, and local workstations.

    Configures temporary storage, initializes holdout persistence directories, checks
    kernel and hardware acceleration, and issues epistemic discipline warnings.

    Example usage in Colab or Jupyter:
    >>> from apedp_dyn.notebook_adapter import init_environment
    >>> env = init_environment(allow_drive_mount=True)
    """
    if mount_gdrive is not None:
        allow_drive_mount = mount_gdrive

    platform = detect_platform()
    is_nb = detect_notebook()

    # Configure storage
    t_dir, p_dir, tier, warnings = configure_storage(
        temp_dir=temp_dir,
        persistent_dir=persistent_dir,
        allow_drive_mount=allow_drive_mount,
    )

    # Check kernel status
    kernel_status = ensure_kernel(
        allow_pure_python_fallback=allow_pure_python_fallback,
        auto_compile=auto_compile_kernel,
    )
    has_rust = kernel_status["rust_kernel_available"]
    if not has_rust:
        warnings.append(
            "Native Rust kernel (apedp_kernel) is not installed. System will use pure-Python "
            "reference engines. Symbolic & PDE computations will run slower, but yield identical math."
        )

    # Hardware detection
    has_cuda, cuda_name = check_cuda_available()
    cpu_count = os.cpu_count() or 1
    safe_workers = get_recommended_worker_count(platform=platform)

    # System memory detection
    mem_gb: float | None = None
    try:
        import psutil  # type: ignore[import-untyped]
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception as exc:  # noqa: BLE001 — optional psutil; fallback reads /proc/meminfo below
        logger.debug("psutil unavailable (%r); falling back to /proc/meminfo", exc)
        # Fallback reading /proc/meminfo on Linux
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = float(line.split()[1])
                            mem_gb = kb / (1024 * 1024)
                            break
        except Exception as exc2:  # noqa: BLE001 — best-effort memory reporting; N/A elsewhere
            logger.debug("/proc/meminfo read failed: %r", exc2)

    config = NotebookRuntimeConfig(
        platform=platform,
        is_notebook=is_nb,
        temp_storage_dir=t_dir,
        persistent_storage_dir=p_dir,
        ledger_dir=p_dir,
        storage_tier=tier,
        has_rust_kernel=has_rust,
        has_cuda=has_cuda,
        cuda_device_name=cuda_name,
        cpu_count=cpu_count,
        recommended_workers=safe_workers,
        memory_total_gb=mem_gb,
        warnings=warnings,
    )

    if not quiet:
        logger.info(config.summary())

    return config
