"""Test Suite for Notebook & Cloud Runtime Adapter and Storage Configuration.

Tests:
1. Platform detection (Colab, Kaggle, Jupyter, CI, Workstation, Cluster).
2. Storage configuration (APEDP_TEMP_STORAGE, /tmp, Google Drive, Kaggle working).
3. Temporary storage scoping context manager.
4. Epistemic anti-snooping safeguards and holdout ledger directory redirection.
5. Integration with FullLineageExperimentLedger and CandidateTrialLedger via from_env.
6. Safe multiprocessing worker count estimation.
7. Kernel status reporting and pure-Python reference fallback behavior.
8. NotebookRuntimeConfig string summary and HTML representation.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import apedp_dyn.env as env
from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.epistemic.trial_ledger import CandidateTrialLedger
from apedp_dyn.notebook_adapter import (
    NotebookRuntimeConfig,
    PlatformType,
    StorageTier,
    check_rust_kernel_available,
    configure_storage,
    detect_notebook,
    detect_platform,
    ensure_kernel,
    get_recommended_worker_count,
    init_environment,
    is_colab,
    is_kaggle,
    mount_google_drive,
    temporary_storage_scope,
)

# ── 1. Platform Detection Tests ──────────────────────────────────────────────

def test_detect_platform_colab_sys_module():
    """Detect Colab when google.colab module is present."""
    with patch.dict(sys.modules, {"google.colab": MagicMock()}):
        assert detect_platform() == PlatformType.COLAB
        assert is_colab() is True
        assert is_kaggle() is False


def test_detect_platform_colab_env_var():
    """Detect Colab when COLAB_GPU env var is set."""
    with patch.dict(os.environ, {"COLAB_GPU": "1"}):
        assert detect_platform() == PlatformType.COLAB
        assert is_colab() is True


def test_detect_platform_kaggle_env_var():
    """Detect Kaggle when KAGGLE_KERNEL_RUN_TYPE is set."""
    with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
        assert detect_platform() == PlatformType.KAGGLE
        assert is_kaggle() is True
        assert is_colab() is False


def test_detect_platform_ci():
    """Detect CI environment."""
    with patch.dict(os.environ, {"CI": "true", "COLAB_GPU": "", "GCS_READ_CACHE": "", "KAGGLE_KERNEL_RUN_TYPE": "", "SLURM_JOB_ID": ""}):
        with patch.dict(sys.modules, {}):
            sys.modules.pop("google.colab", None)
            assert detect_platform() == PlatformType.CI


def test_detect_platform_cluster():
    """Detect Slurm cluster environment."""
    with patch.dict(os.environ, {"SLURM_JOB_ID": "123456", "CI": "", "COLAB_GPU": "", "GCS_READ_CACHE": "", "KAGGLE_KERNEL_RUN_TYPE": ""}):
        with patch.dict(sys.modules, {}):
            sys.modules.pop("google.colab", None)
            assert detect_platform() == PlatformType.CLUSTER



def test_detect_notebook_mocked():
    """Detect interactive Jupyter shell."""
    mock_ipython = MagicMock()
    mock_shell = MagicMock()
    mock_shell.__class__.__name__ = "ZMQInteractiveShell"
    mock_ipython.get_ipython.return_value = mock_shell

    with patch.dict(sys.modules, {"IPython": mock_ipython}):
        assert detect_notebook() is True

    # And when get_ipython returns None (headless terminal / script)
    mock_ipython.get_ipython.return_value = None
    with patch.dict(sys.modules, {"IPython": mock_ipython}):
        assert detect_notebook() is False


# ── 2. Storage Configuration & Temp Storage Tests ────────────────────────────

def test_configure_storage_explicit_paths(tmp_path):
    """Explicit paths take top priority."""
    t_dir = tmp_path / "custom_temp"
    p_dir = tmp_path / "custom_persist"

    resolved_t, resolved_p, tier, warnings = configure_storage(
        temp_dir=t_dir,
        persistent_dir=p_dir,
    )
    assert resolved_t == t_dir
    assert resolved_p == p_dir
    assert resolved_t.is_dir()
    assert resolved_p.is_dir()
    assert os.environ["APEDP_TEMP_STORAGE"] == str(t_dir)
    assert os.environ["APEDP_STORAGE_DIR"] == str(p_dir)
    assert tempfile.tempdir == str(t_dir)


def test_configure_storage_env_vars(tmp_path):
    """Environment variables APEDP_TEMP_STORAGE and APEDP_STORAGE_DIR are honored."""
    t_dir = tmp_path / "env_temp"
    p_dir = tmp_path / "env_persist"

    with patch.dict(os.environ, {
        "APEDP_TEMP_STORAGE": str(t_dir),
        "APEDP_STORAGE_DIR": str(p_dir),
    }):
        resolved_t, resolved_p, tier, warnings = configure_storage()
        assert resolved_t == t_dir
        assert resolved_p == p_dir
        assert resolved_t.is_dir()
        assert resolved_p.is_dir()


def test_configure_storage_colab_epistemic_warning(tmp_path):
    """In Colab without mounted Drive, issues holdout anti-snooping epistemic warning."""
    with patch.dict(sys.modules, {"google.colab": MagicMock()}):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APEDP_STORAGE_DIR", None)
            os.environ.pop("APEDP_TEMP_STORAGE", None)
            # Simulate non-existent Google Drive
            t_dir, p_dir, tier, warnings = configure_storage(
                mount_point=str(tmp_path / "non_existent_drive"),
                allow_drive_mount=False,
            )
            assert tier == StorageTier.EPHEMERAL
            assert any("EPISTEMIC RISK" in w for w in warnings)
            assert any("AGENTS.md §1.2" in w for w in warnings)


def test_configure_storage_colab_with_mounted_drive(tmp_path):
    """In Colab with mounted Drive, tier is DRIVE_MOUNTED with zero epistemic warnings."""
    drive_mock = tmp_path / "drive" / "MyDrive"
    drive_mock.mkdir(parents=True)

    with patch.dict(sys.modules, {"google.colab": MagicMock()}):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APEDP_STORAGE_DIR", None)
            os.environ.pop("APEDP_TEMP_STORAGE", None)
            t_dir, p_dir, tier, warnings = configure_storage(
                mount_point=str(tmp_path / "drive"),
                allow_drive_mount=False,
            )
            assert tier == StorageTier.DRIVE_MOUNTED
            assert p_dir == drive_mock / "apedp_storage"
            assert not any("EPISTEMIC RISK" in w for w in warnings)


def test_mount_google_drive_outside_colab():
    """Calling mount_google_drive outside Colab returns False safely."""
    with patch("apedp_dyn.notebook_adapter.is_colab", return_value=False):
        assert mount_google_drive() is False


# ── 3. Temporary Storage Scope Context Manager ───────────────────────────────

def test_temporary_storage_scope_creates_and_cleans():
    """temporary_storage_scope isolates tempdir and cleans up automatically."""
    prev_env = os.environ.get("APEDP_TEMP_STORAGE")
    prev_tempfile = tempfile.tempdir

    created_path = None
    with temporary_storage_scope() as temp_dir:
        created_path = temp_dir
        assert temp_dir.exists()
        assert os.environ["APEDP_TEMP_STORAGE"] == str(temp_dir)
        assert tempfile.tempdir == str(temp_dir)
        # Write dummy file
        (temp_dir / "test.txt").write_text("hello", encoding="utf-8")
        assert (temp_dir / "test.txt").exists()

    # After exit, directory is cleaned and environment restored
    assert not created_path.exists()
    assert os.environ.get("APEDP_TEMP_STORAGE") == prev_env
    assert tempfile.tempdir == prev_tempfile


def test_temporary_storage_scope_custom_dir(tmp_path):
    """Custom temp dir preserves user data when clean_on_exit is False."""
    custom = tmp_path / "my_custom_scope"
    with temporary_storage_scope(custom_temp_dir=custom, clean_on_exit=False) as s_dir:
        assert s_dir == custom
        (s_dir / "data.csv").write_text("a,b\n1,2", encoding="utf-8")

    assert custom.exists()
    assert (custom / "data.csv").exists()


# ── 4. Safe Multiprocessing Worker Counts ────────────────────────────────────

def test_recommended_worker_count():
    """Worker counts respect platform-specific concurrency limitations."""
    assert get_recommended_worker_count(PlatformType.COLAB, task_type="cpu") in (1, 2)
    assert get_recommended_worker_count(PlatformType.COLAB, task_type="gpu") == 1
    assert get_recommended_worker_count(PlatformType.KAGGLE, task_type="cpu") in (1, 2, 3, 4)
    assert get_recommended_worker_count(PlatformType.KAGGLE, task_type="gpu") == 2
    assert get_recommended_worker_count(PlatformType.WORKSTATION, task_type="cpu") >= 1


# ── 5. Kernel Status & Pure-Python Fallback Verification ─────────────────────

def test_check_rust_kernel_status():
    """check_rust_kernel_available returns a valid boolean."""
    status = check_rust_kernel_available()
    assert isinstance(status, bool)


def test_ensure_kernel_pure_python_fallback():
    """ensure_kernel with pure Python fallback permitted never raises."""
    with patch("apedp_dyn.notebook_adapter.check_rust_kernel_available", return_value=False):
        res = ensure_kernel(allow_pure_python_fallback=True)
        assert res["rust_kernel_available"] is False
        assert res["pure_python_fallback_active"] is True
        assert "pure-Python" in res["message"]


def test_ensure_kernel_strict_raises_when_missing():
    """ensure_kernel raises ImportError if kernel is missing and fallback is forbidden."""
    with patch("apedp_dyn.notebook_adapter.check_rust_kernel_available", return_value=False):
        with pytest.raises(ImportError, match="apedp_kernel is required"):
            ensure_kernel(allow_pure_python_fallback=False)


# ── 6. NotebookRuntimeConfig Representation ──────────────────────────────────

def test_runtime_config_summary_and_html(tmp_path):
    """Runtime config generates informative plain text summary and HTML widget."""
    cfg = NotebookRuntimeConfig(
        platform=PlatformType.COLAB,
        is_notebook=True,
        temp_storage_dir=tmp_path / "temp",
        persistent_storage_dir=tmp_path / "persist",
        ledger_dir=tmp_path / "persist",
        storage_tier=StorageTier.DRIVE_MOUNTED,
        has_rust_kernel=False,
        has_cuda=False,
        cpu_count=2,
        recommended_workers=2,
        memory_total_gb=12.5,
        warnings=["Holdout ledger caution"],
    )

    summary_text = cfg.summary()
    assert "COLAB" in summary_text
    assert "DRIVE_MOUNTED" in summary_text
    assert "Pure-Python Fallback Active" in summary_text
    assert "Holdout ledger caution" in summary_text

    html = cfg._repr_html_()
    assert "<table" in html
    assert "COLAB" in html
    assert "Holdout ledger caution" in html
    assert "Pure Python Fallback" in html


# ── 7. Integration with FullLineageExperimentLedger & CandidateTrialLedger ───

def test_experiment_ledger_from_env_persistence(tmp_path):
    """FullLineageExperimentLedger.from_env() persists and recovers holdout state."""
    ledger_path = tmp_path / "exp_ledger.jsonl"
    with patch.dict(os.environ, {"APEDP_LEDGER_PATH": str(ledger_path)}):
        ledger = FullLineageExperimentLedger.from_env()

        # Register and fail candidate on confirmation holdout
        ledger.register_candidate(
            candidate_id="cand_ephemeral_1",
            proposer_id="sparse_linear",
            formula="x + y",
        )
        ledger.record_data_exposure(
            candidate_id="cand_ephemeral_1",
            dataset_id="btc_lob_2025",
            slice_identifier="slice_confirmation_01",
            role=DatasetRole.CONFIRMATION_HOLDOUT,
            outcome_status="FAILED",
        )
        ledger.update_lifecycle_state(
            candidate_id="cand_ephemeral_1",
            new_state=CandidateLifecycleState.CONFIRMATION_REJECTED,
            rejection_reason="Out-of-sample overfit",
        )

        # Check burned holdout slice is recorded
        burned = ledger.get_burned_holdouts()
        assert ("btc_lob_2025", "slice_confirmation_01") in burned

        # Simulate reload from environment path
        reloaded = FullLineageExperimentLedger.from_env()
        cand = reloaded.get_candidate("cand_ephemeral_1")
        assert cand.state == CandidateLifecycleState.CONFIRMATION_REJECTED
        assert reloaded.get_burned_holdouts() == burned

        # Attempting to reuse burned holdout on a new candidate MUST raise HoldoutContaminationError
        reloaded.register_candidate(
            candidate_id="cand_replacement_2",
            proposer_id="sparse_linear",
            formula="2*x + y",
        )
        with pytest.raises(HoldoutContaminationError, match="DATA SNOOPING VIOLATION"):
            reloaded.record_data_exposure(
                candidate_id="cand_replacement_2",
                dataset_id="btc_lob_2025",
                slice_identifier="slice_confirmation_01",
                role=DatasetRole.CONFIRMATION_HOLDOUT,
            )


def test_candidate_trial_ledger_from_env(tmp_path):
    """CandidateTrialLedger.from_env() initializes at env persistent storage path."""
    with patch.dict(os.environ, {"APEDP_STORAGE_DIR": str(tmp_path)}):
        trial_ledger = CandidateTrialLedger.from_env(filename="trial_audit.jsonl")
        trial_ledger.record_trial(
            candidate_id="cand_trial_1",
            proposer_id="rime_proposer",
            formula="sin(x)",
            strategy_archetype="rime_flow",
            in_sample_sharpe=0.92,
            metadata={"generation": 1, "complexity": 2},
        )
        assert (tmp_path / "trial_audit.jsonl").exists()
        reloaded = CandidateTrialLedger.from_env(filename="trial_audit.jsonl")
        assert len(reloaded.get_records()) == 1
        assert reloaded.trial_count == 1


# ── 8. apedp_dyn.env Interface Helpers ───────────────────────────────────────

def test_env_helpers(tmp_path):
    """apedp_dyn.env accessors work predictably."""
    with patch.dict(os.environ, {
        "APEDP_TEMP_STORAGE": str(tmp_path / "temp_env"),
        "APEDP_STORAGE_DIR": str(tmp_path / "storage_env"),
    }):
        assert env.get_temp_storage_dir() == tmp_path / "temp_env"
        assert env.get_persistent_storage_dir() == tmp_path / "storage_env"
        assert env.get_ledger_path("ledger.jsonl") == tmp_path / "storage_env" / "ledger.jsonl"
        cfg = env.get_runtime_config(quiet=True)
        assert isinstance(cfg, NotebookRuntimeConfig)


# ── 9. Robustness & Edge Cases ───────────────────────────────────────────────

def test_configure_storage_kaggle():
    """Kaggle environment storage defaults and epistemic notice."""
    with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, clear=True):
        with patch.dict(sys.modules, {}):
            sys.modules.pop("google.colab", None)
            t_dir, p_dir, tier, warnings = configure_storage()
            assert tier == StorageTier.EPHEMERAL
            assert any("Kaggle" in w for w in warnings)


def test_temporary_storage_scope_exception_cleanup():
    """Temporary storage directory is purged even if an unhandled exception occurs."""
    leaked_path = None
    with pytest.raises(RuntimeError, match="Intentional failure inside context"):
        with temporary_storage_scope(clean_on_exit=True) as temp_dir:
            leaked_path = temp_dir
            assert temp_dir.exists()
            (temp_dir / "crash_test.dat").write_bytes(b"data")
            raise RuntimeError("Intentional failure inside context")

    assert leaked_path is not None
    assert not leaked_path.exists()


def test_init_environment_full_integration(tmp_path):
    """init_environment properly initializes and configures system."""
    with patch.dict(os.environ, {
        "APEDP_TEMP_STORAGE": str(tmp_path / "temp"),
        "APEDP_STORAGE_DIR": str(tmp_path / "storage"),
    }):
        config = init_environment(quiet=True)
        assert isinstance(config, NotebookRuntimeConfig)
        assert config.temp_storage_dir == tmp_path / "temp"
        assert config.persistent_storage_dir == tmp_path / "storage"
        assert config.recommended_workers >= 1
        assert config.get_ledger_path().name == "experiment_ledger.jsonl"


def test_ensure_kernel_wheel_install_mocked():
    """Mock successful wheel installation via pip."""
    with patch("apedp_dyn.notebook_adapter.check_rust_kernel_available", side_effect=[False, True]):
        with patch("subprocess.check_call") as mock_pip:
            res = ensure_kernel(precompiled_wheel_url="https://example.com/apedp_kernel.whl")
            assert res["rust_kernel_available"] is True
            assert res["wheel_installed"] is True
            assert mock_pip.called


def test_tutorial_notebook_structure_and_execution(tmp_path):
    """Verify that notebooks/apedp_scientific_discovery_tutorial.ipynb is valid nbformat 4 and runs."""
    import json
    nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "apedp_scientific_discovery_tutorial.ipynb"
    assert nb_path.is_file(), f"Tutorial notebook not found at {nb_path}"

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10

    # Execute cells in isolated test environment with temporary directories
    with patch.dict(os.environ, {
        "APEDP_TEMP_STORAGE": str(tmp_path / "temp"),
        "APEDP_STORAGE_DIR": str(tmp_path / "storage"),
    }):
        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                exec(code, global_ns)


