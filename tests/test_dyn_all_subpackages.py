"""The domain APIs, asserted against the modules that own them.

Six `packages/apedp-*` shells (`audio`, `continuum`, `distributed`, `molecular`,
`quantum`, `robotics`) used to live here as `pip` names. None held an
implementation: each `__init__.py` inserted the repo root on `sys.path` and
re-exported from `apedp_dyn`, with `apedp-continuum` adding two forwarding module
files. A wheel that ships a second name for code the core already ships is a
claim about an artifact nobody maintains, so the shells, their `wheels.yml`
matrix entries and their dependabot configs were deleted on 2026-11-06.
`packages/` now holds `jetflow-video` only, which is standalone by contract, and
`tests/test_scaffold_refusals.py` pins the absence so they cannot creep back.

What is tested here is the behaviour those wheels exposed, now asserted against
the module that implements it:

1. quantum DMRG (`apedp_dyn.quantum.dmrg`)
2. robotics unicycle kinematics (`apedp_dyn.robotics.trajectory_transport`)
3. molecular SE(3) flow (`apedp_dyn.molecular.se3_flow`)
4. audio spectrogram transport (`apedp_dyn.audio.spectrogram_transport`)
5. distributed chunked execution (`apedp_dyn.distributed.chunked_executor`)
6. continuum ODE/PDE discovery (`apedp_dyn.ode`, `apedp_dyn.pde`)
7. the one remaining package, `jetflow-video`, stays fully self-contained
8. the `apedp_dyn.kernel_shim` sentinel keeps `import apedp_dyn` working when
   the native `apedp_kernel` extension is absent, and kernel-only engines fail
   their *calls* with an actionable RuntimeError rather than crashing the import
   chain.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

PACKAGES_DIR = REPO_ROOT / "packages"


def test_quantum_dmrg_api():
    from apedp_dyn.quantum.dmrg import (
        MatrixProductState,
        dmrg_ground_state,
        heisenberg_exact_ground_energy,
        heisenberg_mpo,
    )

    assert MatrixProductState is not None
    assert callable(heisenberg_mpo)

    e_exact = heisenberg_exact_ground_energy(n_sites=4, j_coupling=1.0, field=0.0)
    res, _mps = dmrg_ground_state(n_sites=4, j_coupling=1.0, field=0.0, max_bond_dim=10, max_sweeps=2)
    assert abs(res.energy - e_exact) < 1e-4


def test_robotics_unicycle_api():
    from apedp_dyn.robotics.trajectory_transport import (
        exact_unicycle_step,
        navigate_to_goal,
        wrap_angle,
    )

    assert callable(navigate_to_goal)
    assert wrap_angle(3.5 * np.pi) == -0.5 * np.pi

    pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    next_pose = exact_unicycle_step(pose, linear_velocity=1.0, angular_velocity=0.0, dt=1.0)
    assert abs(next_pose[0] - 1.0) < 1e-9
    assert abs(next_pose[1]) < 1e-9
    assert abs(next_pose[2]) < 1e-9


def test_molecular_se3_flow_api():
    from apedp_dyn.molecular.se3_flow import (
        MolecularForceField,
        energy,
        energy_gradient,
        equivariance_error,
        relax,
    )

    assert callable(energy) and callable(energy_gradient) and callable(equivariance_error)

    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    ff = MolecularForceField(bonds=((0, 1),), target_length=1.0)
    res = relax(coords, force_field=ff, n_steps=50, step_size=0.1)
    d = float(np.linalg.norm(res.coordinates[0] - res.coordinates[1]))
    assert abs(d - 1.0) < 1e-2


def test_audio_spectrogram_transport_api():
    from apedp_dyn.audio.spectrogram_transport import (
        ChirpSpec,
        ridge_instantaneous_frequency,
        stft_magnitude,
        synthesize_chirps,
    )

    assert callable(stft_magnitude) and callable(ridge_instantaneous_frequency)

    spec = ChirpSpec(start_freq=100.0, linear_rate=100.0)
    sig = synthesize_chirps([spec], sample_rate=1000.0, duration=0.25)
    assert len(sig) == 250


def test_distributed_chunked_executor_api():
    from apedp_dyn.distributed.chunked_executor import (
        DistributedChunkedExecutor,
        PartitionSpec,
        square_worker,
    )

    assert PartitionSpec is not None

    executor = DistributedChunkedExecutor(backend="serial")
    out = list(executor.map_stream(square_worker, [1, 2, 3], chunk_size=2))
    assert out == [1, 4, 9]


def test_continuum_discovery_api():
    """`apedp-continuum` forwarded `apedp_dyn.ode` and `apedp_dyn.pde` verbatim."""
    from apedp_dyn.ode import discover_ode
    from apedp_dyn.pde import discover_pde_1d

    assert callable(discover_ode)
    assert callable(discover_pde_1d)


def test_packages_holds_only_the_standalone_video_package():
    """`packages/` is not a place to publish a second name for core code."""
    remaining = sorted(p.name for p in PACKAGES_DIR.iterdir() if p.is_dir())
    assert remaining == ["jetflow-video"], remaining

    pkg = PACKAGES_DIR / "jetflow-video"
    for name in ("pyproject.toml", "setup.py", "README.md"):
        assert (pkg / name).is_file(), f"Missing {name} in jetflow-video"


def test_jetflow_video_stays_self_contained():
    # jetflow-video is intentionally standalone: no apedp_dyn imports allowed,
    # so its wheel must never grow a hidden core dependency.
    src = REPO_ROOT / "packages" / "jetflow-video" / "jetflow_video"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            module = stripped.split()[1].split(".")[0].split(",")[0]
            # `apedp_kernel` is the native extension, not the core package: comparing
            # the module token (rather than the string prefix) keeps the guard on the
            # dependency it is about, which the stepper's Rust fast path legitimately uses.
            assert module not in {"apedp", "apedp_dyn"}, (
                f"jetflow-video must stay standalone: {path} imports {module}"
            )


def test_kernel_shim_sentinel_is_probe_safe_and_actionable():
    # The sentinel replaces `apedp_kernel` on installs without the native
    # extension. Probing (hasattr/repr) must behave like a missing attribute,
    # while real engine entry points must raise with the install hint.
    from apedp_dyn.kernel_shim import missing_kernel

    assert not hasattr(missing_kernel, "__file__")
    assert "MissingKernel" in repr(missing_kernel)
    try:
        _ = missing_kernel.rime_diff_pade_discover_json
    except RuntimeError as exc:
        assert "apedp_kernel.rime_diff_pade_discover_json" in str(exc)
        assert "maturin" in str(exc), "error must name the concrete install step"
    else:
        raise AssertionError("kernel-only entry point must raise RuntimeError when absent")


def test_kernel_less_import_chain_survives_absent_extension():
    # Simulate a kernel-less install in a subprocess: `None` in sys.modules
    # forces ImportError on `import apedp_kernel` even though the dev env has
    # the extension, so this pins the guard without uninstalling anything.
    # Before the shim, importing apedp_dyn.ode exploded here with a bare
    # ModuleNotFoundError; now the chain imports and only *calling* a
    # kernel-only engine fails, with the actionable message.
    code = (
        "import sys\n"
        "sys.modules['apedp_kernel'] = None\n"
        "from apedp_dyn.ode.dispatch import discover_ode\n"
        "try:\n"
        "    discover_ode([0.0, 1.0, 2.0, 3.0], [1.0, 1.5, 2.0, 2.5], max_degree=1)\n"
        "except RuntimeError as exc:\n"
        "    assert 'apedp-kernel extension is not installed' in str(exc), str(exc)\n"
        "    print('KERNEL_LESS_CHAIN_OK')\n"
        "else:\n"
        "    raise SystemExit('expected RuntimeError from kernel-only engine')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=180, check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "KERNEL_LESS_CHAIN_OK" in proc.stdout
