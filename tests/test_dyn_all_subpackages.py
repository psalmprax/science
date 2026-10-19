"""Tests verifying standalone pip packaging and public API exposure across all domain subpackages.

Verifies:
1. apedp-continuum (apedp_continuum)
2. jetflow-video (jetflow_video)
3. apedp-quantum (apedp_quantum)
4. apedp-robotics (apedp_robotics)
5. apedp-molecular (apedp_molecular)
6. apedp-audio (apedp_audio)
7. apedp-distributed (apedp_distributed)
8. Packaging contract: every domain wrapper declares the `apedp` core dependency
   (the wrappers re-export from apedp_dyn, so installing one without the core
   yields ModuleNotFoundError at first import); jetflow-video is guarded to
   stay fully self-contained.
9. Kernel-less installs: the `apedp_dyn.kernel_shim` sentinel keeps
   `import apedp_dyn` working when the native `apedp_kernel` extension is
   absent, and kernel-only engines fail their *calls* with an actionable
   RuntimeError instead of crashing the import chain.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Register all package directories in sys.path
for pkg in (
    "apedp-continuum",
    "jetflow-video",
    "apedp-quantum",
    "apedp-robotics",
    "apedp-molecular",
    "apedp-audio",
    "apedp-distributed",
):
    pkg_dir = REPO_ROOT / "packages" / pkg
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))


def test_apedp_quantum_package_api():
    import apedp_quantum as aq

    assert hasattr(aq, "dmrg_ground_state")
    assert hasattr(aq, "heisenberg_exact_ground_energy")
    assert hasattr(aq, "MatrixProductState")
    assert hasattr(aq, "heisenberg_mpo")

    # Verify functionality through package
    e_exact = aq.heisenberg_exact_ground_energy(n_sites=4, j_coupling=1.0, field=0.0)
    res, _mps = aq.dmrg_ground_state(n_sites=4, j_coupling=1.0, field=0.0, max_bond_dim=10, max_sweeps=2)
    assert abs(res.energy - e_exact) < 1e-4


def test_apedp_robotics_package_api():
    import apedp_robotics as ar

    assert hasattr(ar, "exact_unicycle_step")
    assert hasattr(ar, "navigate_to_goal")
    assert hasattr(ar, "wrap_angle")

    # Verify step kinematics through package
    pose = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    next_pose = ar.exact_unicycle_step(pose, linear_velocity=1.0, angular_velocity=0.0, dt=1.0)
    assert abs(next_pose[0] - 1.0) < 1e-9
    assert abs(next_pose[1]) < 1e-9
    assert abs(next_pose[2]) < 1e-9


def test_apedp_molecular_package_api():
    import apedp_molecular as am

    assert hasattr(am, "energy")
    assert hasattr(am, "energy_gradient")
    assert hasattr(am, "relax")
    assert hasattr(am, "equivariance_error")

    # Verify molecular relaxation through package
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    ff = am.MolecularForceField(bonds=((0, 1),), target_length=1.0)
    res = am.relax(coords, force_field=ff, n_steps=50, step_size=0.1)
    d = float(np.linalg.norm(res.coordinates[0] - res.coordinates[1]))
    assert abs(d - 1.0) < 1e-2


def test_apedp_audio_package_api():
    import apedp_audio as aa

    assert hasattr(aa, "stft_magnitude")
    assert hasattr(aa, "ridge_instantaneous_frequency")
    assert hasattr(aa, "synthesize_chirps")

    # Verify audio synthesis through package
    spec = aa.ChirpSpec(start_freq=100.0, linear_rate=100.0)
    sig = aa.synthesize_chirps([spec], sample_rate=1000.0, duration=0.25)
    assert len(sig) == 250


def test_apedp_distributed_package_api():
    import apedp_distributed as ad

    assert hasattr(ad, "DistributedChunkedExecutor")
    assert hasattr(ad, "PartitionSpec")
    assert hasattr(ad, "square_worker")

    # Verify distributed stream through package
    executor = ad.DistributedChunkedExecutor(backend="serial")
    out = list(executor.map_stream(ad.square_worker, [1, 2, 3], chunk_size=2))
    assert out == [1, 4, 9]


def test_all_packages_have_pyproject_and_setup():
    packages_dir = REPO_ROOT / "packages"
    expected_packages = [
        "apedp-continuum",
        "jetflow-video",
        "apedp-quantum",
        "apedp-robotics",
        "apedp-molecular",
        "apedp-audio",
        "apedp-distributed",
    ]
    for pkg in expected_packages:
        p = packages_dir / pkg
        assert p.is_dir(), f"Missing package directory: {pkg}"
        assert (p / "pyproject.toml").is_file(), f"Missing pyproject.toml in {pkg}"
        assert (p / "setup.py").is_file(), f"Missing setup.py in {pkg}"
        assert (p / "README.md").is_file(), f"Missing README.md in {pkg}"


def test_domain_wrappers_declare_apedp_core_dependency():
    # The domain wrapper packages import `apedp_dyn` at module level but previously
    # declared only numpy/scipy, so a wheel installed outside the monorepo failed
    # with ModuleNotFoundError for apedp_dyn. Both metadata files must declare it.
    import tomllib

    wrappers = [
        "apedp-continuum",
        "apedp-quantum",
        "apedp-robotics",
        "apedp-molecular",
        "apedp-audio",
        "apedp-distributed",
    ]
    for pkg in wrappers:
        pkg_dir = REPO_ROOT / "packages" / pkg
        with (pkg_dir / "pyproject.toml").open("rb") as fh:
            deps = tomllib.load(fh)["project"]["dependencies"]
        assert any(dep.split(">")[0].split("=")[0].strip() == "apedp" for dep in deps), (
            f"{pkg}/pyproject.toml must declare the apedp core dependency, got {deps}"
        )
        setup_text = (pkg_dir / "setup.py").read_text(encoding="utf-8")
        assert '"apedp' in setup_text, (
            f"{pkg}/setup.py install_requires must declare the apedp core dependency"
        )


def test_jetflow_video_stays_self_contained():
    # jetflow-video is intentionally standalone: no apedp_dyn imports allowed,
    # so its wheel must never grow a hidden core dependency.
    src = REPO_ROOT / "packages" / "jetflow-video" / "jetflow_video"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import apedp" not in text, f"jetflow-video must stay standalone: {path}"
        assert "from apedp_dyn" not in text, f"jetflow-video must stay standalone: {path}"


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
