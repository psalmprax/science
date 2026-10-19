"""Tests verifying standalone pip packaging and public API exposure across all domain subpackages.

Verifies:
1. apedp-continuum (apedp_continuum)
2. jetflow-video (jetflow_video)
3. apedp-quantum (apedp_quantum)
4. apedp-robotics (apedp_robotics)
5. apedp-molecular (apedp_molecular)
6. apedp-audio (apedp_audio)
7. apedp-distributed (apedp_distributed)
"""
from __future__ import annotations

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
