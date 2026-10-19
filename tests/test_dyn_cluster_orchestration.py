"""Tests for multi-node HPC and cloud cluster orchestration.

Verifies:
1. ClusterJobSpec validation.
2. Slurm job array generation with task array indices, memory, and CPUs.
3. Kubernetes Indexed Job manifest generation with completionMode: Indexed.
4. KubeRay Helm values generation.
5. Consistency and existence of checked-in cluster templates in cluster/.
"""
from __future__ import annotations

from pathlib import Path

from apedp_dyn.distributed.cluster_launcher import (
    ClusterJobSpec,
    generate_kuberay_helm_values,
    generate_kubernetes_job_manifest,
    generate_slurm_script,
    validate_cluster_spec,
)


def test_cluster_spec_validation():
    # Valid spec
    spec = ClusterJobSpec(total_partitions=100, max_parallel=20)
    assert validate_cluster_spec(spec) == []

    # Invalid total_partitions
    invalid_spec = ClusterJobSpec(total_partitions=0)
    assert len(validate_cluster_spec(invalid_spec)) > 0

    # max_parallel > total_partitions
    invalid_parallel = ClusterJobSpec(total_partitions=10, max_parallel=50)
    assert len(validate_cluster_spec(invalid_parallel)) > 0


def test_slurm_script_generation():
    spec = ClusterJobSpec(
        job_name="my_sweep",
        total_partitions=50,
        max_parallel=10,
        cpus_per_worker=16,
        memory_gb=64,
        walltime_hours=8,
        gpus_per_worker=2,
    )
    script = generate_slurm_script(spec)
    assert "#SBATCH --job-name=my_sweep" in script
    assert "#SBATCH --array=0-49%10" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --time=08:00:00" in script
    assert "#SBATCH --gres=gpu:2" in script
    assert "export APEDP_STORAGE_DIR=" in script
    assert "--partition-index" in script


def test_kubernetes_manifest_generation():
    spec = ClusterJobSpec(
        job_name="k8s-sweep",
        total_partitions=20,
        max_parallel=5,
        container_image="ghcr.io/science/apedp:v1.0",
    )
    manifest = generate_kubernetes_job_manifest(spec)
    assert "kind: Job" in manifest
    assert "name: k8s-sweep" in manifest
    assert "completionMode: Indexed" in manifest
    assert "completions: 20" in manifest
    assert "parallelism: 5" in manifest
    assert "image: ghcr.io/science/apedp:v1.0" in manifest
    assert "apedp-holdout-storage-pvc" in manifest


def test_kuberay_helm_values_generation():
    spec = ClusterJobSpec(
        max_parallel=30,
        cpus_per_worker=12,
        memory_gb=48,
        gpus_per_worker=1,
    )
    helm_values = generate_kuberay_helm_values(spec)
    assert "maxReplicas: 30" in helm_values
    assert 'cpu: "12"' in helm_values
    assert 'memory: "48Gi"' in helm_values
    assert 'nvidia.com/gpu: "1"' in helm_values


def test_cluster_templates_on_disk():
    repo_root = Path(__file__).resolve().parent.parent
    cluster_dir = repo_root / "cluster"
    assert cluster_dir.is_dir()

    slurm_script = cluster_dir / "slurm_job_array.sh"
    k8s_job = cluster_dir / "k8s_batch_sweep_job.yaml"
    ray_helm = cluster_dir / "ray_cluster_helm_values.yaml"

    assert slurm_script.is_file()
    assert k8s_job.is_file()
    assert ray_helm.is_file()

    # Verify key tokens
    assert "--array=0-99%20" in slurm_script.read_text(encoding="utf-8")
    assert "completionMode: Indexed" in k8s_job.read_text(encoding="utf-8")
    assert "maxReplicas: 50" in ray_helm.read_text(encoding="utf-8")
