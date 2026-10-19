"""Multi-Node HPC & Cloud Cluster Orchestration for APEDP-X / CPMD.

Generates production deployment manifests and job execution scripts for:
1. Slurm Job Arrays (HPC scientific supercomputing clusters).
2. Kubernetes Indexed Batch Jobs (cloud-native autoscaling sweeps).
3. KubeRay / Ray Cluster Helm values (multi-node distributed actor sweeps).

Guarantees anti-snooping storage discipline (AGENTS.md §1.2) by configuring
shared persistent volumes for holdout ledgers and node-local fast scratch storage.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path


class ClusterScheduler(str, enum.Enum):
    """Supported multi-node cluster schedulers."""
    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    RAY = "ray"


@dataclass
class ClusterJobSpec:
    """Specification for distributed multi-node hypothesis sweeps."""
    job_name: str = "apedp-discovery-sweep"
    total_partitions: int = 12
    max_parallel: int = 8
    cpus_per_worker: int = 8
    memory_gb: int = 32
    gpus_per_worker: int = 0
    walltime_hours: int = 4
    shared_storage_dir: str = "/mnt/apedp_storage"
    local_scratch_dir: str = "/tmp/scratch"
    container_image: str = "ghcr.io/owner/science:latest"
    slurm_partition: str = "compute"


def validate_cluster_spec(spec: ClusterJobSpec) -> list[str]:
    """Validate cluster job parameters."""
    errors = []
    if spec.total_partitions <= 0:
        errors.append("total_partitions must be > 0.")
    if spec.max_parallel <= 0 or spec.max_parallel > spec.total_partitions:
        errors.append("max_parallel must be between 1 and total_partitions.")
    if spec.cpus_per_worker <= 0:
        errors.append("cpus_per_worker must be > 0.")
    if spec.memory_gb <= 0:
        errors.append("memory_gb must be > 0.")
    return errors


def _validate_partitioned_sweep_spec(spec: ClusterJobSpec) -> list[str]:
    errors = validate_cluster_spec(spec)
    if spec.total_partitions > 12:
        errors.append("total_partitions must be <= 12 for the real-market L2 sweep.")
    return errors


def generate_slurm_script(spec: ClusterJobSpec) -> str:
    """Generate a production Slurm job array submission script."""
    errors = _validate_partitioned_sweep_spec(spec)
    if errors:
        raise ValueError(f"Invalid ClusterJobSpec: {'; '.join(errors)}")

    gpu_directive = f"#SBATCH --gres=gpu:{spec.gpus_per_worker}\n" if spec.gpus_per_worker > 0 else ""

    return f"""#!/bin/bash
# ==============================================================================
# Slurm Job Array: {spec.job_name}
# ==============================================================================
#SBATCH --job-name={spec.job_name}
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-{spec.total_partitions - 1}%{spec.max_parallel}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={spec.cpus_per_worker}
#SBATCH --mem={spec.memory_gb}G
#SBATCH --time={spec.walltime_hours:02d}:00:00
#SBATCH --partition={spec.slurm_partition}
{gpu_directive}set -euo pipefail

export APEDP_STORAGE_DIR="{spec.shared_storage_dir}"
export APEDP_TEMP_STORAGE="{spec.local_scratch_dir}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "${{APEDP_TEMP_STORAGE}}"
mkdir -p "${{APEDP_STORAGE_DIR}}"

trap 'rm -rf "${{APEDP_TEMP_STORAGE}}"' EXIT

python3 -m apedp_dyn.sandbox.real_market_l2_sweep \\
--partition-index "${{SLURM_ARRAY_TASK_ID}}" \\
--total-partitions {spec.total_partitions}
"""


def generate_kubernetes_job_manifest(spec: ClusterJobSpec) -> str:
    """Generate a Kubernetes Indexed batch Job manifest."""
    errors = _validate_partitioned_sweep_spec(spec)
    if errors:
        raise ValueError(f"Invalid ClusterJobSpec: {'; '.join(errors)}")

    gpu_request = f'              nvidia.com/gpu: "{spec.gpus_per_worker}"\n' if spec.gpus_per_worker > 0 else ""

    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {spec.job_name}
  labels:
    app.kubernetes.io/name: apedp-x
    app.kubernetes.io/component: distributed-sweep
spec:
  completionMode: Indexed
  completions: {spec.total_partitions}
  parallelism: {spec.max_parallel}
  backoffLimit: 3
  template:
    metadata:
      labels:
        app: apedp-worker
    spec:
      restartPolicy: OnFailure
      containers:
        - name: sweep-worker
          image: {spec.container_image}
          imagePullPolicy: IfNotPresent
          command: ["python3", "-m", "apedp_dyn.sandbox.real_market_l2_sweep"]
          args:
            - "--partition-index"
            - "$(JOB_COMPLETION_INDEX)"
            - "--total-partitions"
            - "{spec.total_partitions}"
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
            - name: APEDP_STORAGE_DIR
              value: "{spec.shared_storage_dir}"
            - name: APEDP_TEMP_STORAGE
              value: "{spec.local_scratch_dir}"
          resources:
            requests:
              cpu: "{max(1, spec.cpus_per_worker // 2)}"
              memory: "{spec.memory_gb // 2}Gi"
{gpu_request}            limits:
              cpu: "{spec.cpus_per_worker}"
              memory: "{spec.memory_gb}Gi"
{gpu_request}          volumeMounts:
            - name: shared-storage
              mountPath: {spec.shared_storage_dir}
            - name: local-scratch
              mountPath: {spec.local_scratch_dir}
      volumes:
        - name: shared-storage
          persistentVolumeClaim:
            claimName: apedp-holdout-storage-pvc
        - name: local-scratch
          emptyDir: {{}}
"""


def generate_kuberay_helm_values(spec: ClusterJobSpec) -> str:
    """Generate KubeRay Helm Chart configuration values."""
    errors = validate_cluster_spec(spec)
    if errors:
        raise ValueError(f"Invalid ClusterJobSpec: {'; '.join(errors)}")

    gpu_req = f'      nvidia.com/gpu: "{spec.gpus_per_worker}"\n' if spec.gpus_per_worker > 0 else ""

    return f"""image:
  repository: {spec.container_image.split(':')[0]}
  tag: {spec.container_image.split(':')[-1] if ':' in spec.container_image else 'latest'}
  pullPolicy: IfNotPresent

head:
  rayStartParams:
    dashboard-host: "0.0.0.0"
    num-cpus: "{spec.cpus_per_worker}"
  resources:
    limits:
      cpu: "{spec.cpus_per_worker}"
      memory: "{spec.memory_gb}Gi"
    requests:
      cpu: "2"
      memory: "8Gi"
  volumes:
    - name: shared-storage
      persistentVolumeClaim:
        claimName: apedp-holdout-storage-pvc
  volumeMounts:
    - mountPath: {spec.shared_storage_dir}
      name: shared-storage

worker:
  replicas: {min(10, spec.max_parallel)}
  minReplicas: 2
  maxReplicas: {spec.max_parallel}
  resources:
    limits:
      cpu: "{spec.cpus_per_worker}"
      memory: "{spec.memory_gb}Gi"
{gpu_req}    requests:
      cpu: "{max(1, spec.cpus_per_worker // 2)}"
      memory: "{spec.memory_gb // 2}Gi"
{gpu_req}  volumes:
    - name: shared-storage
      persistentVolumeClaim:
        claimName: apedp-holdout-storage-pvc
    - name: scratch-storage
      emptyDir: {{}}
  volumeMounts:
    - mountPath: {spec.shared_storage_dir}
      name: shared-storage
    - mountPath: {spec.local_scratch_dir}
      name: scratch-storage
  env:
    - name: APEDP_STORAGE_DIR
      value: "{spec.shared_storage_dir}"
    - name: APEDP_TEMP_STORAGE
      value: "{spec.local_scratch_dir}"
"""
