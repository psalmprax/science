#!/bin/bash
# ==============================================================================
# APEDP-X / CPMD: Multi-Node Slurm Job Array Submission Script
# ==============================================================================
# Scales candidate hypothesis sweeps across HPC supercomputing clusters.
# Binds SLURM_ARRAY_TASK_ID directly to chunked partition specifications.
#
# Usage:
#   sbatch cluster/slurm_job_array.sh
# ==============================================================================

#SBATCH --job-name=apedp_discovery_sweep
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-11%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=compute

set -euo pipefail

echo "=== APEDP-X Cluster Sweep: Task ${SLURM_ARRAY_TASK_ID} on $(hostname) ==="
echo "Started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Environment & Module Setup
module purge || true
module load python/3.12 || true
module load cuda/12.2 || true

# 2. Epistemic Storage Invariant Protection (AGENTS.md §1.2)
export APEDP_STORAGE_DIR="${APEDP_SHARED_STORAGE:-/shared/apedp_storage}"
export APEDP_TEMP_STORAGE="${SLURM_TMPDIR:-/tmp/apedp_scratch_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "${APEDP_TEMP_STORAGE}"
mkdir -p "${APEDP_STORAGE_DIR}"

# 3. Clean up node scratch space on exit
trap 'rm -rf "${APEDP_TEMP_STORAGE}"' EXIT

# 4. Execute Partition Sweep
python3 -m apedp_dyn.sandbox.real_market_l2_sweep \
    --partition-index "${SLURM_ARRAY_TASK_ID}" \
    --total-partitions 12

echo "=== APEDP-X Task ${SLURM_ARRAY_TASK_ID} Finished Successfully ==="
