#!/usr/bin/env python3
"""Batched Full-Domain & Range Verification Harness for 62Q/89Q InvSqrt Circuits.

Verifies basis states across arbitrary domain ranges up to the full benchmark domain:
    x in [1, max_x], where max_x = 3 * (2^10 - 2)^2 = 3,133,452.
    Official benchmark requires 64 shots per x (200,540,928 shots total).

Key Features:
  - Exact simulation using km.Simulator (Python 3.13)
  - Strict validation: basis-state input restoration on register 'x' and scratch == 0
  - Phase tracking in 64-bit fixed point (units of 2^-64 turns) vs tol(x) = 2^-24 * beta / sqrt(x)
  - Exact official referee scoring: S = Q * (T_toffoli + T_gates / 4)
  - Atomic JSON checkpointing and resumption (--resume)
  - Multi-process parallel chunking with progress telemetry and throughput reporting
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from fractions import Fraction

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/kickmix"))
sys.path.insert(0, str(REPO_ROOT / "apedp_dyn/sandbox/google-verifiable-quantum-circuit-compilation/verifiers/invsqrt"))

import kickmix as km
from verify_submission_invsqrt import (
    ProblemSpecInvSqrt,
    _circuit_sha256,
    _initial_seed,
    validate_registers,
    reference_values,
    half_turns_to_units,
    rotation_t_count,
    to_bits,
    from_bits,
    T_PER_TOFFOLI,
    DEFAULT_BATCH_SIZE,
)

DEFAULT_CIRCUIT_PATH = REPO_ROOT / "artifacts/quantum_compilation/invsqrt/invsqrt_62q_archived.kmb"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "artifacts/quantum_compilation/invsqrt/invsqrt_62q_domain_checkpoint.json"


def run_chunk_worker(
    task: tuple[int, int, int, str, int, int, Fraction, int]
) -> tuple[int, int, float, int, int, int, int]:
    """Worker task executed in parallel process.

    Task tuple:
        (chunk_start, chunk_stop, sim_seed, circuit_path, batch_size, width, beta, precision_bits)
    Returns:
        (shots_count, min_margin, worst_ratio, worst_x, toffoli_sum, t_sum, rot_sum)
    """
    chunk_start, chunk_stop, sim_seed, circuit_path, batch_size, width, beta, precision_bits = task

    circuit = km.Circuit.from_file(circuit_path)
    spec = ProblemSpecInvSqrt(
        width=width,
        max_x=3133452,
        beta=beta,
        precision_bits=precision_bits,
        shots_per_x=1,
        system_qubits=3 * 218 * 10,
    )
    x_reg, scratch = validate_registers(circuit, spec)

    sim = km.Simulator(
        batch_size=batch_size,
        debug_print=False,
        counting_operations=True,
        seed=sim_seed,
    )

    min_margin = 1 << 64
    worst_ratio = -1.0
    worst_x = 0

    chunk_shots = chunk_stop - chunk_start

    for start in range(chunk_start, chunk_stop, batch_size):
        bs = min(batch_size, chunk_stop - start)
        # Input x values for this batch (1-indexed)
        # When evaluating an x-range directly:
        xs = np.arange(start + 1, start + bs + 1, dtype=np.uint64)
        target, tol = reference_values(spec, xs)

        sim.clear_for_shot()
        sim.write_across_shots(x_reg, to_bits(xs, spec.width))
        sim.do(circuit)

        # 1. Check register x restoration
        read_x = from_bits(sim.read_across_shots(x_reg))
        if not np.array_equal(read_x[:bs], xs):
            raise AssertionError(f"Register 'x' was not restored for inputs {xs[read_x[:bs] != xs]}.")

        # 2. Check scratch qubits return to |0>
        if len(scratch):
            dirty = sim.read_across_shots(scratch).any(axis=1)
            if dirty.any():
                idx = int(np.argmax(dirty))
                raise AssertionError(f"Scratch qubit {scratch[idx]} did not return to |0>.")

        # 3. Tracked phase validation
        theta = np.array([half_turns_to_units(sim.read_shot_phase(i)) for i in range(bs)], dtype=np.uint64)
        allowed = tol
        error = (theta - target).view(np.int64)  # in [-1/2, 1/2) turns
        used = np.abs(error).view(np.uint64)

        bad = np.flatnonzero(used > allowed)
        if len(bad):
            bad_idx = bad[0]
            raise AssertionError(
                f"x={xs[bad_idx]}: phase error {error[bad_idx] / 2**64:.3e} turns "
                f"exceeds tolerance {allowed[bad_idx] / 2**64:.3e}."
            )

        ratio = np.divide(used, allowed, out=np.zeros(bs, dtype=np.float64), where=allowed > 0)
        curr_min_margin = int((allowed - used).min())
        if curr_min_margin < min_margin:
            min_margin = curr_min_margin

        curr_max_ratio = float(ratio.max())
        if curr_max_ratio > worst_ratio:
            worst_ratio = curr_max_ratio
            worst_x = int(xs[np.argmax(ratio)])

    stats = sim.operation_statistics()
    return (
        chunk_shots,
        min_margin,
        worst_ratio,
        worst_x,
        int(stats.num_toffoli_like),
        int(stats.num_t_like),
        int(stats.num_z_pow(min_clifford_hierarchy_level=4)),
    )


def compute_official_score(
    num_qubits: int,
    total_shots: int,
    total_toffoli: int,
    total_t_base: int,
    total_rot: int,
    min_margin: int,
) -> dict:
    toffoli_count = total_toffoli / total_shots if total_shots else 0.0
    t_base = total_t_base / total_shots if total_shots else 0.0
    n_rot = total_rot / total_shots if total_shots else 0.0

    eps = math.pi * min_margin / 2**64 / n_rot if n_rot else None
    if n_rot and (eps is None or eps <= 0):
        raise AssertionError("No tolerance left for rotation synthesis error (min_margin <= 0).")

    t_count = t_base + (n_rot * rotation_t_count(eps) if n_rot else 0)
    toffoli_equiv = toffoli_count + t_count / T_PER_TOFFOLI
    score = num_qubits * toffoli_equiv

    return {
        "score": score,
        "toffoli_equivalent": toffoli_equiv,
        "toffoli_count": toffoli_count,
        "t_count": t_base,
        "rotation_count": n_rot,
        "rotation_eps": eps,
        "num_qubits": num_qubits,
    }


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--circuit", type=str, default=str(DEFAULT_CIRCUIT_PATH), help="Path to .kmb circuit file")
    parser.add_argument("--start-x", type=int, default=1, help="Starting x index (inclusive, 1-indexed)")
    parser.add_argument("--max-x", type=int, default=3133452, help="Ending x index (inclusive, default 3,133,452)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for km.Simulator")
    parser.add_argument("--chunk-batches", type=int, default=32, help="Batches per worker chunk")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    parser.add_argument("--checkpoint-file", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Path for checkpoint JSON")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint file")
    parser.add_argument("--test-batch", action="store_true", help="Quick run on x in [1, 2048]")
    parser.add_argument("--salt", type=str, default="", help="Salt for simulation RNG seeding")
    args = parser.parse_args()

    circuit_path = Path(args.circuit).resolve()
    if not circuit_path.exists():
        raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

    circuit = km.Circuit.from_file(str(circuit_path))
    circuit_hash = _circuit_sha256(circuit)
    num_qubits = circuit.num_qubits
    static_toffolis = int(circuit.operation_statistics().num_toffoli_like)

    print(f"Loaded circuit: {circuit_path.name}")
    print(f"  Qubits: {num_qubits}")
    print(f"  Static Toffoli-like ops: {static_toffolis}")
    print(f"  SHA-256: {circuit_hash}")

    start_x = args.start_x
    max_x = 2048 if args.test_batch else args.max_x

    if start_x < 1 or max_x < start_x:
        raise ValueError(f"Invalid x range: [{start_x}, {max_x}]")

    total_inputs = max_x - start_x + 1
    batch_size = args.batch_size
    chunk_size = args.chunk_batches * batch_size

    checkpoint_file = Path(args.checkpoint_file).resolve()
    state = {
        "circuit_sha256": circuit_hash,
        "circuit_path": str(circuit_path),
        "start_x": start_x,
        "max_x": max_x,
        "completed_inputs": 0,
        "min_margin": 1 << 64,
        "worst_error_over_tolerance": 0.0,
        "worst_x": 0,
        "total_toffoli": 0,
        "total_t_base": 0,
        "total_rot": 0,
        "elapsed_seconds": 0.0,
        "status": "IN_PROGRESS",
    }

    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            saved = json.load(f)
        if saved.get("circuit_sha256") == circuit_hash and saved.get("max_x") == max_x:
            state = saved
            print(f"Resuming from checkpoint at input x={start_x + state['completed_inputs']} ({state['completed_inputs']}/{total_inputs} completed).")
        else:
            print("Checkpoint metadata does not match current run. Starting fresh.")

    remaining_start_idx = start_x - 1 + state["completed_inputs"]
    remaining_total = max_x - remaining_start_idx
    if remaining_total <= 0:
        print("Range already completely verified according to checkpoint!")
        metrics = compute_official_score(
            num_qubits,
            state["completed_inputs"],
            state["total_toffoli"],
            state["total_t_base"],
            state["total_rot"],
            state["min_margin"],
        )
        print(json.dumps(metrics, indent=2))
        return

    # Build chunk tasks
    base_seed = _initial_seed(circuit_hash, args.salt)
    chunk_starts = list(range(remaining_start_idx, max_x, chunk_size))
    seeds = np.random.SeedSequence(base_seed, pool_size=8).generate_state(4 * len(chunk_starts), dtype=np.uint64).reshape(-1, 4)

    tasks = [
        (
            s,
            min(s + chunk_size, max_x),
            int.from_bytes(seeds[i], "little"),
            str(circuit_path),
            batch_size,
            22,
            Fraction(1),
            24,
        )
        for i, s in enumerate(chunk_starts)
    ]

    cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = min(len(tasks), cpus if args.workers is None else args.workers)
    print(f"Starting verification across x in [{remaining_start_idx + 1}, {max_x}] ({remaining_total:,} inputs)")
    print(f"  Workers: {workers}, Batch size: {batch_size}, Chunk size: {chunk_size}")

    t0 = time.perf_counter()
    pool_context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=pool_context) as executor:
        futures = {executor.submit(run_chunk_worker, t): t for t in tasks}

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            shots_count, c_min_margin, c_worst_ratio, c_worst_x, c_toff, c_t, c_rot = res

            state["completed_inputs"] += shots_count
            state["min_margin"] = min(state["min_margin"], c_min_margin)
            if c_worst_ratio > state["worst_error_over_tolerance"]:
                state["worst_error_over_tolerance"] = c_worst_ratio
                state["worst_x"] = c_worst_x
            state["total_toffoli"] += c_toff
            state["total_t_base"] += c_t
            state["total_rot"] += c_rot
            state["elapsed_seconds"] += time.perf_counter() - t0
            t0 = time.perf_counter()

            # Telemetry
            pct = 100.0 * state["completed_inputs"] / total_inputs
            rate = state["completed_inputs"] / max(0.001, state["elapsed_seconds"])
            print(
                f"Progress: {state['completed_inputs']:,}/{total_inputs:,} ({pct:5.2f}%) | "
                f"Rate: {rate:8.0f} inputs/sec | "
                f"Worst ratio: {state['worst_error_over_tolerance']:.6f} at x={state['worst_x']} | "
                f"Min margin: {state['min_margin']}"
            )
            save_checkpoint(checkpoint_file, state)

    state["status"] = "COMPLETED"
    save_checkpoint(checkpoint_file, state)

    metrics = compute_official_score(
        num_qubits,
        state["completed_inputs"],
        state["total_toffoli"],
        state["total_t_base"],
        state["total_rot"],
        state["min_margin"],
    )

    final_receipt = {
        "status": "PASS",
        "verified_inputs": state["completed_inputs"],
        "max_x": max_x,
        "worst_error_over_tolerance": state["worst_error_over_tolerance"],
        "worst_x": state["worst_x"],
        "min_margin_units": state["min_margin"],
        "circuit_sha256": circuit_hash,
        "elapsed_seconds": round(state["elapsed_seconds"], 2),
        **metrics,
    }

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(json.dumps(final_receipt, indent=2))


if __name__ == "__main__":
    main()
