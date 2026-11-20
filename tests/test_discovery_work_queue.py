"""The discovery work queue (apedp_dyn/discovery_l2/work_queue.py).

The central test runs the same discovery jobs on one worker and on three
worker processes and requires identical candidates, verdicts and trial counts.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from apedp_dyn.discovery_l2 import work_queue as wq
from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_referee.trial_ledger import CandidateTrialLedger

ROOT = Path(__file__).resolve().parents[1]


def _jobs(tmp_path: Path) -> list[dict]:
    """Six small problems: three laws, two seeds each, one with no law in reach."""
    laws = [
        ("velocity", {"dist": Dimension(l=1), "dur": Dimension(t=1)}, Dimension(l=1, t=-1), lambda d: d["dist"] / d["dur"]),
        ("area", {"side": Dimension(l=1), "mass": Dimension(m=1)}, Dimension(l=2), lambda d: 0.5 * d["side"] ** 2),
        ("noise", {"u": Dimension(l=1), "v": Dimension(t=1)}, Dimension(m=1), None),
    ]
    jobs = []
    for name, dims, target, fn in laws:
        for seed in (0, 1):
            rng = np.random.default_rng(seed + 10 * len(jobs))
            data = {k: rng.uniform(1, 10, 80) for k in dims}
            y = fn(data) * (1 + 0.01 * rng.standard_normal(80)) if fn else rng.uniform(1, 10, 80)
            path = tmp_path / f"{name}-{seed}.npz"
            np.savez(path, y=y, **data)
            jobs.append({"job_id": f"{name}-{seed}", "data": str(path), "seed": seed,
                         "dimensions": {k: wq.dimension_to_json(v) for k, v in dims.items()},
                         "target_dimension": wq.dimension_to_json(target),
                         "noise_sigma": 0.01 * float(np.mean(np.abs(y)))})
    return jobs


def _without_timestamps(ledger: CandidateTrialLedger) -> list[dict]:
    out = []
    for record in ledger.get_records():
        row = dataclasses.asdict(record)
        row.pop("timestamp")
        row["selection_statistic"] = repr(row["selection_statistic"])  # nan != nan
        out.append(row)
    return out


def test_one_worker_and_three_processes_reach_identical_verdicts(tmp_path):
    jobs = _jobs(tmp_path)
    serial = wq.WorkQueue.create(tmp_path / "q1", jobs, chunk_size=1)
    assert wq.run_worker(serial.root, wq.discover_job, "solo") == len(jobs)
    serial_ledger = CandidateTrialLedger()
    serial_results = wq.collect(serial.root, serial_ledger)

    parallel = wq.WorkQueue.create(tmp_path / "q3", jobs, chunk_size=1)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    procs = [subprocess.Popen([sys.executable, "-m", "apedp_dyn.discovery_l2.work_queue", "worker",
                               "--root", str(parallel.root), "--worker-id", f"w{k}"],
                              cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k in range(3)]
    outputs = [p.communicate(timeout=600) for p in procs]
    assert all(p.returncode == 0 for p in procs), outputs
    assert sum(int(out.split()[0]) for out, _ in outputs) == len(jobs)
    parallel_ledger = CandidateTrialLedger()
    parallel_results = wq.collect(parallel.root, parallel_ledger)

    assert parallel_results == serial_results
    assert _without_timestamps(parallel_ledger) == _without_timestamps(serial_ledger)
    assert parallel_ledger.trial_count == serial_ledger.trial_count == len(jobs)
    assert parallel_ledger.effective_trial_count == serial_ledger.effective_trial_count
    # The jobs are real: the two laws are found and the pure-noise target is not.
    by_id = {r["job_id"]: r for r in serial_results}
    assert by_id["velocity-0"]["promoted"] and by_id["area-0"]["promoted"]
    assert by_id["noise-0"]["promoted"] is None


def test_a_claimed_chunk_cannot_be_claimed_again(tmp_path):
    queue = wq.WorkQueue.create(tmp_path / "q", [{"job_id": str(k)} for k in range(3)], chunk_size=2)
    first, second = queue.claim("a"), queue.claim("b")
    assert first is not None and second is not None and first.name.split(".json")[0] != second.name.split(".json")[0]
    assert queue.claim("c") is None


def test_a_dead_workers_chunk_is_requeued_and_finished_once(tmp_path):
    queue = wq.WorkQueue.create(tmp_path / "q", [{"job_id": "x"}], chunk_size=1)
    claim = queue.claim("dead")
    assert claim is not None and queue.requeue_stale(older_than_seconds=3600) == []
    os.utime(claim, (0, 0))
    assert queue.requeue_stale(older_than_seconds=3600) == ["chunk-00000.json"]
    assert wq.run_worker(queue.root, lambda job: {"job_id": job["job_id"], "ok": True}, "alive") == 1
    assert queue.done() and wq.collect(queue.root) == [{"job_id": "x", "ok": True}]


def test_collect_refuses_an_unfinished_queue(tmp_path):
    queue = wq.WorkQueue.create(tmp_path / "q", [{"job_id": "x"}, {"job_id": "y"}], chunk_size=1)
    claim = queue.claim("w")
    assert claim is not None
    queue.complete(claim, [{"job_id": "x"}])
    with pytest.raises(RuntimeError, match="not finished"):
        wq.collect(queue.root)


def test_results_come_back_in_job_order_whatever_order_chunks_finish(tmp_path):
    queue = wq.WorkQueue.create(tmp_path / "q", [{"job_id": k} for k in "abcd"], chunk_size=1)
    claims = [queue.claim("w") for _ in range(4)]
    for claim in reversed(claims):
        assert claim is not None
        job = json.loads(claim.read_text())[0]
        queue.complete(claim, [{"job_id": job["job_id"]}])
    assert [r["job_id"] for r in wq.collect(queue.root)] == list("abcd")


def test_a_queue_is_created_once_with_unique_ids(tmp_path):
    wq.WorkQueue.create(tmp_path / "q", [{"job_id": "x"}])
    with pytest.raises(FileExistsError):
        wq.WorkQueue.create(tmp_path / "q", [{"job_id": "y"}])
    with pytest.raises(ValueError, match="unique"):
        wq.WorkQueue.create(tmp_path / "q2", [{"job_id": "x"}, {"job_id": "x"}])


def test_dimensions_survive_the_round_trip():
    dim = Dimension(m=1, l=Fraction(-3, 2), t=-2, th=1)
    assert wq.dimension_from_json(json.loads(json.dumps(wq.dimension_to_json(dim)))) == dim
    assert wq.dimension_to_json(Dimension()) == {}
