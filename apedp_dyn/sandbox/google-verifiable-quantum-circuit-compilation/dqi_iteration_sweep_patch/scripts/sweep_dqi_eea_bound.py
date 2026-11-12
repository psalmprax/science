#!/usr/bin/env python3
"""Verifier-driven sweep of DQI EEA iteration caps through a repository-specific builder adapter.

This is intentionally an adapter, not a speculative edit to the official decoder. Your local
competition source must provide a builder callable accepting `n_iters` and returning a Kickmix
Circuit. The official verifier's measured cost (including R(epsilon+)) decides whether a shorter
iteration schedule is actually better.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_callable(spec: str):
    if ':' not in spec:
        raise ValueError('--builder must be module:function')
    module_name, func_name = spec.split(':', 1)
    obj = getattr(importlib.import_module(module_name), func_name)
    if not callable(obj):
        raise TypeError(f'{spec} is not callable')
    return obj


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--competition-root', type=Path, default=os.environ.get('QCOMP_ROOT'))
    p.add_argument('--builder', required=True, help='repository adapter: module:function accepting n_iters=...')
    p.add_argument('--iterations', type=int, nargs='+', default=[222, 192, 180, 168, 156])
    p.add_argument('--shots', type=int, default=102400)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, default=Path('artifacts/tier1_dqi_iteration_sweep'))
    a = p.parse_args()
    if a.competition_root is None:
        p.error('Set QCOMP_ROOT or --competition-root')
    if any(n <= 0 for n in a.iterations):
        p.error('iteration caps must be positive')
    root = a.competition_root.resolve()
    sys.path[:0] = [str(root / 'kickmix'), str(root / 'verifiers')]
    from dqi import verify_submission_dqi as dv
    builder = load_callable(a.builder)
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    spec = dv.ProblemSpecDQI.benchmark_instance(num_shots=a.shots)
    rows = []
    for n_iters in a.iterations:
        row = {'n_iters': n_iters}
        started = time.monotonic()
        try:
            circuit = builder(n_iters=n_iters)
            path = out / f'dqi_niters_{n_iters}.kmb'
            circuit.to_file(str(path), format='kmb')
            row.update({'artifact': str(path), 'artifact_sha256': sha256(path),
                        'num_qubits': int(circuit.num_qubits),
                        'toffoli_like': int(circuit.operation_statistics().num_toffoli_like)})
            report = dv.verify(circuit, spec, max_workers=a.workers)
            row['verification'] = dict(report)
            row['status'] = 'VERIFIED' if report.get('status') == 'PASS' else 'REJECTED'
            row['measured_cost'] = report.get('score')
        except Exception as exc:
            row.update({'status': 'ERROR', 'error': repr(exc)})
        row['seconds'] = time.monotonic() - started
        rows.append(row)
        (out / 'results.json').write_text(json.dumps(rows, indent=2, default=str))
        print(json.dumps({'n_iters': n_iters, 'status': row['status'],
                          'measured_cost': row.get('measured_cost'), 'seconds': row['seconds']}), flush=True)
    verified = [r for r in rows if r.get('status') == 'VERIFIED' and r.get('measured_cost') is not None]
    if verified:
        incumbent = min(verified, key=lambda r: float(r['measured_cost']))
        result = {'best_verified': incumbent['n_iters'], 'best_cost': incumbent['measured_cost'],
                  'rows': rows, 'note': 'The official failure penalty is included in each verifier-reported score.'}
    else:
        result = {'best_verified': None, 'best_cost': None, 'rows': rows}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, default=str))
    return 0 if verified else 2

if __name__ == '__main__':
    raise SystemExit(main())
