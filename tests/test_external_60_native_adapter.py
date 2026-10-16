"""Smoke pins for the RIME-AD native external-corpus adapter.

Scope: plumbing only. The adapter must satisfy the frozen round's contract
(public data only, predictions on all four public grids, abstention never
carries numbers) without consulting hidden truth. Hidden-truth scoring of
this engine requires its own preregistered round -- see the adapter docstring
in `benchmarks/external_60_adapters.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.external_60_adapters import RimeADNativeAdapter, load_public_grids

CORPUS = ROOT / "benchmark_corpus_external_60"


def _small_adapter() -> RimeADNativeAdapter:
    return RimeADNativeAdapter(population_size=60, generations=12, islands=2, seed=42)


def test_native_adapter_contract_on_p01():
    adapter = _small_adapter()
    res = adapter.run(CORPUS, "P01")
    assert res.competitor == "rime_ad_native"
    assert res.problem_id == "P01"
    assert res.expression, res
    assert res.metadata["n_evaluations"] > 0
    if res.abstained:
        assert not res.predictions, "abstention must not ship numeric predictions"
    else:
        grids = load_public_grids(CORPUS, "P01")
        for key in ("train.csv", "validation.csv", "near_ood.csv", "far_ood.csv"):
            preds = np.asarray(res.predictions[key], dtype=float)
            assert preds.shape[0] == grids[key].shape[0], key
    assert res.wall_seconds > 0.0


def test_native_adapter_works_without_hidden_dir(tmp_path):
    # Behavioral proof of public-only access: a corpus copy with no hidden/
    # directory at all must still run the adapter to completion.
    import os

    (tmp_path / "public").mkdir()
    os.symlink(CORPUS / "public" / "P01", tmp_path / "public" / "P01")
    res = _small_adapter().run(tmp_path, "P01")
    assert res.expression, res
    assert res.competitor == "rime_ad_native"
