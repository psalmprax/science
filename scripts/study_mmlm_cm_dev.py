"""Development study of MMLM-CM on WikiText-2: fitted on train, tuned on half the validation text, measured on the other half.

    python scripts/study_mmlm_cm_dev.py

Never reads the test split. Writes ``benchmarks/mmlm_cm_dev_study.json``.
"""
from __future__ import annotations

import bz2
import gzip
import json
import lzma
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_mmlm import neural_expert  # noqa: E402
from apedp_mmlm.context_mixing import ALL_EXPERTS, MATCH_EXPERTS, RECENT_EXPERTS, ContextMixingLM  # noqa: E402
from apedp_mmlm.text_corpus import split_bytes  # noqa: E402

OUT = ROOT / "benchmarks" / "mmlm_cm_dev_study.json"
WEIGHTS = ROOT / "benchmarks" / "mmlm" / "neural_expert_wikitext2.pt"


def compressors() -> dict[str, Callable[[bytes], bytes]]:
    """Reference compressors. Optional ones are skipped if the package is missing."""
    out: dict[str, Callable[[bytes], bytes]] = {
        "gzip -9": lambda d: gzip.compress(d, 9),
        "bz2 -9": lambda d: bz2.compress(d, 9),
        "lzma -9e, 64 MB dictionary": lambda d: lzma.compress(d, format=lzma.FORMAT_RAW, filters=[
            {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": 1 << 26}]),
    }
    try:
        import pyppmd
        for order in (6, 12, 16):
            out[f"PPMd (var. I) order {order}"] = (lambda o: lambda d: pyppmd.compress(d, max_order=o, mem_size=1 << 30))(order)
    except ImportError:
        pass
    try:
        import zstandard
        out["zstd -22 --long=27"] = lambda d: zstandard.ZstdCompressor(compression_params=zstandard.ZstdCompressionParameters.from_level(
            22, window_log=27, enable_ldm=True)).compress(d)
    except ImportError:
        pass
    return out


def compressor_bits(train: bytes, data: bytes) -> dict[str, dict[str, float]]:
    """Bits per byte alone, and primed: the extra bytes needed for ``data`` after the training text."""
    out = {}
    for name, fn in compressors().items():
        alone, base, both = len(fn(data)), len(fn(train)), len(fn(train + data))
        out[name] = {"alone": 8 * alone / len(data), "primed_with_training_text": 8 * (both - base) / len(data)}
    return out


def main() -> int:
    train, fit_text, dev_text = split_bytes("train"), split_bytes("valid_a"), split_bytes("valid_b")
    started = time.time()
    model = ContextMixingLM().fit_counts(train).calibrate(fit_text)
    network = neural_expert.load(WEIGHTS)
    fit = model.features(fit_text, neural_expert.bit_probabilities(network, fit_text))
    dev = model.features(dev_text, neural_expert.bit_probabilities(network, dev_text))
    results: dict[str, Any] = {"fit_seconds_before_mixer": round(time.time() - started, 1)}
    arms = {
        "MMLM-CM (all experts)": ALL_EXPERTS,
        "without the neural expert": tuple(e for e in ALL_EXPERTS if e != "nn"),
        "without read-so-far counts": tuple(e for e in ALL_EXPERTS if e not in RECENT_EXPERTS),
        "without the match experts": tuple(e for e in ALL_EXPERTS if e not in MATCH_EXPERTS),
        "without the word experts": tuple(e for e in ALL_EXPERTS if e not in ("w1", "w2", "dw1", "dw2")),
        "training-text counts only": tuple(e for e in ALL_EXPERTS if e[0] in "ow"),
    }
    for name, experts in arms.items():
        t = time.time()
        model.fit_mixer(fit, experts)
        results[name] = {"bits_per_byte": float(model.byte_bits(dev).mean()), "bits_per_byte_on_fit_half": float(model.byte_bits(fit).mean()),
                         "mixer_seconds": round(time.time() - t, 1)}
        print(name, results[name], flush=True)
    results["experts alone (calibrated)"] = {name: ContextMixingLM.expert_alone(dev, name) for name in ALL_EXPERTS}
    results["compressors"] = compressor_bits(train, dev_text)
    results["neural_expert"] = json.loads(WEIGHTS.with_suffix(".json").read_text())
    results["corpus"] = "WikiText-2 raw: counts and network on train; calibration and mixer on the first half of valid; measured on the second half"
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k in ("experts alone (calibrated)", "compressors")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
