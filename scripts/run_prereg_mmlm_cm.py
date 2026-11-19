"""PREREG-MMLM-CM-2026-11: the generic model (bytes) on the WikiText-2 test split, once.

    python scripts/run_prereg_mmlm_cm.py --out preregistration/RESULT-MMLM-CM-2026-11

Refuses to run if a frozen file changed, or if the output directory exists.
Protocol and expectations: ``preregistration/PREREG-MMLM-CM-2026-11.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import study_mmlm_cm_dev as dev  # noqa: E402

from apedp_mmlm import neural_expert  # noqa: E402
from apedp_mmlm.context_mixing import ALL_EXPERTS, MATCH_EXPERTS, RECENT_EXPERTS, ContextMixingLM  # noqa: E402
from apedp_mmlm.text_corpus import split_bytes  # noqa: E402

PREREGISTRATION_ID = "PREREG-MMLM-CM-2026-11"
DIGEST = ROOT / "preregistration" / "DIGEST-MMLM-CM.json"
WEIGHTS = ROOT / "benchmarks" / "mmlm" / "neural_expert_wikitext2.pt"
FROZEN = ("preregistration/PREREG-MMLM-CM-2026-11.md", "scripts/run_prereg_mmlm_cm.py", "scripts/study_mmlm_cm_dev.py",
          "apedp_mmlm/context_mixing.py", "apedp_mmlm/neural_expert.py", "apedp_mmlm/text_corpus.py",
          "benchmarks/mmlm/neural_expert_wikitext2.pt", "benchmarks/mmlm/neural_expert_wikitext2.json")
ARMS: dict[str, tuple[str, ...]] = {
    "MMLM-CM (all experts)": ALL_EXPERTS,
    "without the neural expert": tuple(e for e in ALL_EXPERTS if e != "nn"),
    "without read-so-far counts": tuple(e for e in ALL_EXPERTS if e not in RECENT_EXPERTS),
    "without the match experts": tuple(e for e in ALL_EXPERTS if e not in MATCH_EXPERTS),
    "without the word experts": tuple(e for e in ALL_EXPERTS if e not in ("w1", "w2", "dw1", "dw2")),
    "training-text counts only": tuple(e for e in ALL_EXPERTS if e[0] in "ow"),
}


def frozen_digests() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FROZEN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if json.loads(DIGEST.read_text(encoding="utf-8"))["sha256"] != frozen_digests():
        raise SystemExit(f"{PREREGISTRATION_ID} changed since it was frozen")
    args.out.mkdir(parents=True, exist_ok=False)
    train, tune, test = split_bytes("train"), split_bytes("valid"), split_bytes("test")
    started = time.time()
    model = ContextMixingLM().fit_counts(train).calibrate(tune)
    network = neural_expert.load(WEIGHTS)
    fit = model.features(tune, neural_expert.bit_probabilities(network, tune))
    held_out = model.features(test, neural_expert.bit_probabilities(network, test))
    model.release_counts()
    results: dict[str, Any] = {"preregistration": PREREGISTRATION_ID, "frozen": frozen_digests(),
                               "bytes": {"train": len(train), "tuning": len(tune), "test": len(test)},
                               "seconds_before_mixer": round(time.time() - started, 1), "arms": {}}
    for name, experts in ARMS.items():
        t = time.time()
        model.fit_mixer(fit, experts)
        results["arms"][name] = {"bits_per_byte": float(model.byte_bits(held_out).mean()),
                                 "bits_per_byte_on_tuning_text": float(model.byte_bits(fit).mean()),
                                 "mixer_seconds": round(time.time() - t, 1)}
        print(name, results["arms"][name], flush=True)
    results["experts_alone"] = {name: ContextMixingLM.expert_alone(held_out, name) for name in ALL_EXPERTS}
    results["compressors"] = dev.compressor_bits(train, test)
    results["neural_expert"] = json.loads(WEIGHTS.with_suffix(".json").read_text(encoding="utf-8"))
    (args.out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
