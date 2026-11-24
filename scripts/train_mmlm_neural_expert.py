"""Train MMLM-CM's one neural expert on the WikiText-2 training text, on CPU.

    python scripts/train_mmlm_neural_expert.py --seconds 2400 --out data/mmlm/neural_expert.pt

Reads only the training split. Writes the weights and a JSON beside them with the
training record (steps, bytes seen, seconds, sizes, seed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_mmlm import neural_expert  # noqa: E402
from apedp_mmlm.text_corpus import split_bytes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=2400.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "mmlm" / "neural_expert.pt")
    args = parser.parse_args()
    train = split_bytes("train")
    _model, info = neural_expert.train(train, seconds=args.seconds, hidden=args.hidden, log=lambda m: print(m, flush=True),
                                      checkpoint=args.out)
    info["weights_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    info["trained_on"] = "WikiText-2 raw, training split only"
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    print(json.dumps(info, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
