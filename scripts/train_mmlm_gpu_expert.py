"""Train MMLM-CM's transformer expert (on a GPU if there is one) and measure the mixture with it.

    python scripts/train_mmlm_gpu_expert.py --hours 2 --corpus wikitext103

Written to be run in Google Colab (``notebooks/mmlm_gpu_expert_colab.ipynb``); it also runs on a CPU.

- ``--corpus wikitext2``: the 10.8 MB training split already used by MMLM-CM.
- ``--corpus wikitext103``: the 540 MB training text of WikiText-103, fetched with the ``datasets``
  package. WikiText-103 and WikiText-2 share their validation and test articles, so the network sees
  fifty times more training text and is measured on exactly the text MMLM-CM was. The tokenised
  version is used (rare words written ``<unk>``), because that is the form of the text MMLM-CM is
  measured on.

After training it measures, on the second half of the validation split (the development half):
the network alone, and the full mixture with this network in place of the LSTM (counting experts
from the WikiText-2 training split, calibration and mixer from the first half of the validation
split, exactly as in ``scripts/study_mmlm_cm_dev.py``). It never reads the test split.

Writes the weights and a JSON record beside them. Bring both files back to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_mmlm import transformer_expert  # noqa: E402
from apedp_mmlm.context_mixing import ContextMixingLM  # noqa: E402
from apedp_mmlm.text_corpus import split_bytes  # noqa: E402

REFERENCE = {"mixture with the CPU LSTM": 1.4757, "the CPU LSTM alone": 1.7890, "best primed compressor (PPMd order 16)": 1.5363}


def training_text(corpus: str) -> bytes:
    if corpus == "wikitext2":
        return split_bytes("train")
    from datasets import load_dataset  # only needed for the large corpus; present in Colab

    rows = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    text = str("".join(rows["text"])).encode("utf-8")
    probe = split_bytes("valid_b")
    sample = probe[len(probe) // 2 : len(probe) // 2 + 400]
    if sample in text:
        raise SystemExit("the training text contains validation text; refusing to train on it")
    return text


def alone_bits(bits: np.ndarray, data: bytes) -> float:
    """Bits per byte of the network's own prediction, from its bit probabilities."""
    array: np.ndarray = np.frombuffer(data, dtype=np.uint8)
    total = 0.0
    for j in range(8):
        one = ((array >> (7 - j)) & 1).astype(bool)
        p = np.clip(bits[:, j].astype(np.float64), 1e-9, 1 - 1e-9)
        total += float(-np.log2(np.where(one, p, 1 - p)).sum())
    return total / len(array)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--corpus", choices=("wikitext2", "wikitext103"), default="wikitext103")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-mixture", action="store_true")
    args = parser.parse_args()
    out = args.out or ROOT / "benchmarks" / "mmlm" / f"transformer_expert_{args.corpus}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = training_text(args.corpus)
    fit_text, dev_text = split_bytes("valid_a"), split_bytes("valid_b")
    print(f"training on {len(text):,} bytes ({args.corpus}) for {args.hours} h on {transformer_expert.pick_device()}", flush=True)
    model, info = transformer_expert.train(
        text, fit_text, seconds=args.hours * 3600, d_model=args.d_model, layers=args.layers, heads=args.heads,
        context=args.context, dropout=args.dropout, batch=args.batch, log=lambda m: print(m, flush=True), checkpoint=out)
    record = {**info, "corpus": args.corpus, "training_bytes": len(text),
              "training_sha256": hashlib.sha256(text).hexdigest(),
              "weights_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "reference_development_numbers": REFERENCE}
    started = time.time()
    dev_bits = transformer_expert.bit_probabilities(model, dev_text)
    record["development"] = {"network_alone_bits_per_byte": alone_bits(dev_bits, dev_text)}
    print(f"network alone on the development half: {record['development']['network_alone_bits_per_byte']:.4f} bits per byte", flush=True)
    if not args.skip_mixture:
        mixer = ContextMixingLM().fit_counts(split_bytes("train")).calibrate(fit_text)
        fit = mixer.features(fit_text, transformer_expert.bit_probabilities(model, fit_text))
        dev = mixer.features(dev_text, dev_bits)
        mixer.release_counts()
        mixer.fit_mixer(fit)
        record["development"]["mixture_bits_per_byte"] = float(mixer.byte_bits(dev).mean())
        mixer.fit_mixer(fit, tuple(e for e in mixer.experts if e != "nn"))
        record["development"]["mixture_without_the_network"] = float(mixer.byte_bits(dev).mean())
        print(f"mixture with this network: {record['development']['mixture_bits_per_byte']:.4f} bits per byte "
              f"(without any network {record['development']['mixture_without_the_network']:.4f})", flush=True)
    record["development"]["seconds"] = round(time.time() - started, 1)
    for name, value in REFERENCE.items():
        print(f"  reference, {name}: {value:.4f}", flush=True)
    if not math.isfinite(record["development"]["network_alone_bits_per_byte"]):
        raise SystemExit("the network's predictions are not finite")
    out.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} and {out.with_suffix('.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
