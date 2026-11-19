"""Development study: MMLM tracks A, B and C on real text, in bits per byte.

    python scripts/study_mmlm_text.py

NOT a preregistered result. Text: this repository's own prose (MIT licence), the
non-empty lines of ``docs/SYSTEM_FLAWS_AUDIT.md`` up to the 2026-11-17 round
heading's end of file, normalised to the tokeniser's tokens joined by single
spaces so that every method codes exactly the same bytes. First 80 % of lines
train, last 20 % test (chronologically older text, since the file is newest
first).

Cost of a line under a model: minus log2 of each next-token probability,
end-of-line token included; a token outside the vocabulary costs the unknown
token plus 8 bits per byte to spell it. Compressors (gzip, bz2, lzma at their
strongest settings) are charged conditionally: bytes to compress train + test
minus bytes to compress train. An order-0 byte entropy is the trivial reference.
"""
from __future__ import annotations

import bz2
import gzip
import json
import lzma
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apedp_mmlm.engines import MMLMTokenizer, WeightedAutomatonEngine  # noqa: E402
from apedp_mmlm.models import (  # noqa: E402
    AttentionFreeMMLM_C0,
    HybridMMLM_B0,
    NonNeuralMMLM_A0,
    RecurrentBaseline,
    SSMBaseline,
    TransformerBaseline,
)

OUT = ROOT / "benchmarks" / "mmlm_text_dev_study.json"
SOURCE = ROOT / "docs" / "SYSTEM_FLAWS_AUDIT.md"
VOCAB = 2048


class AutomatonOnly:
    """The Kneser-Ney automaton by itself, to show what the other engines add or cost."""

    def __init__(self, tokenizer: MMLMTokenizer) -> None:
        self.tokenizer = tokenizer
        self.engine = WeightedAutomatonEngine(tokenizer.vocab_size)

    def fit(self, corpus: list[str]) -> AutomatonOnly:
        self.engine.fit_sequences([self.tokenizer.encode(line) for line in corpus])
        return self

    def next_token_proba(self, context_ids: list[int]) -> np.ndarray:
        return self.engine.predict_next_proba(context_ids)


def model_bits(model: Any, tokenizer: MMLMTokenizer, lines: list[str]) -> float:
    unk = tokenizer.token_to_id.get("<unk>", -1)
    total = 0.0
    for line in lines:
        words = line.split(" ")
        ids = tokenizer.encode(line)
        offset = len(ids) - len(words) - 1 if len(ids) > len(words) + 1 else 0  # a leading start token, if any
        for i in range(max(1, offset), len(ids)):
            total -= float(np.log2(max(float(model.next_token_proba(ids[:i])[ids[i]]), 1e-300)))
            word_index = i - offset
            if ids[i] == unk and 0 <= word_index < len(words):
                total += 8.0 * len(words[word_index].encode())
    return total


def main() -> int:
    raw = [line for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [" ".join(MMLMTokenizer.split_raw(line)) for line in raw]
    lines = [line for line in lines if line]
    cut = int(0.8 * len(lines))
    train, test = lines[:cut], lines[cut:]
    test_bytes = sum(len(line.encode()) + 1 for line in test)
    tokenizer = MMLMTokenizer(max_vocab_size=VOCAB).fit(train)
    results: dict[str, Any] = {}
    blob_train = ("\n".join(train) + "\n").encode()
    blob_all = blob_train + ("\n".join(test) + "\n").encode()
    for name, compress in (("gzip-9", lambda b: gzip.compress(b, 9)), ("bz2-9", lambda b: bz2.compress(b, 9)),
                           ("lzma-9e", lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))):
        results[name] = {"bits_per_byte": 8.0 * (len(compress(blob_all)) - len(compress(blob_train))) / test_bytes}
    counts = Counter(blob_train)
    probs = {b: (c + 1) / (len(blob_train) + 256) for b, c in counts.items()}
    order0 = -sum(np.log2(probs.get(b, 1 / (len(blob_train) + 256))) for b in blob_all[len(blob_train):])
    results["order-0 bytes"] = {"bits_per_byte": float(order0) / test_bytes}
    for cls in (AutomatonOnly, NonNeuralMMLM_A0, HybridMMLM_B0, AttentionFreeMMLM_C0,
                TransformerBaseline, SSMBaseline, RecurrentBaseline):
        started = time.perf_counter()
        model = cls(tokenizer).fit(train)
        fitted = time.perf_counter() - started
        bits = model_bits(model, tokenizer, test)
        results[cls.__name__] = {"bits_per_byte": bits / test_bytes, "fit_seconds": round(fitted, 1),
                                 "eval_seconds": round(time.perf_counter() - started - fitted, 1)}
        print(cls.__name__, results[cls.__name__], flush=True)
    summary = {"status": "development study, not preregistered", "source": str(SOURCE.relative_to(ROOT)),
               "licence": "MIT (this repository)", "train_lines": len(train), "test_lines": len(test),
               "test_bytes": test_bytes, "vocabulary": tokenizer.vocab_size, "results": results}
    OUT.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for name, row in sorted(results.items(), key=lambda kv: kv[1]["bits_per_byte"]):
        print(f"{name:24s} {row['bits_per_byte']:.3f} bits/byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
