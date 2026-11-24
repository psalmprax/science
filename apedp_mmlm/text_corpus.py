"""WikiText-2 (raw) as bytes, fetched once and refused if a file's hash differs.

Status: experimental

The source is the copy in the PyTorch examples repository. Files are cached in the
ignored ``data/mmlm/``. ``split_bytes("valid_a")`` and ``"valid_b"`` are the two
halves of the validation split, cut at the line boundary nearest its middle: the
first fits the calibration tables and the mixer, the second measures them during
development. The test split is read only by the preregistered run.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "mmlm"
SOURCE = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
SHA256 = {
    "train": "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f",
    "valid": "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8",
    "test": "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0",
}


def split_bytes(split: str) -> bytes:
    name = "valid" if split in ("valid_a", "valid_b") else split
    path = CACHE / f"wiki.{name}.txt"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{SOURCE}/{name}.txt", timeout=120) as response:  # noqa: S310 - fixed https host
            path.write_bytes(response.read())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise ValueError(f"{path} does not match its pinned SHA-256")
    if split in ("valid_a", "valid_b"):
        cut = data.index(b"\n", len(data) // 2) + 1
        return data[:cut] if split == "valid_a" else data[cut:]
    return data
