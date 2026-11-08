"""Adapter connecting frozen chemistry submission.npz to the RM block2 J4 table transformer."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_j4_batch_xor_words(submission_npz: Path | str) -> np.ndarray:
    """Extract (19, 3976, 4) batch XOR words for J=4 from submission.npz.

    Follows the official competition batch_deltas specification:
      K = ceil(75 / 4) = 19 batches
      delta[0] = batch[0]
      delta[k] = batch[k] XOR batch[k-1]
    """
    path = Path(submission_npz).resolve()
    with np.load(path) as data:
        if "angle_words" not in data:
            raise ValueError(f"angle_words not found in {path}")
        words = np.asarray(data["angle_words"], dtype=np.int64)

    num_vectors, num_angles = words.shape
    num_batches = max(1, (num_angles + 3) // 4)  # 19 batches
    padded = np.zeros((num_vectors, num_batches * 4), dtype=np.int64)
    padded[:, :num_angles] = words
    batches = padded.reshape(num_vectors, num_batches, 4).transpose(1, 0, 2)
    deltas = batches.copy()
    deltas[1:] ^= batches[:-1]
    return deltas.astype(np.uint16)


def extract_j3_signed_words(
    submission_npz: Path | str,
    target_brot: int = 10,
    source_brot: int = 12,
) -> np.ndarray:
    """Extract (25, 3976, 3) batch words for J=3 from submission.npz.

    Follows the 4-address banking specification:
      K = 75 / 3 = 25 batches (exactly partitions 75 Givens angles)
      returns shape (25, 3976, 3) with target_brot-bit words.
      When source_brot > target_brot, words are scaled and quantized:
      w_target = round(w_source / 2**(source_brot - target_brot)) % 2**target_brot.
    """
    path = Path(submission_npz).resolve()
    with np.load(path) as data:
        if "angle_words" not in data:
            raise ValueError(f"angle_words not found in {path}")
        words = np.asarray(data["angle_words"], dtype=np.int64)

    num_vectors, num_angles = words.shape
    if num_angles != 75:
        raise ValueError(f"Expected 75 angles, got {num_angles}")

    if source_brot > target_brot:
        shift = source_brot - target_brot
        words = np.rint(words / (1 << shift)).astype(np.int64) % (1 << target_brot)

    num_batches = 25
    batches = words.reshape(num_vectors, num_batches, 3).transpose(1, 0, 2)
    return batches.astype(np.uint16)

