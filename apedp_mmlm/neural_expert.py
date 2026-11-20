"""The one neural expert of MMLM-CM: a small byte-level LSTM trained on CPU.

Status: experimental

Everything else in the model counts. This expert is a single-layer LSTM over
bytes (a few hundred thousand weights), trained by backpropagation through time
on the training text only. It is not the model: its predictions enter the mixer
as one more input beside the counting experts, and
``scripts/run_prereg_mmlm_cm.py`` reports the model with and without it, and the
network alone.

``bit_probabilities`` returns what the mixer consumes: for every byte of a
stream and each of its eight bits (most significant first), the probability that
the bit is 1 given the earlier bytes and the higher bits of the same byte.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class ByteLSTM(nn.Module):
    def __init__(self, hidden: int = 256, embed: int = 64, layers: int = 1) -> None:
        super().__init__()
        self.config = {"hidden": hidden, "embed": embed, "layers": layers}
        self.embed = nn.Embedding(256, embed)
        self.lstm = nn.LSTM(embed, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, 256)

    def forward(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        h, state = self.lstm(self.embed(x), state)
        return self.out(h), state


def train(
    data: bytes, *, seconds: float, hidden: int = 256, embed: int = 64, layers: int = 1, batch: int = 64,
    steps_per_batch: int = 100, lr: float = 2e-3, lr_floor: float = 2e-4, seed: int = 0,
    log: Any = None, checkpoint: Path | None = None,
) -> tuple[ByteLSTM, dict[str, Any]]:
    """Train for a wall-clock budget; the learning rate follows a cosine over that budget."""
    torch.manual_seed(seed)
    model = ByteLSTM(hidden, embed, layers)
    stream = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    width = (len(stream) - 1) // batch
    x_all = stream[: width * batch].view(batch, width)
    y_all = stream[1 : width * batch + 1].view(batch, width)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    start, step, state, position, running = time.time(), 0, None, 0, None
    last_save = start
    while True:
        elapsed = time.time() - start
        if elapsed >= seconds:
            break
        for group in optimiser.param_groups:
            group["lr"] = lr_floor + 0.5 * (lr - lr_floor) * (1 + math.cos(math.pi * elapsed / seconds))
        if position + steps_per_batch > width:
            position, state = 0, None
        x = x_all[:, position : position + steps_per_batch]
        y = y_all[:, position : position + steps_per_batch]
        position += steps_per_batch
        logits, state = model(x, state)
        state = tuple(s.detach() for s in state)
        loss = loss_fn(logits.reshape(-1, 256), y.reshape(-1))
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        step += 1
        bits = loss.item() / math.log(2)
        running = bits if running is None else 0.98 * running + 0.02 * bits
        if log is not None and step % 50 == 0:
            log(f"step {step} bytes {step * batch * steps_per_batch} elapsed {elapsed:.0f}s train bits/byte {running:.3f}")
        if checkpoint is not None and time.time() - last_save > 300:
            save(model, checkpoint)
            last_save = time.time()
    info = {"steps": step, "bytes_seen": step * batch * steps_per_batch, "seconds": round(time.time() - start, 1),
            "train_bits_per_byte_running": running, "parameters": sum(p.numel() for p in model.parameters()),
            "seed": seed, "batch": batch, "steps_per_batch": steps_per_batch, "lr": lr, "lr_floor": lr_floor, **model.config}
    if checkpoint is not None:
        save(model, checkpoint)
    return model, info


def save(model: ByteLSTM, path: Path) -> None:
    torch.save({"config": model.config, "state": model.state_dict()}, path)


def load(path: Path) -> ByteLSTM:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = ByteLSTM(**blob["config"])
    model.load_state_dict(blob["state"])
    return model


@torch.no_grad()
def bit_probabilities(model: ByteLSTM, data: bytes, *, segments: int = 64, chunk: int = 2000) -> np.ndarray:
    """``[len(data), 8]`` probabilities that each bit is 1, given the past and the higher bits.

    The stream is cut into ``segments`` equal runs that are read in parallel, each
    starting from an empty state, so the first bytes of a run are predicted with
    less context. The first byte of every run is predicted from nothing (uniform).
    """
    model.eval()
    n = len(data)
    array: np.ndarray = np.frombuffer(data, dtype=np.uint8)
    width = math.ceil(n / segments)
    padded: np.ndarray = np.zeros(segments * width, dtype=np.int64)
    padded[:n] = array
    grid = torch.from_numpy(padded).view(segments, width)
    out: np.ndarray = np.empty((segments, width, 8), dtype=np.float32)
    state = None
    carry = torch.full((segments, 1, 256), -math.log(256.0))  # the first byte of a run is predicted from nothing
    for begin in range(0, width, chunk):
        x = grid[:, begin : begin + chunk]
        logits, state = model(x, state)
        logp = torch.log_softmax(logits, dim=-1)  # prediction for the byte after each input byte
        previous, carry = carry, logp[:, -1:, :]
        logp = torch.cat([previous, logp[:, :-1, :]], dim=1)  # aligned: row t predicts byte t
        prob = logp.exp().double()
        target = x  # the byte at each position
        for j in range(8):
            groups = prob.view(x.shape[0], x.shape[1], 1 << (j + 1), 1 << (7 - j)).sum(-1)
            prefix = target >> (8 - j) if j else torch.zeros_like(target)
            one = groups.gather(2, (2 * prefix + 1).unsqueeze(-1)).squeeze(-1)
            zero = groups.gather(2, (2 * prefix).unsqueeze(-1)).squeeze(-1)
            out[:, begin : begin + x.shape[1], j] = (one / (one + zero).clamp_min(1e-300)).float().numpy()
    result: np.ndarray = out.reshape(segments * width, 8)[:n]
    return result
