"""A larger neural expert for MMLM-CM, trained on a GPU: a causal byte-level transformer.

Status: experimental

The model's first network (:mod:`apedp_mmlm.neural_expert`) is an 806k-weight
LSTM trained for 40 minutes on two CPU cores. It is already the best single
expert, so the network is where more compute pays. This module is the same
kind of expert at GPU scale: a decoder-only transformer over bytes, trained
with mixed precision for a wall-clock budget, keeping the weights that scored
best on held-out text. It runs on a CPU as well (slowly); nothing else in
MMLM needs a GPU.

It enters the mixture exactly as the LSTM does. ``bit_probabilities`` returns,
for every byte of a stream and each of its eight bits, the probability that
the bit is 1 given the earlier bytes and the higher bits of the same byte;
``ContextMixingLM.features(data, neural=...)`` takes it from there.

A byte is predicted from at most ``context`` earlier bytes. A stream is scored
in overlapping windows (stride half a window), so every byte after the first
window has at least half a window of context; window boundaries depend on
position only, never on content. The first byte of a stream is predicted from
nothing (uniform).

``scripts/train_mmlm_gpu_expert.py`` trains it and measures the mixture on the
development half; ``notebooks/mmlm_gpu_expert_colab.ipynb`` runs that in Colab.
"""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp.grad_scaler import GradScaler


class _Block(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, c = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        h = h + self.drop(self.proj(attended.transpose(1, 2).reshape(b, t, c)))
        out: torch.Tensor = h + self.drop(self.mlp(self.norm2(h)))
        return out


class ByteTransformer(nn.Module):
    def __init__(self, d_model: int = 512, layers: int = 8, heads: int = 8, context: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be a multiple of heads")
        self.config = {"d_model": d_model, "layers": layers, "heads": heads, "context": context, "dropout": dropout}
        self.context = context
        self.token = nn.Embedding(256, d_model)
        self.position = nn.Embedding(context, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(_Block(d_model, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)
        self.apply(self._initialise)
        self.head.weight = self.token.weight  # one table for reading and writing bytes

    @staticmethod
    def _initialise(module: nn.Module) -> None:
        # small weights: with a shared byte table, default-sized embeddings give logits far too large to train from
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits for the byte after each position of ``x`` (``[batch, time]``, time at most ``context``)."""
        h = self.drop(self.token(x) + self.position(torch.arange(x.shape[1], device=x.device)))
        for block in self.blocks:
            h = block(h)
        out: torch.Tensor = self.head(self.norm(h))
        return out


def pick_device(device: str | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def bits_per_byte(model: ByteTransformer, data: bytes, device: torch.device, batch: int = 64) -> float:
    """Mean bits per byte over non-overlapping windows: quick, used to pick the best weights while training."""
    model.eval()
    stream = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    t = model.context
    windows = (len(stream) - 1) // t
    x = stream[: windows * t].view(windows, t)
    y = stream[1 : windows * t + 1].view(windows, t)
    total = 0.0
    for begin in range(0, windows, batch):
        logits = model(x[begin : begin + batch].to(device)).float()
        total += float(F.cross_entropy(logits.reshape(-1, 256), y[begin : begin + batch].to(device).reshape(-1), reduction="sum"))
    return total / (windows * t) / math.log(2)


def train(
    data: bytes, valid: bytes, *, seconds: float, d_model: int = 512, layers: int = 8, heads: int = 8, context: int = 512,
    dropout: float = 0.1, batch: int = 64, lr: float = 6e-4, weight_decay: float = 0.1, warmup_steps: int = 500,
    seed: int = 0, device: str | None = None, eval_every: int = 500, eval_bytes: int = 200_000,
    log: Callable[[str], None] | None = None, checkpoint: Path | None = None,
) -> tuple[ByteTransformer, dict[str, Any]]:
    """Train for a wall-clock budget; returns the weights that scored best on ``valid`` and the training record."""
    dev = pick_device(device)
    torch.manual_seed(seed)
    model = ByteTransformer(d_model, layers, heads, context, dropout).to(dev)
    stream = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    plain = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimiser = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}],
                                  lr=lr, betas=(0.9, 0.95))
    use_amp = dev.type == "cuda"
    scaler = GradScaler(dev.type, enabled=use_amp)
    generator = torch.Generator(device=dev).manual_seed(seed)
    offsets = torch.arange(context + 1, device=dev)
    held_out = valid[:eval_bytes]
    start, step, running = time.time(), 0, None
    best: tuple[float, dict[str, torch.Tensor]] | None = None
    curve: list[dict[str, float]] = []

    def evaluate() -> None:
        nonlocal best
        score = bits_per_byte(model, held_out, dev)
        curve.append({"step": step, "seconds": round(time.time() - start, 1), "valid_bits_per_byte": score})
        if best is None or score < best[0]:
            best = (score, {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()})
            if checkpoint is not None:
                save(model, checkpoint)
        if log is not None:
            log(f"step {step} elapsed {time.time() - start:.0f}s train {running or float('nan'):.3f} valid {score:.3f} bits/byte")
        model.train()

    model.train()
    while True:
        elapsed = time.time() - start
        if elapsed >= seconds:
            break
        scale = min(1.0, (step + 1) / warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, elapsed / seconds)))
        for group in optimiser.param_groups:
            group["lr"] = max(lr * scale, lr * 0.02)
        starts = torch.randint(0, len(stream) - context - 1, (batch,), device=dev, generator=generator)
        window = stream[starts[:, None] + offsets[None, :]].long()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            loss = F.cross_entropy(model(window[:, :-1]).reshape(-1, 256).float(), window[:, 1:].reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser)
        scaler.update()
        step += 1
        bits = loss.item() / math.log(2)
        running = bits if running is None else 0.98 * running + 0.02 * bits
        if step % eval_every == 0:
            evaluate()
    evaluate()
    assert best is not None
    final = copy.deepcopy(model).to("cpu")
    final.load_state_dict(best[1])
    info = {"steps": step, "bytes_seen": step * batch * context, "seconds": round(time.time() - start, 1),
            "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "mixed_precision": use_amp, "parameters": sum(p.numel() for p in final.parameters()),
            "best_valid_bits_per_byte": best[0], "valid_curve": curve, "seed": seed, "batch": batch, "lr": lr,
            "weight_decay": weight_decay, **final.config}
    if checkpoint is not None:
        save(final, checkpoint)
    return final, info


def save(model: ByteTransformer, path: Path) -> None:
    torch.save({"config": model.config, "state": {k: v.to("cpu") for k, v in model.state_dict().items()}}, path)


def load(path: Path) -> ByteTransformer:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = ByteTransformer(**blob["config"])
    model.load_state_dict(blob["state"])
    return model


def _bit_probabilities_from(logp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``[m, 8]``: per row of byte log-probabilities, the probability of each bit of ``target`` being 1 along its path."""
    prob = logp.double().exp()
    out = torch.empty((len(target), 8), dtype=torch.float64)
    for j in range(8):
        groups = prob.view(len(target), 1 << (j + 1), 1 << (7 - j)).sum(-1)
        prefix = target >> (8 - j) if j else torch.zeros_like(target)
        one = groups.gather(1, (2 * prefix + 1).unsqueeze(-1)).squeeze(-1)
        zero = groups.gather(1, (2 * prefix).unsqueeze(-1)).squeeze(-1)
        out[:, j] = one / (one + zero).clamp_min(1e-300)
    return out


@torch.no_grad()
def bit_probabilities(model: ByteTransformer, data: bytes, *, batch: int = 64, device: str | None = None) -> np.ndarray:
    """``[len(data), 8]`` probabilities that each bit is 1, given the earlier bytes and the higher bits.

    float32, each strictly between 0 and 1 (the nearest representable value when the model is certain).
    """
    dev = pick_device(device)
    model = model.to(dev).eval()
    n, t = len(data), model.context
    stride = max(1, t // 2)
    stream = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    out: np.ndarray = np.full((n, 8), 0.5, dtype=np.float32)  # byte 0 is predicted from nothing
    starts = list(range(0, max(1, n - 1), stride))
    for begin in range(0, len(starts), batch):
        chunk = starts[begin : begin + batch]
        x = torch.zeros((len(chunk), t), dtype=torch.long)
        for row, s in enumerate(chunk):
            piece = stream[s : s + t]
            x[row, : len(piece)] = piece
        logp = torch.log_softmax(model(x.to(dev)).float(), dim=-1).to("cpu")
        for row, s in enumerate(chunk):
            first = 0 if s == 0 else t - stride  # later windows keep only positions with at least half a window of context
            low, high = s + first + 1, min(s + t + 1, n)  # the bytes this window is responsible for
            if low >= high:
                continue
            predicted = torch.arange(low, high)
            rows = predicted - s - 1
            out[predicted.numpy()] = _bit_probabilities_from(logp[row, rows], stream[predicted]).float().numpy()
    model.to("cpu")
    # A confident prediction rounds to exactly 0 or 1 in float32, and a consumer that takes a logarithm
    # or a logit of it gets an infinity. Every probability returned is strictly inside (0, 1).
    np.clip(out, np.finfo(np.float32).tiny, np.nextafter(np.float32(1.0), np.float32(0.0)), out=out)
    return out
