"""Distillation-parity: TRAINED 4x-run head weights executed by the Rust kernel.

Status: audit-only

The 4x budget rung trains a LearnedFlowHead on CPU (`train_budget_s=360`); the
benchmark's `--save-head` export persists those weights. This test loads them
and re-verifies the torch <-> Rust kernel forward contract on THE WEIGHTS THAT
WILL SERVE, not on fresh random inits: the serving path must reproduce the
studio path pixel-for-pixel.

Skips (reports, never passes) when either:
- the weights artifact is absent (run the 4x rung with --save-head first), or
- the Rust kernel extension is not importable.

Parity target: max|torch - kernel| <= 1e-12 on dense random pixel contexts
(train-clip frames only — holdout ids are never consumed here; the context
frames are drawn from the train split of the domain manifest).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apedp_dyn.video.flow_head_rust import RustFlowHeadSpec, pack_weights
from apedp_dyn.video.fluid_domain import fluid_domain_manifest, make_domain_set
from apedp_dyn.video.learned_flow_head import FlowHeadConfig, LearnedFlowHead

WEIGHTS_PATH = _REPO / "benchmarks" / "learned_flow_dcstt_4x_periodic_head.pt"

kernel_available = importlib.util.find_spec("apedp_kernel") is not None


def _parity_guard():
    if not WEIGHTS_PATH.exists():
        pytest.skip(
            f"trained head artifact {WEIGHTS_PATH.name} absent — run "
            "`run_learned_flow_dcstt_benchmark.py --save-head` first"
        )
    if not kernel_available:
        pytest.skip("Rust kernel extension not importable")


def test_trained_weights_drive_the_kernel_in_parity() -> None:
    _parity_guard()
    import torch

    import apedp_kernel

    spec = RustFlowHeadSpec()  # context=2, hidden=24, n_layers=3, scale=4.0
    cfg = FlowHeadConfig()
    head = LearnedFlowHead(cfg)
    state = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
    head.load_state_dict({k: torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)
                          for k, v in state.items()})
    head.double().eval()

    packed = pack_weights(head.state_dict())
    assert packed.shape[0] == spec.weights_length, (
        f"packed {packed.shape[0]} != spec {spec.weights_length} — layout drift"
    )

    # Context frames come from the TRAIN split only (holdout stays untouched).
    man = fluid_domain_manifest(n_clips=10)
    holdout = set(man["holdout_ids"])
    t, h, w = 16, 64, 64
    train_ids = tuple(cid for cid in man["train_ids"] if cid not in holdout)
    clips = make_domain_set(train_ids, tuple(holdout), t=t, h=h, w=w)
    best = 0.0
    with torch.no_grad():
        for cid in list(train_ids)[:2]:
            for ti in (2, 5, 9, 13):
                z_prev = clips[cid][ti - 1]
                z_curr = clips[cid][ti]
                zp = torch.from_numpy(z_prev[1:3]).unsqueeze(0)
                zc = torch.from_numpy(z_curr[1:3]).unsqueeze(0)
                v_t = head(zp, zc).numpy()
                vx_k, vy_k = apedp_kernel.flow_head_forward(
                    np.ascontiguousarray(z_prev[1:3]).ravel(),
                    np.ascontiguousarray(z_curr[1:3]).ravel(),
                    cfg.context_channels, cfg.hidden_dim, cfg.n_layers,
                    cfg.flow_scale, h, w, packed,
                )
                err = max(
                    float(np.max(np.abs(np.asarray(vx_k).reshape(h, w) - v_t[0, 0]))),
                    float(np.max(np.abs(np.asarray(vy_k).reshape(h, w) - v_t[0, 1]))),
                )
                best = max(best, err)
                assert err <= 1e-12, f"kernel/torch drift {err:.3e} at {cid} t={ti}"
    # Emit the actual parity so regressions carry the number, not just a bool.
    print(f"trained-weights kernel parity max|Δv| = {best:.3e}")
