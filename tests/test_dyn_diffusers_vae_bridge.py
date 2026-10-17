"""Interface tests for the diffusers AutoencoderKL bridge (Stage-2 wiring).

Covers, WITHOUT any network dependency:
- config-only construction path runs the exact VideoVAEBridge interface
  surface (encode/decode/video/TT/psnr) with correct shapes and [0,1] output
- the VAE is deterministic in mode() (two identical calls, identical latent)
- channel-count refusal (no silent remap)
- honest provenance labels: config_only NEVER claims pretrained weights; a
  pretrained construction requires model_id
- graceful ImportError guidance when diffusers is absent

The PRETRAINED weights path is a network/cache-dependent probe: it is
SKIPPED (not passed) when weights are unavailable, and exercised for real
when the cache is warm.
"""
from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apedp_dyn.video.vae_bridge import DiffusersVAEBridge

diffusers_available = importlib.util.find_spec("diffusers") is not None

requires_diffusers = pytest.mark.skipif(
    not diffusers_available, reason="diffusers not installed in this venv"
)


@pytest.fixture(scope="module")
def bridge() -> "DiffusersVAEBridge":
    assert diffusers_available
    return DiffusersVAEBridge(config_only=True)


def _frame(h: int = 32, w: int = 32, seed: int = 3) -> np.ndarray:
    """Deterministic smooth [0,1] RGB frame."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.5 + 0.3 * np.sin(xx / 5.0) + 0.2 * np.cos(yy / 4.0)
    return np.stack([base, 0.5 + 0.5 * (xx / (w - 1)), base * base], axis=0)


@requires_diffusers
def test_config_only_round_trip_shapes_and_range(bridge: DiffusersVAEBridge) -> None:
    frame = _frame()
    z = bridge.encode_frame(frame)
    assert z.ndim == 3
    assert z.shape[0] == bridge.latent_channels
    # downsample factor: 32/8 = 4 for this config
    assert z.shape[1:] == (32 // bridge.downsample_factor, 32 // bridge.downsample_factor)
    recon = bridge.decode_frame(z, target_shape=(32, 32))
    assert recon.shape == (3, 32, 32)
    assert float(recon.min()) >= 0.0 and float(recon.max()) <= 1.0
    assert recon.dtype == np.float64


@requires_diffusers
def test_encoding_is_deterministic_in_mode(bridge: DiffusersVAEBridge) -> None:
    frame = _frame(12, 12, seed=8)
    z1 = bridge.encode_frame(frame)
    z2 = bridge.encode_frame(frame)
    assert np.array_equal(z1, z2)


@requires_diffusers
def test_channel_count_refusal_is_loud(bridge: DiffusersVAEBridge) -> None:
    with pytest.raises(ValueError, match="refusing silent remap"):
        bridge.decode_frame(np.zeros((bridge.latent_channels + 1, 4, 4)))


@requires_diffusers
def test_provenance_labels_are_honest(bridge: DiffusersVAEBridge) -> None:
    assert bridge.pretrained_weights_loaded is False
    assert bridge.source_model_id is None
    # pretrained path demanded: no model_id and not config_only raises
    with pytest.raises(ValueError, match="model_id or config_only"):
        DiffusersVAEBridge()
    # mutually exclusive flags raise
    with pytest.raises(ValueError, match="mutually exclusive"):
        DiffusersVAEBridge(model_id="stabilityai/sd-vae-ft-mse", config_only=True)


@requires_diffusers
def test_video_and_tt_composition(bridge: DiffusersVAEBridge) -> None:
    video = np.stack([_frame(16, 16, seed=k) for k in range(3)], axis=0)  # (3,3,16,16)
    lat = bridge.encode_video_to_latent(video)
    assert lat.shape == (3, bridge.latent_channels, 2, 2)
    vid = bridge.decode_latent_to_video(lat)
    assert vid.shape == (3, 3, 16, 16)
    tt = bridge.encode_video_to_tt(video, bond_ranks=2)
    vid_tt = bridge.decode_tt_to_video(tt)
    assert vid_tt.shape == (3, 3, 16, 16)
    psnr = bridge.compute_rgb_psnr(video, vid_tt)
    assert psnr > 0.0
    # interface parity with the analytic bridge's signature surface
    for name in ("encode_frame", "decode_frame", "encode_video_to_latent",
                 "decode_latent_to_video", "encode_video_to_tt",
                 "decode_tt_to_video", "compute_rgb_psnr"):
        assert callable(getattr(bridge, name))


@requires_diffusers
def test_frame_shape_guard(bridge: DiffusersVAEBridge) -> None:
    with pytest.raises(ValueError, match="Expected frame"):
        bridge.encode_frame(np.zeros((2, 8, 8)))
    with pytest.raises(ValueError, match="Expected \\(T, 3, H, W\\)"):
        bridge.encode_video_to_latent(np.zeros((2, 8, 8)))


@requires_diffusers
def test_graceful_import_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without diffusers importable, the error must point at the fix, not crash."""
    real_import = builtins.__import__

    def _block(name: str, *args, **kwargs):
        if name.startswith("diffusers"):
            raise ImportError("No module named 'diffusers' [blocked for test]")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(ImportError, match="pip install diffusers"):
        DiffusersVAEBridge(config_only=True)


@pytest.mark.skipif(
    not diffusers_available, reason="diffusers not installed"
)
@pytest.mark.parametrize("model_id", ["stabilityai/sd-vae-ft-mse"])
def test_pretrained_weights_path_when_available(model_id: str) -> None:
    """Real pretrained prior: exercised only when the HF cache/network is
    available; otherwise recorded as SKIPPED, never as a pass."""
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
        import urllib.request

        urllib.request.urlopen(
            f"https://huggingface.co/{model_id}/resolve/main/config.json", timeout=6
        ).read()
        weights_ok = True
    except Exception as exc:  # offline / dns / proxy
        pytest.skip(f"pretrained VAE weights unavailable in this env: {exc!r}")
    br = DiffusersVAEBridge(model_id=model_id)
    assert br.pretrained_weights_loaded is True
    assert br.source_model_id == model_id
    z = br.encode_frame(_frame(64, 64))
    assert z.shape[0] == br.latent_channels
