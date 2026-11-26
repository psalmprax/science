"""Pins for the JetFlow / flow matching / MeanFlow head-to-head (packages/jetflow-video/jetflow_video/head_to_head.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "packages" / "jetflow-video"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from jetflow_video import head_to_head as h


def _mf() -> h.MeanFlowUNet:
    torch.manual_seed(0)
    model = h.build("meanflow-unet")
    with torch.no_grad():                              # zero-initialised paths: give every one something to say
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model  # type: ignore[return-value]


def test_meanflow_with_r_equal_t_is_a_velocity_model_and_r_matters() -> None:
    model = _mf()
    z = torch.randn(h.FRAMES, h.LATENT, 8, 8)
    a = model(z, 0.7, text_tokens=h.TEXT)
    assert torch.allclose(a, model(z, 0.7, text_tokens=h.TEXT, r=0.7))
    assert not torch.allclose(a, model(z, 0.7, text_tokens=h.TEXT, r=0.2))


def test_the_total_derivative_is_the_one_meanflow_needs() -> None:
    """jvp along (v, 0, 1) equals the finite difference of u(z + s v, r, t + s)."""
    model = _mf().double()
    z = torch.randn(1, h.FRAMES, h.LATENT, 8, 8, dtype=torch.float64)
    v = torch.randn_like(z)
    r, t = torch.tensor(0.3, dtype=torch.float64), torch.tensor(0.8, dtype=torch.float64)
    text = h.TEXT.double()

    def u(z_: torch.Tensor, r_: torch.Tensor, t_: torch.Tensor) -> torch.Tensor:
        return model(z_, t_, text_tokens=text, r=r_)
    _, dudt = torch.func.jvp(u, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    eps = 1e-5
    fd = (u(z + eps * v, r, t + eps) - u(z - eps * v, r, t - eps)) / (2 * eps)
    assert float((dudt - fd).abs().max()) < 1e-4 * float(fd.abs().max())


def test_meanflow_loss_trains_the_network() -> None:
    model = _mf()
    x = torch.randn(1, h.FRAMES, h.LATENT, 8, 8)
    loss = h.meanflow_loss(model, x, t=0.8, r=0.3)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


class _Oracle(torch.nn.Module):
    """The exact velocity (and average velocity) when all data is one clip x0: (z_t - x0) / t."""

    def __init__(self, x0: torch.Tensor) -> None:
        super().__init__()
        self.x0 = x0
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor, t: float, text_tokens: torch.Tensor | None = None,
                r: float | None = None) -> torch.Tensor:
        return (z - self.x0) / float(t)


@pytest.mark.parametrize(("how", "steps", "calls"), [("euler", 1, 1), ("euler", 4, 4), ("meanflow", 1, 1),
                                                     ("meanflow", 2, 2), ("jet2", 2, 4)])
def test_samplers_go_from_noise_to_data_with_the_calls_they_claim(how: str, steps: int, calls: int) -> None:
    x0 = torch.randn(h.FRAMES, h.LATENT, 8, 8)
    oracle = _Oracle(x0)
    counter = h.counter_for(oracle)
    out = h.sample(oracle, how, steps, n=3, seed=1)
    assert counter.calls == 3 * calls
    assert torch.allclose(out, x0.expand_as(out), atol=1e-2 if how == "jet2" else 1e-5)


def test_jetflow_sampler_is_counted_and_its_latent_kept() -> None:
    jet = h.build("fm-jetflow")
    h.install_latent_capture(jet)  # type: ignore[arg-type]
    counter = h.counter_for(jet)
    out = h.sample(jet, "jetflow", 2, n=2, seed=0)
    assert out.shape == (2, h.FRAMES, h.LATENT, 8, 8)
    assert counter.calls == 2 * 4                      # two Jet2 steps, two calls each, guidance off


def test_frechet_distance_and_copying_check() -> None:
    rng = np.random.default_rng(0)
    train = rng.random((120, 6, 3, 4, 4))
    held = rng.random((80, 6, 3, 4, 4))
    assert h.frechet_pca(train, held, held, k=8) == pytest.approx(0.0, abs=1e-8)
    noisy = h.frechet_pca(train, held, rng.random((80, 6, 3, 4, 4)) * 0.2, k=8)
    assert noisy > h.frechet_pca(train, held, rng.random((80, 6, 3, 4, 4)), k=8)
    assert h.memorisation_ratio(train, held, train[:40]) < 1e-6
    assert h.memorisation_ratio(train, held, rng.random((80, 6, 3, 4, 4))) == pytest.approx(1.0, abs=0.1)


def test_the_wavelet_latent_round_trips() -> None:
    clips = h.toy_clips("fluid", range(3000, 3003))
    assert clips.shape == (3, h.FRAMES, 3, h.SIZE, h.SIZE)
    assert np.abs(h.decode(h.encode(clips)) - clips).max() < 1e-5


def test_video_windows_read_a_real_file(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "train" / "Class" / "v_Class_g09_c01.avi"
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for i in range(40):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, : 16 + i, 2] = 255                         # red in BGR, growing to the right
        writer.write(frame)
    writer.release()
    clips = h.video_windows(path)
    assert len(clips) == 4 and clips[0].shape == (h.FRAMES, 3, h.SIZE, h.SIZE)
    assert 0.0 <= clips[0].min() and clips[0].max() <= 1.0
    assert clips[0][:, 0].mean() > 0.2 and clips[0][:, 2].mean() < 0.05       # red came out as red
    assert clips[0][-1, 0].mean() > clips[0][0, 0].mean()                     # and it grows in time
    folder, names = h.video_folder(tmp_path, "train")
    assert len(folder) == 4 and set(names) == {path.name}


def test_toy_splits_for_development_and_the_run_do_not_share_clips() -> None:
    import run_prereg_video_flow as run

    a_train, a_held = run._toy("fluid", "final")
    b_train, _ = run._toy("fluid", "dev")
    assert len(a_train) == 400 and len(a_held) == 100
    assert not np.array_equal(a_train[0], b_train[0])


def test_the_expectations_are_read_as_written() -> None:
    import run_prereg_video_flow as run

    def job(kind: str, arm: str, seed: int, fds: dict[tuple[str, int], float]) -> dict:
        return {"floor_frechet_pca": 1.0,
                "samples": [{"sampler": how, "steps": k, "frechet_pca": fds.get((how, k), 50.0) + seed * 0.1}
                            for how, k in run.GRID[arm]]}
    jobs = {}
    for kind in run.DATA:
        for seed in run.SEEDS:
            jobs[(kind, "fm-jetflow", seed)] = job(kind, "fm-jetflow", seed, {("jetflow", 2): 30.0, ("euler", 4): 40.0})
            jobs[(kind, "fm-unet", seed)] = job(kind, "fm-unet", seed, {("euler", 4): 40.0, ("euler", 1): 80.0,
                                                                         ("jet2", 2): 39.0})
            jobs[(kind, "meanflow-unet", seed)] = job(kind, "meanflow-unet", seed, {("meanflow", 1): 32.0})
    e = run.expectations(jobs)["ucf101"]
    assert e["E1 JetFlow beats flow matching at four calls"]["held"]           # 30 against 40
    assert e["E2 MeanFlow beats flow matching at one call"]["held"]            # 32 against 80
    assert e["E3 MeanFlow at one call is no worse than JetFlow at four"]["held"]   # 32 <= 30 / 0.9
    assert not e["E4 the Jet2 step beats Euler at four calls, UNet"]["held"]  # 39 against 40: not by 10 %
    assert e["E4 the Jet2 step beats Euler at four calls, JetFlow"]["held"]
    assert not e["E5 JetFlow's network beats the UNet, Euler 32"]["held"]     # equal
    assert run.beats([1.0, 1.0, 5.0], [2.0, 2.0, 2.0])["held"] is False     # mean 2.33 > 0.9 * 2
    assert run.beats([1.0, 1.0, 1.7], [2.0, 2.0, 1.0])["seeds_lower"] == 2
