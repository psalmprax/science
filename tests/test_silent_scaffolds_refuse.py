"""Scaffolds whose failure looked like a result: each now refuses, or measures.

Sixth sweep. Two new defects of a familiar shape — a value read from the wrong
place, passing silently because nothing compared it:

- **The JetFlow stepper's native acceleration branch was a bare ``pass``.** The
  availability flag was true (the kernel really does export
  ``jetflow_compute_acceleration``), the kernel was never called, and the
  "Rust vs Python parity" pin monkeypatched the flag off and compared the
  Python path with itself. A parity test that cannot fail is decoration; the
  counter in the module is what lets a test show the branch ran.
- **The LLM proposer turned every failure into ``[]``.** A missing API key, an
  unreachable base URL, a provider 503 and "the model proposed nothing" all
  returned the same empty list, so a pipeline reported a null result from a
  process that never ran. Configuration and transport failures now raise; the
  deliberate case (disabled) returns empty and says so in ``last_outcome``.

Registered rather than changed in this sweep: the video package's
``XFORMERS_AVAILABLE`` / ``CHECKPOINT_AVAILABLE`` / ``LPIPS_AVAILABLE`` are
import probes whose fallbacks are real and exercised (SDPA, eager blocks, no
perceptual loss), so they are honest even though they name a capability rather
than a result. ``problems/finance.py`` keeps a placeholder RHS in its truth
tuple; it is inert (`finalize` is bypassed and the field is never integrated),
and rewriting the truth arity would change a contract the problem zoo relies
on, so it is recorded in `docs/SCAFFOLD_INVENTORY.md` instead.
"""
from __future__ import annotations

import inspect
import sys
import types

import pytest

# ------------------------------------------------------------- JetFlow stepper


def _stepper_and_torch():
    torch = pytest.importorskip("torch")
    stepper = pytest.importorskip("jetflow_video.stepper")
    return torch, stepper


def _velocity(torch):
    def velocity(z, t):
        return torch.sin(z) + t * z

    return velocity


def test_the_native_acceleration_is_actually_called(monkeypatch):
    """The flag was true and the kernel was never called; now it is."""
    torch, stepper = _stepper_and_torch()
    calls: list[float] = []

    def recording_kernel(v_from, v_to, h_eff):
        calls.append(float(h_eff))
        return (v_to - v_from) / h_eff

    monkeypatch.setattr(stepper, "_RUST_AVAILABLE", True)
    monkeypatch.setattr(stepper, "apedp_kernel", types.SimpleNamespace(
        jetflow_compute_acceleration=recording_kernel))
    before = stepper.NATIVE_ACCELERATION_CALLS[0]

    velocity = _velocity(torch)
    z = torch.randn(2, 3, dtype=torch.float64)
    stepper.JetFlowStepper(stepper.JetFlowConfig()).compute_acceleration_torch(
        velocity, z, 0.3, velocity(z, 0.3), dt=0.1)

    assert calls, "the availability flag was true and the kernel was never called"
    assert stepper.NATIVE_ACCELERATION_CALLS[0] == before + 1


def test_the_central_difference_path_also_uses_the_kernel(monkeypatch):
    torch, stepper = _stepper_and_torch()
    calls: list[float] = []

    def recording_kernel(v_from, v_to, h_eff):
        calls.append(float(h_eff))
        return (v_to - v_from) / h_eff

    monkeypatch.setattr(stepper, "_RUST_AVAILABLE", True)
    monkeypatch.setattr(stepper, "apedp_kernel", types.SimpleNamespace(
        jetflow_compute_acceleration=recording_kernel))
    velocity = _velocity(torch)
    z = torch.randn(2, 3, dtype=torch.float64)
    config = stepper.JetFlowConfig(use_central_diff=True)
    stepper.JetFlowStepper(config).compute_acceleration_torch(
        velocity, z, 0.3, velocity(z, 0.3), dt=0.1)
    assert len(calls) == 1, calls


def test_a_refusing_kernel_falls_back_to_the_same_acceleration(monkeypatch):
    torch, stepper = _stepper_and_torch()

    def refusing_kernel(*_args, **_kwargs):
        raise RuntimeError("kernel refused this shape")

    velocity = _velocity(torch)
    z = torch.randn(2, 3, dtype=torch.float64)
    v = velocity(z, 0.3)
    simulator = stepper.JetFlowStepper(stepper.JetFlowConfig())

    monkeypatch.setattr(stepper, "_RUST_AVAILABLE", False)
    reference = simulator.compute_acceleration_torch(velocity, z, 0.3, v, dt=0.1)

    monkeypatch.setattr(stepper, "_RUST_AVAILABLE", True)
    monkeypatch.setattr(stepper, "apedp_kernel", types.SimpleNamespace(
        jetflow_compute_acceleration=refusing_kernel))
    fallback = simulator.compute_acceleration_torch(velocity, z, 0.3, v, dt=0.1)

    assert torch.allclose(fallback, reference, atol=1e-12)


def test_the_placeholder_branch_is_gone_from_the_source():
    torch, stepper = _stepper_and_torch()
    del torch  # the source check needs the module, not the tensors
    source = inspect.getsource(stepper)
    assert "# Try Rust fast path if available" not in source
    assert "jetflow_compute_acceleration(" in inspect.getsource(
        stepper._acceleration_between)


# --------------------------------------------------------------- LLM proposer


def _proposer(**kwargs):
    from apedp_dyn.grammar.llm_proposer import LLMProposer

    return LLMProposer(**kwargs)


def _clear_keys(monkeypatch):
    for env in ("OPENROUTER_API_KEY", "APEDP_LLM_KEY"):
        monkeypatch.delenv(env, raising=False)


def test_a_disabled_proposer_returns_nothing_and_says_why(monkeypatch):
    _clear_keys(monkeypatch)
    proposer = _proposer(enabled=False)
    assert list(proposer.propose(object())) == []
    assert proposer.last_outcome == "disabled"


def test_an_enabled_proposer_without_a_key_refuses(monkeypatch):
    from apedp_dyn.grammar.llm_proposer import LLMProposerUnavailable

    _clear_keys(monkeypatch)
    proposer = _proposer(enabled=True, provider="openrouter")
    with pytest.raises(LLMProposerUnavailable) as exc:
        proposer.propose(object())
    assert "OPENROUTER_API_KEY" in str(exc.value)
    assert proposer.last_outcome == "no_api_key"


def test_a_refused_request_is_not_a_null_result(monkeypatch):
    from apedp_dyn.grammar.llm_proposer import LLMProposerUnavailable

    def refusing_post(*_args, **_kwargs):
        raise RuntimeError("503 from provider")

    monkeypatch.setitem(sys.modules, "httpx",
                        types.SimpleNamespace(post=refusing_post))
    proposer = _proposer(enabled=True, provider="openrouter", api_key="test-key")
    with pytest.raises(LLMProposerUnavailable) as exc:
        proposer.propose(object())
    assert "503" in str(exc.value)
    assert proposer.last_outcome == "no_api_key" or "failed" in str(exc.value)
