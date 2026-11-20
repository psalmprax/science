"""Placeholder kernels must refuse rather than answer with a plausible constant.

The SSM scan copied its input, the "selective scan" multiplied by ``0.9``, and the
latent decoder returned ``"x0"`` for every vector. A constant is indistinguishable from
a result, so each now raises ``NotImplementedError`` — unconditionally. An earlier
version let ``APEDP_ALLOW_PLACEHOLDER_KERNELS=1`` restore the fake answers, which meant
a fabricated number could still be requested; that opt-in has been removed, and the
integer-relation search names the brute-force method it actually uses instead of
advertising PSLQ.
"""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "rust" / "apedp-kernel" / "src"


def test_latent_decode_refuses_in_python():
    from apedp.advanced_search import LatentSpace

    space = LatentSpace(dim=8)
    with pytest.raises(NotImplementedError, match="no inverse"):
        space.decode([0.1] * 8)
    # The hash itself still works; only decoding is refused.
    assert len(space.encode("a*x/(1+b*x)")) == 8


def test_latent_decode_refuses_in_native_code_with_no_opt_in():
    source = (KERNEL / "latent.rs").read_text(encoding="utf-8")
    assert "feature hash" in source
    assert "-> PyResult<String>" in source, "decode must be able to refuse"
    # The fixed "x0" token is gone, and no variable can bring it back.
    assert "decode_placeholder" not in source
    assert '"x0".to_string()' not in source
    assert "APEDP_ALLOW_PLACEHOLDER_KERNELS" not in source
    assert "if !allowed" not in source


def test_ssm_and_selective_scan_refuse_before_computing():
    source = (KERNEL / "ssm.rs").read_text(encoding="utf-8")
    assert source.count("PyNotImplementedError") == 2
    # Neither placeholder body survives: no pass-through copy, no 0.9 scaling.
    assert "out[i] = x_slice[i]" not in source
    assert "u_slice[i] * 0.9" not in source
    assert "APEDP_ALLOW_PLACEHOLDER_KERNELS" not in source
    # The refusal states what the placeholder used to do, so the reader knows why
    # nothing was returned rather than assuming a silent failure.
    assert "returned its input" in source
    assert "0.9*u" in source


def test_no_kernel_can_opt_in_to_a_fabricated_answer():
    """The flag must not exist anywhere in the native crate.

    Retiring it in one file and leaving it in another would leave a route to the fake
    output; this scans the whole crate rather than the two files that used to hold it.
    """
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(KERNEL.rglob("*.rs"))
        if "APEDP_ALLOW_PLACEHOLDER_KERNELS" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"placeholder opt-in still referenced in {offenders}"


def test_integer_relation_names_its_method_and_its_bound():
    source = (KERNEL / "rime" / "number_theory.rs").read_text(encoding="utf-8")
    assert ('pub const INTEGER_RELATION_METHOD: &str = "brute_force_branch_and_bound";'
            in source)
    assert "branch-and-bound search over" in source
    assert "lattice reduction" not in source.lower()
    assert "Searched, not proved" in source
    assert "n > 8" in source, "the coefficient search refuses more than eight inputs"


def test_native_extension_refuses_when_it_has_been_rebuilt():
    kernel = pytest.importorskip("apedp_kernel")
    if not hasattr(kernel, "LatentSpace"):
        pytest.skip("this build of the extension has no LatentSpace")
    try:
        kernel.LatentSpace(4).decode([0.0] * 4)
    except NotImplementedError:
        return
    pytest.skip("the built extension predates the refusal; rebuild it with maturin")
