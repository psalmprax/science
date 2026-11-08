"""The envelope may admit a candidate; it may never call one correct.

These tests pin the three ways an admitted candidate could be mistaken for a verified
one: a state called "verified", a gate set with no stated limits, and a claim with no
external measurement attached.
"""
from __future__ import annotations

import pytest

from apedp_dyn.epistemic.discovery_envelope import (
    DiscoveryEnvelope,
    DiscoveryState,
    ExternalVerificationRequest,
    GateResult,
    IncompleteDiscoveryClaimError,
    admit_for_external_verification,
)

REQUEST = ExternalVerificationRequest(
    domain="quantum",
    measurement="run the arm on a calibrated device and score it",
    acceptance="R * Q^2 * T below the reference",
    why_not_available_here="no quantum hardware in this environment",
)


def test_passing_every_gate_yields_admission_not_verification():
    envelope = admit_for_external_verification(
        candidate_id="dqi-arm-7",
        claim="686,014.8 Toffolis per shot at the pinned instance",
        gates=[
            GateResult("decoder_correct", True, "64/64 shots"),
            GateResult("schedule_gates_additive", True, "532,500 + 2,952 exact"),
        ],
        evidence={"toffolis_per_shot": 686_014.8, "target": 445_229},
        withheld_claims=("that the arm beats the reference on hardware",),
        external=REQUEST,
    )
    assert envelope.state is DiscoveryState.AWAITING_EXTERNAL_VERIFICATION
    assert envelope.admitted is True
    assert envelope.refused_by is None
    assert envelope.to_dict()["state"] == "awaiting_external_verification"


def test_there_is_no_verified_state_to_promote_into():
    assert {state.value for state in DiscoveryState} == {
        "refused",
        "awaiting_external_verification",
    }
    assert not any("verif" in state.value and "awaiting" not in state.value for state in DiscoveryState)


def test_a_failing_gate_refuses_the_candidate_and_names_it():
    envelope = admit_for_external_verification(
        candidate_id="c-1",
        claim="binds with Kd below 250 nM",
        gates=[
            GateResult("valence", True),
            GateResult("retrosynthesis", False, "SAScore 4.1 exceeds 3.5"),
        ],
        evidence={"sa_score": 4.1},
        withheld_claims=("that the compound is synthesizable at scale",),
        external=REQUEST,
    )
    assert envelope.state is DiscoveryState.REFUSED
    assert envelope.admitted is False
    assert envelope.refused_by == "retrosynthesis"


def test_a_claim_without_stated_limits_is_refused():
    with pytest.raises(IncompleteDiscoveryClaimError):
        admit_for_external_verification(
            candidate_id="c-2",
            claim="this is a drug",
            gates=[GateResult("valence", True)],
            evidence={},
            withheld_claims=(),
            external=REQUEST,
        )


def test_a_claim_with_no_external_measurement_is_refused():
    with pytest.raises(IncompleteDiscoveryClaimError):
        admit_for_external_verification(
            candidate_id="c-3",
            claim="this is a drug",
            gates=[GateResult("valence", True)],
            evidence={},
            withheld_claims=("nothing about binding",),
            external=None,  # type: ignore[arg-type]
        )


def test_a_claim_with_no_gate_is_refused():
    with pytest.raises(IncompleteDiscoveryClaimError):
        admit_for_external_verification(
            candidate_id="c-4",
            claim="this is a drug",
            gates=[],
            evidence={},
            withheld_claims=("nothing about binding",),
            external=REQUEST,
        )


@pytest.mark.parametrize(
    "field",
    ["domain", "measurement", "acceptance", "why_not_available_here"],
)
def test_the_external_request_must_be_fully_stated(field):
    kwargs = {
        "domain": "chemistry",
        "measurement": "assay the compound",
        "acceptance": "Kd below 250 nM",
        "why_not_available_here": "no wet lab",
    }
    kwargs[field] = "   "
    with pytest.raises(IncompleteDiscoveryClaimError):
        ExternalVerificationRequest(**kwargs)


def test_a_real_track_result_can_be_admitted_with_its_measured_numbers():
    from apedp_dyn.quantum_compilation.gf2mul_llm_adjustable import MMLMTrackA

    result = MMLMTrackA().process_with_tensor(1234, second_operand=2345)
    envelope = admit_for_external_verification(
        candidate_id="gf2-track-a-1234x2345",
        claim=f"product {result['product']} equals the field product",
        gates=[GateResult("bilinear_exact", result["exact"], f"rank {result['tensor_rank']}")],
        evidence={
            "product": result["product"],
            "expected_product": result["expected_product"],
            "bilinear_products": result["bilinear_products"],
        },
        withheld_claims=("that the decomposition has minimum rank",),
        external=ExternalVerificationRequest(
            domain="quantum",
            measurement="compile the product and execute it on hardware",
            acceptance="the measured amplitude distribution matches the field product",
            why_not_available_here="no device; simulators only",
        ),
    )
    assert isinstance(envelope, DiscoveryEnvelope)
    assert envelope.admitted is True
    assert envelope.evidence["product"] == envelope.evidence["expected_product"]
