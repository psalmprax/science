"""Diagnostic abstention: machine-readable WHY behind every non-promotion.

Composes the already-existing evidence contracts (verdict status,
epistemic attribution, pair/full contract, transfer scope, reason
strings) into one structured diagnostic block:

    epistemic_result:
      status: MULTI_HYPOTHESIS | INSUFFICIENT_EVIDENCE | ...
      diagnosis: {primary, evidence, separation}
      transfer: {...} | NOT_ESTABLISHED
      claim_boundary: {supported: [...], unsupported: [...]}

This module creates no new epistemology; it makes the existing one
queryable. Abstention + reason + evidence + scope, never bare abstention.
"""
from __future__ import annotations


def diagnose(adjudication, transfer=None) -> dict:
    """Compose a diagnostic block from an Adjudication (+ optional TransferScope)."""
    status = getattr(adjudication.status, "value", str(adjudication.status))
    attribution = str(getattr(adjudication, "epistemic_attribution", "unknown"))
    promoted = getattr(adjudication, "promoted", None)
    tau = getattr(adjudication, "tau_epi", None)

    if promoted is not None:
        diagnosis = {"primary": "DECISIVE_EVIDENTIAL_SUPPORT",
                     "evidence": [getattr(promoted.candidate, "candidate_id", "?")],
                     "separation": {"metric": "validation_rmse",
                                    "delta": getattr(adjudication, "delta_rmse", None),
                                    "tau": tau, "insufficient": False}}
        boundary = {"supported": ["observed_regime"],
                    "unsupported": ["global_validity"]}
    elif status == "multi_hypothesis":
        cands = [getattr(s.candidate, "candidate_id", "?")
                 for s in (getattr(adjudication, "ranking", None) or [])[:2]]
        diagnosis = {"primary": attribution,
                     "evidence": cands,
                     "separation": {"metric": "validation_rmse",
                                    "delta": getattr(adjudication, "delta_rmse", None),
                                    "tau": tau, "insufficient": True}}
        boundary = {"supported": ["observed_regime"],
                    "unsupported": ["unique_structure", "global_validity"]}
    else:
        diagnosis = {"primary": attribution, "evidence": [],
                     "separation": {"metric": "validation_rmse",
                                    "delta": None, "tau": tau, "insufficient": True}}
        boundary = {"supported": [],
                    "unsupported": ["observed_regime", "unique_structure",
                                    "global_validity"]}
    transfer_block = transfer.record() if transfer is not None else "NOT_ESTABLISHED"
    return {"status": status, "diagnosis": diagnosis,
            "transfer": transfer_block, "claim_boundary": boundary,
            "contract": str(getattr(adjudication, "contract", "unknown")),
            "reason": getattr(adjudication, "reason", "")}


def full_state(adjudication, transfer=None, stability=None) -> dict:
    """Unified epistemic state: discovery + identification + transfer + promotion.

    Merges the diagnostic block with an optional discovery-stability report
    (see discovery_stability.stability_report). The result carries the full
    inequality chain in one record: fit != recovery != identification !=
    transfer != promotion, plus the stability orthogonal (recurrence,
    entropy, parameter dispersion, variance split). Machine-readable;
    no new epistemology beyond composition.
    """
    base = diagnose(adjudication, transfer=transfer)
    if stability is not None:
        base["discovery"] = dict(stability)
        dist = (stability.get("distribution") or {})
        top = max(dist.values()) if dist else 0.0
        base["discovery"]["structural_recurrence"] = float(top)
    else:
        base["discovery"] = "NOT_ASSESSED"
    return base
