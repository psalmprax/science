"""Transfer scope: where the evidence holds, as a first-class record.

Orthogonal to the epistemic verdict by design. The verdict answers
"can these structures be told apart HERE"; transfer scope answers
"does the ranking survive THERE". A MULTI verdict with
CROSS_DOMAIN_FAILED transfer is not an INSUFFICIENT verdict -- the
interpolation evidence stands, and must not be rewritten by the
extrapolation failure. Never auto-downgrade on this module's output.

Statuses:
    NOT_TESTED:               single-regime evidence only.
    IN_DOMAIN_SUPPORTED:       leadership holds on an independent
                              in-domain slice.
    CROSS_DOMAIN_SUPPORTED:    leadership holds across the regime boundary.
    CROSS_DOMAIN_FAILED:       leadership reverses or error blows up
                              (default factor 5x) across the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class TransferStatus(str):
    NOT_TESTED = "not_tested"
    IN_DOMAIN_SUPPORTED = "in_domain_supported"
    CROSS_DOMAIN_SUPPORTED = "cross_domain_supported"
    CROSS_DOMAIN_FAILED = "cross_domain_failed"


@dataclass(frozen=True)
class TransferScope:
    """Structured transfer record for one ranked candidate pair.

    All RMSEs are holdout (never train) numbers. Regimes are caller-defined
    index ranges; the module does not interpret physics, only ranking
    stability and error blowup across explicitly declared scopes.
    """
    candidate_id: str
    rival_id: str
    interp_rmse_best: float
    interp_rmse_rival: float
    extra_rmse_best: float
    extra_rmse_rival: float
    blowup_factor: float = 5.0
    leadership_reversed: bool = field(init=False)
    error_blowup: float = field(init=False)
    status: str = field(init=False)

    def __post_init__(self):
        # NOTE: dataclass is frozen; use object.__setattr__ for derived fields.
        object.__setattr__(self, "leadership_reversed",
                           bool(self.extra_rmse_rival < self.extra_rmse_best))
        blow = (self.extra_rmse_best / self.interp_rmse_best
                if self.interp_rmse_best > 0 else float("inf"))
        object.__setattr__(self, "error_blowup", float(blow))
        if self.leadership_reversed or blow > self.blowup_factor:
            object.__setattr__(self, "status", TransferStatus.CROSS_DOMAIN_FAILED)
        else:
            object.__setattr__(self, "status", TransferStatus.CROSS_DOMAIN_SUPPORTED)

    def record(self) -> dict:
        """Serialisable form for artifacts and the structured verdict block."""
        return {
            "structure": {"candidate": self.candidate_id, "rival": self.rival_id},
            "evidence": {"scope": "interpolation_and_extrapolation"},
            "transfer": {"status": self.status,
                         "leadership_reversed": self.leadership_reversed,
                         "error_blowup": round(self.error_blowup, 2),
                         "interp_rmse": [self.interp_rmse_best, self.interp_rmse_rival],
                         "extra_rmse": [self.extra_rmse_best, self.extra_rmse_rival]},
            "promotion": {"status": "NOT_PROMOTED_BY_THIS_MODULE",
                          "reason": "transfer scope never promotes; "
                                    "see epistemic verdict"},
        }
