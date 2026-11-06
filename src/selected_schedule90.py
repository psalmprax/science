# SPDX-License-Identifier: Apache-2.0
# Kozykorpesh signed-curvature lifetime schedule, V3/run361677445.
# https://www.kaggle.com/code/kozykappa/quantum-ai-lb-score-6-68108?scriptVersionId=361677445
# Caller supplies the explicit public Kozy coefficient table, already retained
# in our attributed signed_digit_invsqrt_v3 module. No Toolbox data is required.
import hashlib
import io
import json
from pathlib import Path
import fusion_schedule90 as F


SELECTED_OPTIONS = dict(delay_high=True, gray_checkpoint=True,
                        curve_count=2, late_coarse=0)


def build_selected(cfg, tables):
    """Return the raw selected public schedule and its provenance report.

    Apply the existing exact-domain compiler and fixed calibrated rotations
    separately; their canonical source/output hashes bind those transformations.
    """
    if (cfg.f2, cfg.f1, cfg.fyr, cfg.gy, sorted(cfg.drop2)) != (8, 19, 27, 1, [5]):
        raise ValueError("This schedule is pinned to the explicit 256-row Kozy model")
    choices_path = Path(__file__).with_name("boolean_choices90.json")
    choices = json.loads(choices_path.read_text())
    circuit, report = F.build(cfg, tables, boolean_choices=choices, **SELECTED_OPTIONS)
    stream = io.BytesIO()
    circuit.to_file(stream, format="kmb")
    report.update(upstream_version=3, upstream_run_id=361677445,
                  method="Gray curvature checkpoint with two early erased bits and deferred joint ANF correction",
                  boolean_choices_sha256=hashlib.sha256(choices_path.read_bytes()).hexdigest(),
                  raw_circuit_sha256=hashlib.sha256(stream.getvalue()).hexdigest(),
                  table_source="Caller-supplied explicit public Kozy integer triples",
                  selected_options=SELECTED_OPTIONS,
                  verification_status="Requires exact-domain compilation, phase certificate and full official check")
    return circuit, report
