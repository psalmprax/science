"""Probe-envelope constants shared by the falsification axes."""
from __future__ import annotations

#: Samples per symbol used by the structural-invariant probes. Every invariant is
#: decided by evaluating the model across the declared envelope, never at the
#: declared bounds alone and never from the symbols' names.
INVARIANT_SAMPLES = 33


#: Ratio of the largest |f| over the envelope to the median |f| above which the
#: growth is treated as a divergence toward the envelope edge (an uncensored
#: curvature singularity) rather than a smooth physical law.
ENVELOPE_DIVERGENCE_RATIO = 1e3
