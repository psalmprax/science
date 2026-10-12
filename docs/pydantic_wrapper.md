# Python `pydantic` wrappers for RIME v0.1

RIME exposes a JSON‑based API via the PyO3 bridge.  The following example shows how a Python client can deserialize the `SearchPacket` returned by `rime_search_json` into typed `pydantic` models.

```python
import json
import pydantic
from typing import List, Optional

# --- Protobuf‑style schemas (mirror the Rust Value structure) ---
class ScoreInfo(pydantic.BaseModel):
    equation: str
    rewrite: str
    score: float
    mse: float

class SearchPacket(pydantic.BaseModel):
    proposals: List[ScoreInfo]
    accepted: bool

# --- Calling the Rust bridge ---
# The Rust PyO3 module `apedp_kernel` provides `rime_search_json`:
#    fn rime_search_json(data, target, config_json, variable_names=None) -> PyResult<String>
#
# Example call (assuming the module is importable):
#
# ```python
# import apedp_kernel
# data = ...  # JSON‑encoded design matrix
# target = ...  # JSON‑encoded target vector
# config = json.dumps({"budget": 10, "beam": 5, "lambda_c": 1.5, "ridge": 0.0, "seed": 42})
# raw = apedp_kernel.rime_search_json(data, target, config)
# packet = SearchPacket.model_validate_json(raw)
# print("Accepted:", packet.accepted)
# for prop in packet.proposals:
#     print(" - equation:", prop.equation, "rewrite:", prop.rewrite,
#           "score:", prop.score, "mse:", prop.mse)
# ```

# --- referee classification (mirroring the Rust enum) ---
class RefereeOutcome(pydantic.BaseModel):
    value: str  # one of "NO_RECOVERY", "PREDICTIVE_RECOVERY", "MULTI_HYPOTHESIS", "UNIQUE_IDENTIFICATION"

def classify_packet(pkt: SearchPacket) -> RefereeOutcome:
    n_positive = sum(1 for p in pkt.proposals if p.score > 0.0)
    if not pkt.accepted or n_positive == 0:
        return RefereeOutcome(value="NO_RECOVERY")
    if n_positive == 1:
        return RefereeOutcome(value="PREDICTIVE_RECOVERY")
    return RefereeOutcome(value="MULTI_HYPOTHESIS")
```

## Notes

* The Rust types `SearchPacket`, `ScoreInfo` etc. are already JSON‑serialisable via `serde_json`.  The `pydantic` models above merely give Python code a typed view.
* The `referee` module in Rust provides `classify(packet) → RefereeOutcome`; the Python counterpart above reproduces the same logic.
* To add a new outcome (e.g. `UNIQUE_IDENTIFICATION`) you would extend both the Rust `RefereeOutcome` enum and the Python `RefereeOutcome` model, and update the classification logic in both languages.

---

**Next steps**

* Add the `pydantic` dependency to your Python project (`pip install pydantic`).
* Import the `apedp_kernel` module (built with `maturin develop` or `python -m pip install -e .` from the Rust crate).
* Call `rime_search_json` with your design‑matrix and target data, then validate the returned JSON with the `SearchPacket` model.
* Use `classify_packet` to obtain the referee taxonomy outcome without touching the Rust code.