# self_validation_models/

Reserved scratch directory for manual `self_validation.py` exports.

The automated harness (`self_validation.py`, `tests/test_self_validation.py`)
uses temporary directories, so this folder stays empty in a clean checkout
(`.gitkeep` only). Drop hand-checked model JSON here when you need to keep an
artifact out of the temp flow; nothing in `apedp/` or `tests/` reads it.
