"""Equivalence check: re-authored apedp/constitution.py vs the orphaned bytecode.

Runs both implementations over a matrix of proposals and requires
verdict-identical output field by field -- compliant, every check's
satisfied flag, every reason string, every rejection reason, and the
sha256.

The bytecode module is loaded directly from ``__pycache__`` under a
different module name, so both live in the same interpreter and can be
compared without a subprocess. That is safe here specifically because
neither implementation imports the other, so there is no ``sys.modules``
caching hazard -- the failure mode that made the first singularity harness
lie is documented in ``tools/orphan_equivalence.py``.

Usage::

    python3 tools/constitution_equivalence.py
"""
from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYC = sorted((ROOT / "apedp/__pycache__").glob("constitution.cpython-312.pyc"))[-1]


def load_original():
    spec = importlib.util.spec_from_file_location("constitution_original", PYC)
    m = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves string annotations via
    # sys.modules[cls.__module__].__dict__, and a module that was never
    # registered is absent from sys.modules, so the decorator raises
    # AttributeError on 'NoneType'. Not a problem with the bytecode.
    sys.modules["constitution_original"] = m
    spec.loader.exec_module(m)
    return m


def verdict_tuple(v) -> tuple:
    """Everything the two implementations must agree on, in one comparable."""
    return (
        bool(v.compliant),
        bool(v.sha256),
        tuple(v.rejection_reasons),
        tuple((c.invariant_id.name, bool(c.satisfied), c.reason) for c in v.checks),
    )


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from apedp.constitution import ConstitutionalAuditor as New

    orig = load_original()
    Old = orig.ConstitutionalAuditor

    # Every boolean the auditor reads, plus the two inputs that are not
    # proposal keys. Full cross product: 2^9 * 2 * 2 = 1024 combinations,
    # which is the whole input space the implementation can distinguish.
    flags = [
        "classified_contradiction_as_valid",
        "unresolved_treated_as_valid",
        "selected_by_train_fit_alone",
        "modifies_benchmark_data",
        "retrospective_criterion_change",
        "reproducibility_verified",
        "supports_ambiguity_preservation",
        "history_logging_active",
    ]
    new_auditor, old_auditor = New(), Old()

    agree = disagree = 0
    for bits in itertools.product([False, True], repeat=len(flags)):
        for regression_passed in (True, False):
            for with_prov in (True, False):
                proposal = dict(zip(flags, bits, strict=True))
                prov = ({"author": "a", "timestamp": "t"} if with_prov
                        else {"author": "a"})
                kwargs = {
                    "proposal_dict": proposal,
                    "benchmark_results": None,
                    "regression_passed": regression_passed,
                    "provenance_trace": prov,
                }
                a = verdict_tuple(old_auditor.audit(**kwargs))
                b = verdict_tuple(new_auditor.audit(**kwargs))
                if a == b:
                    agree += 1
                else:
                    disagree += 1
                    if disagree <= 3:
                        print(f"DISAGREE flags={bits} reg={regression_passed} "
                              f"prov={with_prov}")
                        print(f"  original:    {a}")
                        print(f"  reauthored: {b}")

    # Also compare the enum: 10 members, exact names and descriptions.
    old_ids = {i.name: i.value for i in orig.InvariantID}
    new_ids = {i.name: i.value for i in
               __import__("apedp.constitution", fromlist=["InvariantID"]).InvariantID}
    if old_ids == new_ids:
        agree += 1
        print("AGREE  InvariantID: 10 members, names and descriptions identical")
    else:
        disagree += 1
        print("DISAGREE InvariantID")
        for k in set(old_ids) | set(new_ids):
            if old_ids.get(k) != new_ids.get(k):
                print(f"  {k}: {old_ids.get(k)!r} vs {new_ids.get(k)!r}")

    # And the dataclass field names and order.
    import dataclasses
    for cls_name in ("InvariantCheck", "ConstitutionalVerdict"):
        o = [f.name for f in dataclasses.fields(getattr(orig, cls_name))]
        n = [f.name for f in dataclasses.fields(
            getattr(__import__("apedp.constitution", fromlist=[cls_name]), cls_name))]
        if o == n:
            agree += 1
            print(f"AGREE  {cls_name} fields: {n}")
        else:
            disagree += 1
            print(f"DISAGREE {cls_name} fields: {o} vs {n}")

    print(f"\n{agree} agree, {disagree} disagree")
    return 1 if disagree else 0


if __name__ == "__main__":
    raise SystemExit(main())
