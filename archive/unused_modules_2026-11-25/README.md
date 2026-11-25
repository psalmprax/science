# Modules moved out of the package on 2026-11-25

By the owner's decision of 2026-11-25. These 14 files were in `apedp_dyn/` and were, on that
date, named by no test, imported by no tracked code, and mentioned in no document, record,
workflow or configuration of the repository other than the module table. They are moved byte for byte, keep their history
(`git log --follow`) and their relative paths:

    archive/unused_modules_2026-11-25/apedp_dyn/<where it was>

Nothing here is imported, tested or linted any more. To bring one back, move it to the same path
under the repository root (`git mv`), or revert the commit that made this folder.

## What was not moved, and why

The outside audit counted 40 modules that no test names and nothing imports. Twenty-six stayed
where they are:

- **22 `apedp_dyn/sandbox/cpmd_*.py` files** are the code of record of the CPMD preregistrations
  (`docs/CPMD_PREREGISTRATION_*.md` and others name them; one is named by a plan that has not been
  run, `preregistration/PREREG-LAW-AUDIT-FINANCE-MORE-2026-11.md`). They are kept for replication,
  untouched, and `docs/DOCUMENTATION.md` says so in each one's row.
- **`apedp_dyn/ui/__main__.py` and `apedp_dyn/ui/streamlit_app.py`** are how the interface is
  started (`python -m apedp_dyn.ui`); nothing imports an entry point.
- **`apedp_dyn/experiments/`** (two files) produced `outputs/d2a_ab_report.json`, which two tests
  read.
- **`apedp_dyn/quantum_compilation/multi_structure_search.py`** is listed in
  `docs/VQC_PHASE0_CAPABILITY_INVENTORY.md`.
