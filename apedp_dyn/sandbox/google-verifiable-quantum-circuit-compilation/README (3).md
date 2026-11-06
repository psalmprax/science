# APEDP-Q Official Receipt Runner (fail-closed orchestrator)

This package is a repo-side **orchestration harness**, not a quantum circuit generator and not a competition verifier. It runs the builders and the unchanged official verifiers configured by the repository owner, then preserves the official JSON receipts, command logs, artifact hashes, protected-input hashes, and a consolidated execution manifest.

## Why it exists

The V4 candidate results should only reach the experiment ledger after all three exact candidates have current official receipts. The current chat execution environment does not contain `kickmix`, the official competition bundle, the user's live checkout, or the frozen chemistry input, so it cannot honestly generate those receipts itself. Run this package from the actual project environment.

## Configure

1. Copy `config/receipt_plan.template.json` to a local path in the project.
2. Set the absolute repository and competition roots.
3. Replace every `PATH/TO/...` command/path with the real build and unchanged verifier commands from the checkout. The current template intentionally uses placeholders and will not run successfully until configured.
4. Put the *receipt-bound* frozen input hash in `protected_files`. Resolve the two previously reported chemistry NPZ hashes first; do not choose one by guess.
5. For DFTHCSOS, list every generated angle circuit in `artifact_paths`, not merely `angle_k_0.kmb`; configure receipt checks using the actual verifier schema. Ensure the signed 4-bank builder checks `bit9 == bit8` for every real word, validates all 25 circuits, and includes measured peak width and Toffoli counts.
6. Confirm the DQI builder really contains the Gluzdov V6 architecture before using a V6 receipt plan. The summarized readiness list shared in the handoff did not list a DQI V6 module by filename, so this is a source-level preflight item.

Commands are argv arrays, not shell strings. Supported placeholders are `{repo_root}`, `{competition_root}`, `{run_dir}`, `{receipt_path}`, and `{component_dir}`. A verifier command must write its fresh official receipt to `{receipt_path}`. Each run gets a unique output directory.

## Run

```bash
python scripts/run_official_receipts.py --plan /path/to/receipt_plan.json
```

Or run one component while debugging:

```bash
python scripts/run_official_receipts.py --plan /path/to/receipt_plan.json --only dfthcsos
```

The runner refuses to continue when protected input hashes differ, a builder or verifier fails, expected artifacts are missing, a verifier does not emit a JSON receipt, or configured receipt checks fail. It does not promote to the experiment ledger. A combined portfolio summary is emitted only when all three components pass in the same run and protected files remain unchanged.

## Limitations

- This script cannot establish the authenticity of a verifier executable; the user must pin/check the official verifier source and its checksum before running.
- The template's builder/verifier command lines are placeholders, not claims about your project's CLI.
- The three benchmark receipt schemas may differ by competition version. Configure field checks against the exact output schema instead of assuming universal field names.
- A passing receipt is eligibility for human/ledger review, not automatic promotion. Keep the baseline receipt immutable.
