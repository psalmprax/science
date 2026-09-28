"""Generator for the APEDP-N10 external benchmark (frozen spec v1.0).

Implements the pasted N01-N10 specification exactly:
- N = 160 uniformly spaced in-domain points per case, PCG64 noise, master
  seed 20260928, case seed = master + case_number (N01 -> 1, ..., N10 -> 10).
- Noise applied BEFORE the train/validation split; split indices from the
  case-seeded PCG64 permutation (112 train / 48 validation).
- OOD grids are independent linspace samples storing CLEAN truth values
  (near=48, far=48 points per case; sizes frozen here — the spec fixes
  regions, not counts).
- Layout mirrors the 60-problem corpus: benchmark_n10/{public,hidden},
  manifest.json, checksums.sha256, generation_seed.txt.

Blinding: public/ carries only domains, sizes, noise-model description and
CSVs. All truth (formula, family, params, asymptotes, poles, equivalence
rules) lives in hidden/. The equations below are the oracle-author role;
discovery processes must never read hidden/ (same split discipline as the
60-corpus, §6.2-style).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MASTER_SEED = 20260928
N_TOTAL = 160
N_TRAIN = 112
N_VAL = 48
N_NEAR = 48
N_FAR = 48

CASES: list[dict] = [
    dict(pid="N01", num=1, name="exp-growth",
         formula="2.4*exp(0.37*x)", family="exponential",
         domain=(0.0, 5.0), noise_rel=0.01,
         near=[(-1.0, 0.0)], far=[(-3.0, -1.0), (5.0, 8.0)],
         fn=lambda x: 2.4 * np.exp(0.37 * x)),
    dict(pid="N02", num=2, name="log-saturation",
         formula="1.8*log(1+2.5*x)", family="logarithmic",
         domain=(0.0, 4.0), noise_rel=0.01,
         near=[(-0.35, 0.0)], far=[(-0.39, -0.35), (4.0, 8.0)],
         fn=lambda x: 1.8 * np.log(1 + 2.5 * x),
         meta={"physical_domain": "x > -0.4"}),
    dict(pid="N03", num=3, name="rational-saturation",
         formula="4.2*x/(1+0.8*x)", family="saturating_rational",
         domain=(0.0, 8.0), noise_rel=0.01,
         near=[(-1.0, 0.0)], far=[(-1.2, -1.0), (8.0, 16.0)],
         fn=lambda x: 4.2 * x / (1 + 0.8 * x),
         meta={"singularity": -1.25, "asymptote": 5.25}),
    dict(pid="N04", num=4, name="phase-shifted-sinusoid",
         formula="3*sin(1.7*x+0.4)", family="sinusoidal",
         domain=(-4.0, 4.0), noise_rel=0.01, noise_ref="amplitude",
         near=[(-6.0, -4.0), (4.0, 6.0)], far=[(-12.0, -6.0), (6.0, 12.0)],
         fn=lambda x: 3 * np.sin(1.7 * x + 0.4),
         meta={"equivalence": ["A*sin(k*x+phi)", "cosine forms"]}),
    dict(pid="N05", num=5, name="squared-harmonic",
         formula="2.5*cos(0.9*x)**2", family="harmonic_squared",
         domain=(-5.0, 5.0), noise_rel=0.005,
         near=[(-7.0, -5.0), (5.0, 7.0)], far=[(-12.0, -7.0), (7.0, 12.0)],
         fn=lambda x: 2.5 * np.cos(0.9 * x) ** 2,
         meta={"equivalence": ["cos(z)**2 == (1+cos(2*z))/2"],
               "rule": "polynomial surrogate with low val MSE is NOT family recovery"}),
    dict(pid="N06", num=6, name="fractional-power-composite",
         formula="5/(1+0.6*x**2)**1.5", family="inverse_fractional_power",
         domain=(-4.0, 4.0), noise_rel=0.01,
         near=[(-6.0, -4.0), (4.0, 6.0)], far=[(-10.0, -6.0), (6.0, 10.0)],
         fn=lambda x: 5 / (1 + 0.6 * x ** 2) ** 1.5,
         meta={"asymptote": 0.0,
               "rule": "polynomial-divergent candidates fail OOD even at small val MSE"}),
    dict(pid="N07", num=7, name="shifted-power-law",
         formula="0.7*x**1.8+1.2", family="shifted_power_law",
         domain=(0.2, 5.0), noise_rel=0.01,
         near=[(0.05, 0.2)], far=[(0.001, 0.05), (5.0, 10.0)],
         fn=lambda x: 0.7 * x ** 1.8 + 1.2,
         meta={"rule": "pure power law without +1.2 offset is NOT exact recovery"}),
    dict(pid="N08", num=8, name="exponential-saturation",
         formula="6*(1-exp(-0.45*x))", family="exponential_saturation",
         domain=(0.0, 6.0), noise_rel=0.01,
         near=[(-1.0, 0.0)], far=[(-3.0, -1.0), (6.0, 12.0)],
         fn=lambda x: 6 * (1 - np.exp(-0.45 * x)),
         meta={"asymptote": 6.0}),
    dict(pid="N09", num=9, name="pole-boundary",
         formula="2.5/(x+0.7)+0.4", family="rational_with_pole",
         domain=(0.2, 5.0), noise_rel=0.01,
         near=[(-0.5, 0.2)], far=[(-0.65, -0.5), (-0.69, -0.65), (5.0, 10.0)],
         fn=lambda x: 2.5 / (x + 0.7) + 0.4,
         meta={"pole": -0.7,
               "rule": "extra pole inside a scored region = domain violation"}),
    dict(pid="N10", num=10, name="power-times-exponential",
         formula="1.5*x**0.6*exp(-0.15*x)", family="power_exponential",
         domain=(0.2, 8.0), noise_rel=0.01,
         near=[(0.02, 0.2)], far=[(0.001, 0.02), (8.0, 16.0)],
         fn=lambda x: 1.5 * x ** 0.6 * np.exp(-0.15 * x),
         meta={"asymptote": 0.0}),
]


def _grid(regions: list[tuple[float, float]], n: int) -> np.ndarray:
    parts, counts = [], []
    widths = [hi - lo for lo, hi in regions]
    total = sum(widths)
    alloc, rem = [], n
    for i, w in enumerate(widths):
        c = n * w / total if i < len(widths) - 1 else rem
        c = max(1, int(round(c)))
        alloc.append(c)
        rem -= c
    for (lo, hi), c in zip(regions, alloc):
        parts.append(np.linspace(lo, hi, c))
    return np.concatenate(parts)[:n]


def generate(root: Path) -> dict:
    pub, hid = root / "public", root / "hidden"
    manifest = {"benchmark": "APEDP-N10", "version": "1.0",
                "master_seed": MASTER_SEED, "status": "FROZEN",
                "problems": []}
    for case in CASES:
        rng = np.random.default_rng(MASTER_SEED + case["num"])
        xs = np.linspace(*case["domain"], N_TOTAL)
        clean = np.asarray(case["fn"](xs), dtype=float)
        sigma = case["noise_rel"] * (3.0 if case.get("noise_ref") == "amplitude"
                                     else float(np.max(np.abs(clean))))
        noisy = clean + rng.normal(0.0, sigma, N_TOTAL)
        perm = rng.permutation(N_TOTAL)
        tr, va = np.sort(perm[:N_TRAIN]), np.sort(perm[N_TRAIN:N_TRAIN + N_VAL])
        xn, xf = _grid(case["near"], N_NEAR), _grid(case["far"], N_FAR)
        (pub / case["pid"]).mkdir(parents=True, exist_ok=True)
        (hid / case["pid"]).mkdir(parents=True, exist_ok=True)
        np.savetxt(pub / case["pid"] / "train.csv",
                   np.column_stack([xs[tr], noisy[tr]]),
                   delimiter=",", header="x,y", comments="")
        np.savetxt(pub / case["pid"] / "validation.csv",
                   np.column_stack([xs[va], noisy[va]]),
                   delimiter=",", header="x,y", comments="")
        np.savetxt(pub / case["pid"] / "near_ood.csv",
                   np.column_stack([xn, case["fn"](xn)]),
                   delimiter=",", header="x,y", comments="")
        np.savetxt(pub / case["pid"] / "far_ood.csv",
                   np.column_stack([xf, case["fn"](xf)]),
                   delimiter=",", header="x,y", comments="")
        (pub / case["pid"] / "problem_spec.json").write_text(json.dumps({
            "problem_id": case["pid"], "block": "N", "name": case["name"],
            "input_variables": ["x"], "target_variable": "y",
            "domain": list(case["domain"]),
            "noise_model": f"additive Gaussian sigma={sigma:.6g} "
                           f"(rel={case['noise_rel']})",
            "dataset_sizes": {"train": N_TRAIN, "validation": N_VAL,
                              "near_ood": N_NEAR, "far_ood": N_FAR},
            "near_ood_regions": case["near"], "far_ood_regions": case["far"],
        }, indent=2))
        (hid / case["pid"] / "ground_truth.json").write_text(json.dumps({
            "problem_id": case["pid"], "block": "N",
            "ground_truth_formula": case["formula"],
            "canonical_structural_family": case["family"],
            "oracle_epistemic_class": "C1", "expected_outcome": "RECOVER",
            "case_seed": MASTER_SEED + case["num"],
            "noise_sigma": sigma,
            "meta": case.get("meta", {}),
        }, indent=2))
        manifest["problems"].append({"problem_id": case["pid"],
                                     "block": "N", "name": case["name"]})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "generation_seed.txt").write_text(str(MASTER_SEED) + "\n")
    sums = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "checksums.sha256":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  "
                        f"{p.relative_to(root)}")
    (root / "checksums.sha256").write_text("\n".join(sums) + "\n")
    return manifest


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmark_n10")
    m = generate(out)
    print(f"generated {len(m['problems'])} problems -> {out}")
