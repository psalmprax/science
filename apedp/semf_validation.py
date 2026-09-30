"""SEMF validation against AME2020 (Future Work item 3).

Parses the AME2020 mass table (mass_1.mas20.txt, Huang et al., Chin. Phys. C
45 030002; Wang et al., 030003) and evaluates the Semi-Empirical Mass Formula
in two modes:

1. literature coefficients (a_v=15.8, a_s=18.3, a_c=0.714, a_a=23.2 MeV),
2. least-squares fit of the four linear coefficients on the same data
   (pairing column fixed at +/-12/sqrt(A)).

Only experimental entries are used (lines containing '#' are extrapolated
estimates in the mas20 format and are excluded). Binding energies come from
the table's binding/A column (keV), converted to MeV.

Residuals at magic numbers (2, 8, 20, 28, 50, 82, 126) are reported
separately: SEMF has no shell physics, so magic nuclei must show larger
residuals -- a built-in honesty check on the model's limits.

Data source (download once, not vendored):
    https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt
    -> pass path explicitly or via the APEDP_AME20_PATH environment variable.

Reference (cite the papers, not the file):
    Huang et al., CPC 45 030002 (2021); Wang et al., CPC 45 030003 (2021).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

MAGIC = (2, 8, 20, 28, 50, 82, 126)
LIT_COEFFS = {"a_v": 15.8, "a_s": 18.3, "a_c": 0.714, "a_a": 23.2}  # MeV


@dataclass
class AmeDataset:
    """Experimental binding energies parsed from mass_1.mas20.txt."""
    Z: np.ndarray
    N: np.ndarray
    A: np.ndarray
    BE_per_nucleon: np.ndarray  # MeV
    n_skipped_extrapolated: int
    n_skipped_missing: int

    def __len__(self) -> int:
        return int(self.A.size)


def find_ame_file(explicit: str | None = None) -> str:
    """Locate the AME2020 mass file without vendoring it into the repo.

    Search order: explicit argument, ``APEDP_AME20_PATH``, the repository's
    ``data/`` directory, then the standard temp directory. The previous entry
    was a hardcoded ``/tmp/opencode/...`` path, which only existed on the
    machine that generated it and made the lookup fail everywhere else.
    """
    import tempfile

    candidates = [
        explicit,
        os.environ.get("APEDP_AME20_PATH"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mass_1.mas20.txt"),
        os.path.join("data", "mass_1.mas20.txt"),
        os.path.join(tempfile.gettempdir(), "mass_1.mas20.txt"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "mass_1.mas20.txt not found; download "
        "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt "
        "and pass its path or set APEDP_AME20_PATH"
    )


def parse_mas20(path: str) -> AmeDataset:
    """Parse experimental binding/A values from a mas20 mass table.

    Fixed columns (0-indexed): N [4:9], Z [9:14], A [14:19],
    binding/A keV [54:67]. Lines with '#' are extrapolated estimates.
    """
    Z, N, A, BE = [], [], [], []
    n_extrap, n_missing = 0, 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 67:
                continue
            try:
                n, z, a = int(line[4:9]), int(line[9:14]), int(line[14:19])
            except ValueError:
                continue
            if a <= 0 or z <= 0 or n < 0 or n + z != a:
                continue
            if "#" in line:
                n_extrap += 1
                continue
            raw = line[54:67].strip().replace("*", "")
            if not raw:
                n_missing += 1
                continue
            try:
                be_kev = float(raw)
            except ValueError:
                n_missing += 1
                continue
            if not np.isfinite(be_kev) or be_kev <= 0:
                n_missing += 1
                continue
            Z.append(z)
            N.append(n)
            A.append(a)
            BE.append(be_kev / 1000.0)  # keV -> MeV per nucleon
    return AmeDataset(
        Z=np.asarray(Z, dtype=float),
        N=np.asarray(N, dtype=float),
        A=np.asarray(A, dtype=float),
        BE_per_nucleon=np.asarray(BE, dtype=float),
        n_skipped_extrapolated=n_extrap,
        n_skipped_missing=n_missing,
    )


def design_columns(Z: np.ndarray, N: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Total-B design matrix [A, -A^2/3, -Z^2/A^1/3, -(A-2Z)^2/A, pair]."""
    pair = np.where((Z % 2 == 0) & (N % 2 == 0), 12.0 / np.sqrt(A),
            np.where((Z % 2 == 1) & (N % 2 == 1), -12.0 / np.sqrt(A), 0.0))
    return np.column_stack([
        A,
        -(A ** (2.0 / 3.0)),
        -(Z ** 2) / (A ** (1.0 / 3.0)),
        -((A - 2.0 * Z) ** 2) / A,
        pair,
    ])


def semf_per_nucleon(
    Z: np.ndarray, N: np.ndarray, A: np.ndarray, coeffs: dict
) -> np.ndarray:
    """SEMF binding energy per nucleon (MeV) for coefficient dict."""
    total: np.ndarray = design_columns(Z, N, A) @ np.array(
        [coeffs["a_v"], coeffs["a_s"], coeffs["a_c"], coeffs["a_a"], 1.0]
    )
    return np.asarray(total / A)


def fit_semf(data: AmeDataset, min_A: int = 16) -> tuple[dict, dict]:
    """Least-squares SEMF fit on nuclei with A >= min_A (light nuclei excluded).

    Returns (fitted_coeffs, report) with RMSE/R^2 on total B and per nucleon.
    """
    keep = data.A >= min_A
    X = design_columns(data.Z[keep], data.N[keep], data.A[keep])
    y_total = data.BE_per_nucleon[keep] * data.A[keep]
    coef, _residuals, _rank, _ = np.linalg.lstsq(X, y_total, rcond=None)
    names = ["a_v", "a_s", "a_c", "a_a", "pair_scale"]
    fitted = dict(zip(names, (float(c) for c in coef), strict=False))
    pred_total = X @ coef
    pred_nuc = pred_total / data.A[keep]
    obs_nuc = data.BE_per_nucleon[keep]
    rmse_total = float(np.sqrt(np.mean((y_total - pred_total) ** 2)))
    rmse_nuc = float(np.sqrt(np.mean((obs_nuc - pred_nuc) ** 2)))
    ss_res = float(np.sum((obs_nuc - pred_nuc) ** 2))
    ss_tot = float(np.sum((obs_nuc - obs_nuc.mean()) ** 2))
    return fitted, {
        "n_fit": int(keep.sum()),
        "rmse_total_MeV": rmse_total,
        "rmse_per_nucleon_MeV": rmse_nuc,
        "r_squared": 1.0 - ss_res / ss_tot,
    }


def magic_residual_report(
    data: AmeDataset, pred_nuc: np.ndarray, mask: np.ndarray | None = None
) -> dict:
    """Compare mean |residual| for magic vs non-magic nuclei (same mask)."""
    sel = np.ones(len(data), dtype=bool) if mask is None else mask
    resid = np.abs(data.BE_per_nucleon[sel] - pred_nuc)
    is_magic = np.array(
        [z in MAGIC or n in MAGIC for z, n in zip(data.Z[sel], data.N[sel], strict=False)]
    )
    return {
        "n_magic": int(is_magic.sum()),
        "n_other": int((~is_magic).sum()),
        "mean_abs_resid_magic_MeV": float(resid[is_magic].mean()),
        "mean_abs_resid_other_MeV": float(resid[~is_magic].mean()),
    }


def shell_columns(N: np.ndarray, Z: np.ndarray, width: float = 3.0,
                  per_magic: bool = False) -> np.ndarray:
    """Phenomenological shell-correction features (Strutinsky-style).

    Gaussian proximity of N (resp. Z) to each magic number; magic nuclei
    come out more bound, so fitted coefficients must be POSITIVE -- a
    built-in sign check on the physics. With per_magic=False the seven
    Gaussians are summed (2 features); per_magic=True keeps all 14.
    """
    if not width > 0:
        raise ValueError("width must be positive")
    N = np.asarray(N, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    M = np.asarray(MAGIC, dtype=float)
    gN = np.exp(-((N[:, None] - M[None, :]) / width) ** 2)
    gZ = np.exp(-((Z[:, None] - M[None, :]) / width) ** 2)
    if per_magic:
        return np.column_stack([gN, gZ])
    return np.column_stack([gN.sum(axis=1), gZ.sum(axis=1)])


def fit_semf_shell(data: AmeDataset, min_A: int = 16, width: float = 3.0,
                   per_magic: bool = False) -> tuple[dict, dict]:
    """SEMF + shell-correction least squares on total B (A >= min_A).

    Returns (fitted, report) with RMSE total/per-nucleon, BIC (total-B
    SSR), and the shell amplitudes for the sign check.
    """
    keep = data.A >= min_A
    Z, N, A = data.Z[keep], data.N[keep], data.A[keep]
    X = np.column_stack([design_columns(Z, N, A),
                         shell_columns(N, Z, width, per_magic)])
    y_total = data.BE_per_nucleon[keep] * A
    coef, *_ = np.linalg.lstsq(X, y_total, rcond=None)
    n, k = len(A), X.shape[1]
    pred_total = X @ coef
    ssr = float(np.sum((y_total - pred_total) ** 2))
    pred_nuc = pred_total / A
    obs_nuc = data.BE_per_nucleon[keep]
    base = ["a_v", "a_s", "a_c", "a_a", "pair_scale"]
    shell_names = ([f"shellN_{m}" for m in MAGIC] + [f"shellZ_{m}" for m in MAGIC]
                   if per_magic else ["shellN", "shellZ"])
    return dict(zip(base + shell_names, (float(c) for c in coef), strict=False)), {
        "n_fit": n, "n_params": k, "width": width, "per_magic": per_magic,
        "rmse_total_MeV": float(np.sqrt(ssr / n)),
        "rmse_per_nucleon_MeV": float(np.sqrt(np.mean((obs_nuc - pred_nuc) ** 2))),
        "bic": float(n * np.log(ssr / n) + k * np.log(n)),
        "shell_amplitudes_MeV": [round(float(c), 3) for c in coef[5:]],
    }


class _IsotopeColumns(Protocol):
    """Structural view of the columns ``rediscover_magic`` actually reads.

    Lets callers pass any object exposing these four arrays (an ``AmeDataset``,
    or a masked subset view) without the function requiring the full record.
    """

    Z: np.ndarray
    N: np.ndarray
    A: np.ndarray
    BE_per_nucleon: np.ndarray


def rediscover_magic(data: _IsotopeColumns, resid_total: np.ndarray,
                     mask: np.ndarray | None = None, by: str = "N",
                     min_samples: int = 8) -> dict:
    """Find magic numbers from residual bumps (no magic input used).

    Groups SIGNED total-B residuals (observed - SEMF, magic = extra bound
    = positive bump) by N (or Z); reports local maxima over values with
    >= min_samples nuclei. The MAGIC tuple is used ONLY to score hits,
    never to find peaks.
    """
    n_all = int(np.asarray(data.Z).size)
    sel = np.ones(n_all, dtype=bool) if mask is None else mask
    key = np.asarray(data.N[sel] if by == "N" else data.Z[sel], dtype=int)
    resid = np.asarray(resid_total, dtype=float).ravel()
    if resid.shape != key.shape:
        raise ValueError(
            f"residual shape {resid.shape} does not match key shape {key.shape}; "
            "the residual and the isotope selection are out of step"
        )
    vals = sorted(set(key.tolist()))
    means = {v: float(resid[key == v].mean()) for v in vals
             if int((key == v).sum()) >= min_samples}
    ordered = sorted(means)
    peaks = [v for v in ordered[1:-1]
             if means[v] > means[ordered[ordered.index(v) - 1]]
             and means[v] > means[ordered[ordered.index(v) + 1]]
             and means[v] > 0]
    hits_tol = sorted({m for m in MAGIC for p in peaks if abs(p - m) <= 1})
    hits_strict = sorted(set(peaks) & set(MAGIC))
    return {"by": by, "peaks": peaks,
            "peak_means_MeV": {v: round(means[v], 3) for v in peaks},
            "hits": hits_tol, "hits_strict": hits_strict,
            "n_magic": len(MAGIC),
            "recall": f"{len(hits_tol)}/{len(MAGIC)} (+-1 tol)",
            "recall_strict": f"{len(hits_strict)}/{len(MAGIC)} (exact)"}


def _lstsq_fit(Z, N, A, BE):
    """Total-B least squares; returns (coef5, rmse_per_nucleon_on_same_set)."""
    X = design_columns(Z, N, A)
    coef, *_ = np.linalg.lstsq(X, BE * A, rcond=None)
    pred = (X @ coef) / A
    rmse = float(np.sqrt(np.mean((BE - pred) ** 2)))
    return coef, rmse, X


def run_semf_active_loop(data: AmeDataset, rounds: int = 3, seed: int = 0) -> dict:
    """Active design->reveal->refit loop on real AME2020 splits (item 1+2+3).

    Train starts small; each round fits SEMF, draws a parametric-bootstrap
    posterior around the fit, proposes the max-variance pool isotope via
    experiment_designer.semf_isotope_design, reveals its tabulated AME2020
    value (held-out reveal stands in for the lab), and refits. Reports test
    RMSE trajectory on a fixed held-out set.
    """
    from apedp.experiment_designer import semf_isotope_design

    order = np.argsort(data.A, kind="stable")
    Z, N, A, BE = data.Z[order], data.N[order], data.A[order], data.BE_per_nucleon[order]
    big = A >= 16
    Z, N, A, BE = Z[big], N[big], A[big], BE[big]
    test_idx = np.arange(len(A))[::5]
    rest = np.delete(np.arange(len(A)), test_idx)
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(rest, size=60, replace=False)
    pool_idx = np.setdiff1d(rest, train_idx)
    Zt, Nt, At, BEt = Z[test_idx], N[test_idx], A[test_idx], BE[test_idx]

    def test_rmse(coef):
        pred = (design_columns(Zt, Nt, At) @ coef) / At
        return float(np.sqrt(np.mean((BEt - pred) ** 2)))

    trajectory, revealed = [], []
    coef, _, Xtr = _lstsq_fit(Z[train_idx], N[train_idx], A[train_idx], BE[train_idx])
    trajectory.append(test_rmse(coef))
    for _ in range(rounds):
        Ztr, Ntr, Atr, BEtr = Z[train_idx], N[train_idx], A[train_idx], BE[train_idx]
        coef, _, Xtr = _lstsq_fit(Ztr, Ntr, Atr, BEtr)
        resid = BEtr - (Xtr @ coef) / Atr
        dof = max(len(Atr) - 5, 1)
        cov = float(resid @ resid / dof) * np.linalg.inv(Xtr.T @ Xtr)
        draws = rng.multivariate_normal(coef, cov, size=200)
        coeffs: dict[str, Any] = {k: draws[:, i] for i, k in enumerate(["a_v", "a_s", "a_c", "a_a"])}
        coeffs["Delta"] = 12.0 / float(np.sqrt(56.0))
        design = semf_isotope_design(
            coeffs, Z[pool_idx], N[pool_idx], A[pool_idx], n_points=1)
        pick_A = float(design.selected_points[0])
        # Reveal the tabulated value (pool holds real AME2020 numbers).
        cand = pool_idx[np.abs(A[pool_idx] - pick_A).argmin()]
        revealed.append((int(Z[cand]), int(A[cand])))
        train_idx = np.append(train_idx, cand)
        pool_idx = pool_idx[pool_idx != cand]
        coef, _, _ = _lstsq_fit(Z[train_idx], N[train_idx], A[train_idx], BE[train_idx])
        trajectory.append(test_rmse(coef))
    return {"test_rmse_trajectory": trajectory, "revealed_ZA": revealed,
            "n_train_final": int(train_idx.size)}


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SEMF vs AME2020 Validation ===\n")

    path = find_ame_file()
    print(f"Test 0: parsing {path}")
    data = parse_mas20(path)
    print(f"  experimental entries: {len(data)}")
    print(f"  skipped extrapolated: {data.n_skipped_extrapolated}")
    print(f"  skipped missing: {data.n_skipped_missing}")
    assert len(data) > 2000, f"unexpectedly few entries: {len(data)}"

    print("\nTest 1: literature coefficients (A >= 16)")
    mask = data.A >= 16
    pred_lit = semf_per_nucleon(data.Z[mask], data.N[mask], data.A[mask], LIT_COEFFS)
    rmse_lit = float(np.sqrt(np.mean((data.BE_per_nucleon[mask] - pred_lit) ** 2)))
    print(f"  RMSE per nucleon: {rmse_lit:.4f} MeV on {int(mask.sum())} nuclei")

    print("\nTest 2: least-squares fit (A >= 16)")
    fitted, report = fit_semf(data)
    print(f"  fitted: a_v={fitted['a_v']:.3f} a_s={fitted['a_s']:.3f} "
          f"a_c={fitted['a_c']:.4f} a_a={fitted['a_a']:.3f} "
          f"pair_scale={fitted['pair_scale']:.3f}")
    print(f"  RMSE total B: {report['rmse_total_MeV']:.3f} MeV; "
          f"per nucleon: {report['rmse_per_nucleon_MeV']:.4f} MeV; "
          f"R2={report['r_squared']:.5f}")
    assert report["rmse_per_nucleon_MeV"] < rmse_lit, "fit must beat literature"
    assert report["rmse_total_MeV"] < 5.0, "SEMF should fit total B within ~5 MeV"

    print("\nTest 3: magic-number residuals (shell physics missing from SEMF)")
    pred_fit = semf_per_nucleon(data.Z[mask], data.N[mask], data.A[mask],
                                {k: fitted[k] for k in LIT_COEFFS})
    rep = magic_residual_report(data, pred_fit, mask)
    print(f"  magic nuclei: {rep['n_magic']}, other: {rep['n_other']}")
    print(f"  mean|resid| magic: {rep['mean_abs_resid_magic_MeV']:.4f} MeV")
    print(f"  mean|resid| other: {rep['mean_abs_resid_other_MeV']:.4f} MeV")
    assert rep["mean_abs_resid_magic_MeV"] > rep["mean_abs_resid_other_MeV"], \
        "magic nuclei must show larger SEMF residuals (no shell terms)"

    print("\nTest 4: active design->reveal->refit loop on AME2020 splits")
    loop = run_semf_active_loop(data, rounds=3, seed=0)
    traj = loop["test_rmse_trajectory"]
    print(f"  test RMSE trajectory: {[f'{v:.4f}' for v in traj]}")
    print(f"  revealed (Z,A): {loop['revealed_ZA']}")
    assert len(traj) == 4 and all(np.isfinite(traj))
    assert traj[-1] <= traj[0], "more informative data must not hurt the fit"

    print("\nTest 5: phenomenological shell corrections (width scan + BIC)")
    base_bic = float(data.A[mask].size
                     * np.log(report["rmse_total_MeV"] ** 2)
                     + 5 * np.log(data.A[mask].size))
    print(f"  base SEMF BIC: {base_bic:.1f}")
    best = None
    for w in (1.5, 2.0, 3.0, 4.0, 6.0):
        _, rep5 = fit_semf_shell(data, width=w)
        tag = ""
        if best is None or rep5["bic"] < best[1]["bic"]:
            best, tag = (w, rep5), " <-- best"
        print(f"  w={w}: RMSE_total={rep5['rmse_total_MeV']:.3f} MeV, "
              f"BIC={rep5['bic']:.1f}, shell amps={rep5['shell_amplitudes_MeV']}{tag}")
    if best is None:
        raise RuntimeError("shell-term fit produced no candidate")
    assert best[1]["bic"] < base_bic - 10, "shell terms must earn their BIC keep"
    assert all(v > 0 for v in best[1]["shell_amplitudes_MeV"]), \
        "shell amplitudes must be positive (extra binding at closures)"
    _, rep5pm = fit_semf_shell(data, width=best[0], per_magic=True)
    print(f"  per-magic (19 params): RMSE_total={rep5pm['rmse_total_MeV']:.3f}, "
          f"BIC={rep5pm['bic']:.1f}")

    print("\nTest 6: magic-number rediscovery from residual bumps (no magic input)")
    Zm, Nm = data.Z[mask], data.N[mask]
    Xb = design_columns(Zm, Nm, data.A[mask])
    coef_b6, *_lstsq_b6 = np.linalg.lstsq(
        Xb, data.BE_per_nucleon[mask] * data.A[mask], rcond=None)
    resid_tot = data.BE_per_nucleon[mask] * data.A[mask] - Xb @ coef_b6
    # rediscover_magic expects full-length arrays; run on masked subset view
    from types import SimpleNamespace
    sub = SimpleNamespace(Z=Zm, N=Nm, A=data.A[mask],
                          BE_per_nucleon=data.BE_per_nucleon[mask])
    rn = rediscover_magic(sub, resid_tot, by="N")
    rz = rediscover_magic(sub, resid_tot, by="Z")
    print(f"  N-peaks: {rn['peaks']} -> hits {rn['hits']} recall {rn['recall']}")
    print(f"  Z-peaks: {rz['peaks']} -> hits {rz['hits']} recall {rz['recall']}")
    found = sorted(set(rn["hits"]) | set(rz["hits"]))
    print(f"  combined rediscovered: {found}")
    assert len(found) >= 3, f"too few magic numbers rediscovered: {found}"

    print("\n=== All SEMF validation tests passed ===")
