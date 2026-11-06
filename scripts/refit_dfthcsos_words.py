"""Measure, then improve, the dfthcsos angle table at a coarser precision.

The Benchmark III verifier rejects the 9-circuit family (`brot = 11`, `J = 9`)
with `error_1 = 221 mHa` and `error_2 = 51.7 mHa` against a 20 mHa budget, and
both numbers come from *requantising* the shipped 12-bit words. This script asks
the question that requantisation cannot answer: how much of that residual is the
grid, and how much is the vector set itself?

`--diagnose` (the default) evaluates the same submission at four settings with
the verifier's own code, without building or simulating a circuit:

  * `continuous`  the factorisation's own vectors, never quantised;
  * `brot 12`     the shipped words, which is the passing submission;
  * `brot 11`     the 9-circuit family's words;
  * `brot 10`     the 8-circuit family's words.

If `continuous` is far inside 20 mHa, the grid is the whole story and a refit has
room to recover it. If it is close to 20 mHa, no refit at a coarser grid can
pass, and the 12-bit floor is a property of the factorisation rather than of the
requantisation, which is a different and stronger finding.

`--refit BROT` then runs a greedy per-row refit in vector space and re-encodes the
result on the Givens tree at that precision, reporting the exact verifier errors
after each sweep. The refit never touches a circuit; it changes the words those
circuits load.

Requires the pinned FCIDUMP, which the competition data ships and this checkout
does not (see `docs/DFTHCSOS_HANDOVER.md`):

    kaggle competitions download -c google-verifiable-quantum-circuit-compilation \\
        -f verifiers/dfthcsos/FeMoCo76o113e.fcidump -p /tmp/qcc
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "apedp_dyn" / "sandbox" / "google-verifiable-quantum-circuit-compilation"
VERIFIERS = BUNDLE / "verifiers"
SAMPLE = BUNDLE / "sample_submission" / "dfthcsos"
DEFAULT_FCIDUMP = pathlib.Path("/tmp/qcc/verifiers/dfthcsos/FeMoCo76o113e.fcidump")

sys.path.insert(0, str(VERIFIERS))

from dfthcsos.scoreinfra.dfthc_math import (  # noqa: E402
    build_h1_prime,
    evaluate_dfthc_vectors,
    reconstruct_u_vectors,
    reconstruct_x,
    symmetric_from_lower,
)
from dfthcsos.scoreinfra.reference import (  # noqa: E402
    UPPER_ENERGY,
    check_fcidump,
    load_fixed_shifts,
    load_factorization,
    load_reference,
)
ERROR_LIMIT = 0.020  # both gates, from dfthcsos.verify


def native_inputs(params: dict, num_orbitals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The trainer's ``(w, W, S, beta_1)``, as ``dfthcsos.verify.native_inputs`` builds them.

    Copied rather than imported: that module pulls in `scoreinfra.circuits`, which
    imports `kickmix`, and nothing here simulates a circuit.
    """
    w = np.asarray(params["W"], dtype=np.float64)
    w_rho = np.asarray(params["H2_I"], dtype=np.float64) - np.sum(w, axis=1)
    s_matrix = np.asarray(symmetric_from_lower(params["Hsym_lower"], num_orbitals), dtype=np.float64)
    beta_1 = float(params["H1_I"]) - 0.5 * float(np.trace(s_matrix))
    return w, w_rho, s_matrix, beta_1


def load_inputs(fcidump: pathlib.Path) -> dict:
    """The factorisation, the fixed shifts and the reference integrals."""
    check_fcidump(fcidump)
    with np.load(SAMPLE / "submission.npz", allow_pickle=False) as data:
        params = load_factorization(SAMPLE / "submission.npz")
        words = np.array(data["angle_words"], dtype=np.int64)
        tree = np.array(data["tree_edges"], dtype=np.int64)
    params.update(load_fixed_shifts())
    ref = load_reference(fcidump)
    w, w_rho, s_matrix, beta_1 = native_inputs(params, int(ref["orbitals"]))
    return dict(params=params, w=w, W=w_rho, S=s_matrix, beta_1=beta_1,
                ref=ref, words=words, tree=tree)


def symmetrised_x(u_two_body: np.ndarray, w: np.ndarray) -> np.ndarray:
    """X from the two-body rows, symmetrised from its lower triangle, as the verifier does."""
    factors = reconstruct_x(u_two_body, w)
    return np.tril(factors) + np.swapaxes(np.tril(factors, k=-1), -1, -2)


def one_body_rows(inputs: dict) -> np.ndarray:
    """The eigenvector rows of h1_prime, which the submission's first N words encode."""
    ref = inputs["ref"]
    tei = np.asarray(ref["tei"], dtype=np.float64)
    h1_thc = np.asarray(ref["oei"], dtype=np.float64) - 0.5 * np.einsum(
        "prrq->pq", tei, optimize=True)
    factors = symmetrised_x(inputs["params"]["U"], inputs["w"])
    h1_prime = build_h1_prime(h1_thc, inputs["S"], inputs["beta_1"],
                              ref["electrons"], inputs["W"], factors)
    return np.linalg.eigh(h1_prime)[1].T


def evaluate(inputs: dict, vectors: np.ndarray) -> dict:
    return evaluate_dfthc_vectors(
        vectors, inputs["w"], inputs["W"], inputs["S"], inputs["beta_1"], inputs["ref"],
        energy_upper=UPPER_ENERGY,
    )


def requantise(words: np.ndarray, src_bits: int, dst_bits: int) -> np.ndarray:
    """Move a turn-count table to fewer bits, wrapping rather than overflowing."""
    mask = (1 << dst_bits) - 1
    if dst_bits >= src_bits:
        return (words << (dst_bits - src_bits)) & mask
    shift = src_bits - dst_bits
    return ((words + (1 << (shift - 1))) >> shift) & mask


def vectors_from_words(words: np.ndarray, brot: int, tree: np.ndarray) -> np.ndarray:
    return reconstruct_u_vectors(words / float(1 << (brot - 1)), tree)


def encode_on_tree(vectors: np.ndarray, tree: np.ndarray, brot: int) -> np.ndarray:
    """The words that regenerate `vectors` on the Givens tree, rounded to `brot` bits.

    Inverts `reconstruct_u_vectors` edge by edge in reverse: undoing a rotation with
    the angle that zeroes the child's coefficient recovers the state the forward pass
    rotated, so the recovered angle is the one it applied. Rounded, it is the best
    word the grid offers for this vector — which is what makes this an oracle: the
    vector set is left free and only the grid binds.
    """
    edges = np.asarray(tree, dtype=np.int64)
    scale = float(1 << (brot - 1))
    modulus = 1 << brot
    rows = np.asarray(vectors, dtype=np.float64)
    words = np.zeros((rows.shape[0], edges.shape[0]), dtype=np.int64)
    for row, target in enumerate(rows):
        residual = target / np.linalg.norm(target)
        for index in range(edges.shape[0] - 1, -1, -1):
            parent, child = edges[index]
            angle = math.atan2(residual[child], residual[parent])
            words[row, index] = int(round(angle / math.pi * scale)) % modulus
            cosine, sine = math.cos(angle), math.sin(angle)
            parent_value, child_value = residual[parent], residual[child]
            residual[parent] = cosine * parent_value + sine * child_value
            residual[child] = -sine * parent_value + cosine * child_value
        if abs(abs(residual[0]) - 1.0) > 1e-6:
            raise AssertionError(f"row {row} did not invert to e_0: {residual[0]}")
    return words


def report(label: str, metrics: dict) -> None:
    verdict = ("PASS" if metrics["error_1"] <= ERROR_LIMIT and metrics["error_2"] <= ERROR_LIMIT
               else "FAIL")
    print(f"{label:>12}  error_1 {metrics['error_1'] * 1e3:9.3f} mHa   "
          f"error_2 {metrics['error_2'] * 1e3:9.3f} mHa   "
          f"lambda_eff {metrics['lambda_eff']:8.3f}   {verdict}")


def continuous_vectors(inputs: dict) -> np.ndarray:
    return np.vstack([one_body_rows(inputs),
                      inputs["params"]["U"].reshape(-1, inputs["ref"]["orbitals"])])


def diagnose(inputs: dict) -> dict:
    metrics = evaluate(inputs, continuous_vectors(inputs))
    print(f"{'setting':>12}  {'error_1':>18}   {'error_2':>18}   {'lambda_eff':>17}   verdict")
    report("continuous", metrics)
    for brot in (12, 11, 10):
        words = requantise(inputs["words"], 12, brot)
        report(f"brot {brot}", evaluate(inputs, vectors_from_words(words, brot, inputs["tree"])))
    headroom = 1e3 * (ERROR_LIMIT - metrics["error_2"])
    print(f"\nheadroom above the continuous floor: {headroom:.3f} mHa")
    return metrics


def oracle(inputs: dict, brots: tuple[int, ...]) -> None:
    """The best error each grid allows, in the order the shipped table is built.

    Two bodies first, then one bodies. The verifier builds `h1_prime` from the
    *decoded* X, so the eigenvectors it pairs with exact eigenvalues are those of
    the rebuilt `h1_prime`, not of the pre-quantisation one. Encoding the latter
    leaves `error_1` at tens of mHa even at 12 bits -- which is precisely the
    mistake plain requantisation makes, and the reason its residual jumps.

    At 12 bits this reproduces the shipped submission's numbers; that is the check
    that the bound it gives at 11 bits is the tight one rather than an artefact.
    """
    ref = inputs["ref"]
    orbitals = int(ref["orbitals"])
    rank_r, bases_b, _ = inputs["w"].shape
    two_body = inputs["params"]["U"].reshape(-1, orbitals)
    tei = np.asarray(ref["tei"], dtype=np.float64)
    h1_thc = np.asarray(ref["oei"], dtype=np.float64) - 0.5 * np.einsum(
        "prrq->pq", tei, optimize=True)
    print(f"\n{'setting':>12}  {'error_1':>18}   {'error_2':>18}   {'lambda_eff':>17}   verdict")
    for brot in brots:
        two_words = encode_on_tree(two_body, inputs["tree"], brot)
        two_vectors = vectors_from_words(two_words, brot, inputs["tree"])
        factors = symmetrised_x(two_vectors.reshape(rank_r, bases_b, orbitals), inputs["w"])
        h1_prime = build_h1_prime(h1_thc, inputs["S"], inputs["beta_1"],
                                  ref["electrons"], inputs["W"], factors)
        one_words = encode_on_tree(np.linalg.eigh(h1_prime)[1].T, inputs["tree"], brot)
        words = np.vstack([one_words, two_words])
        report(f"oracle {brot}", evaluate(inputs, vectors_from_words(words, brot, inputs["tree"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fcidump", type=pathlib.Path, default=DEFAULT_FCIDUMP)
    parser.add_argument("--oracle", type=int, nargs="+", metavar="BROT", default=[],
                        help="encode the unquantised vectors on the tree at these precisions")
    args = parser.parse_args()

    if not args.fcidump.is_file():
        print(f"FCIDUMP not found: {args.fcidump}\n"
              "Download it with the command in this script's docstring.", file=sys.stderr)
        return 2
    inputs = load_inputs(args.fcidump)
    diagnose(inputs)
    if args.oracle:
        oracle(inputs, tuple(args.oracle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
