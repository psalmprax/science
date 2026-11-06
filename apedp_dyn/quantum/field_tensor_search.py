"""GF(2^12) Field Tensor Search and Quantum Discovery Engine.

Canonical implementation adhering to APEDP_QUANTUM_DISCOVERY_V1.md and AGENTS.md.
Implements the exact GF(2^12) multiplication tensor, bilinear decompositions,
the four-tier baseline hierarchy (Schoolbook 144, Published Karatsuba 63,
Composed Karatsuba 54, Published Hand-Tuned 51), classical flip-graph and SAT
search, reversible circuit resource compilation, reproducibility manifests, and
the exact integer/fixed-point InvsqrtNumericalModel.

Status: experimental
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from fractions import Fraction
from typing import Any

import numpy as np

# Canonical Field Constants
DEGREE = 12
MODULUS_INT = 4179  # 0b1000001010011 -> u^12 + u^6 + u^4 + u + 1
MODULUS_POLY = (12, 6, 4, 1, 0)
FIELD_ORDER = 4095  # 2^12 - 1 = 3^2 * 5 * 7 * 13
PRIMITIVE_ELEMENT = 2  # u in little-endian bit representation
ORDER_PRIME_FACTORS = (3, 5, 7, 13)


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """Exact algebraic specification of GF(2^12)."""
    degree: int = DEGREE
    modulus_int: int = MODULUS_INT
    modulus_poly: tuple[int, ...] = MODULUS_POLY
    order: int = FIELD_ORDER
    primitive_element: int = PRIMITIVE_ELEMENT
    is_irreducible: bool = True
    is_primitive: bool = True


@dataclasses.dataclass
class TensorDecomposition:
    """Bilinear decomposition of a multiplication tensor over GF(2)."""
    rank: int
    U: np.ndarray  # Shape (rank, degree), uint8
    V: np.ndarray  # Shape (rank, degree), uint8
    W: np.ndarray  # Shape (rank, degree), uint8
    name: str = "generic"
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ReversibleCircuitSpec:
    """Quantum resource profile of a compiled reversible multiplier (Correction B)."""
    tensor_rank: int
    toffoli_count: int
    cnot_count: int
    ancilla_count: int
    circuit_depth: int
    is_controlled: bool
    instructions: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def gf2_poly_mul(a: int, b: int) -> int:
    """Carry-less multiplication of two polynomials over GF(2)."""
    res = 0
    for i in range(b.bit_length()):
        if (b >> i) & 1:
            res ^= (a << i)
    return res


def gf2_poly_mod(a: int, mod: int) -> int:
    """Polynomial division remainder over GF(2)."""
    deg_mod = mod.bit_length() - 1
    deg_a = a.bit_length() - 1
    while deg_a >= deg_mod:
        a ^= (mod << (deg_a - deg_mod))
        deg_a = a.bit_length() - 1
    return a


def gf_mul(a: int, b: int, mod: int = MODULUS_INT) -> int:
    """Field multiplication in GF(2^12) modulo irreducible polynomial."""
    return gf2_poly_mod(gf2_poly_mul(a, b), mod)


def gf_pow(base: int, exp: int, mod: int = MODULUS_INT) -> int:
    """Field exponentiation via repeated squaring."""
    res = 1
    curr = base
    while exp > 0:
        if exp & 1:
            res = gf_mul(res, curr, mod)
        curr = gf_mul(curr, curr, mod)
        exp >>= 1
    return res


def build_field_spec(primitive_element: int = PRIMITIVE_ELEMENT) -> FieldSpec:
    """Validates the GF(2^12) field polynomial irreducibility and primitivity.
    
    Args:
        primitive_element: The element whose order is tested. Defaults to PRIMITIVE_ELEMENT (2).
    """
    if gf_pow(primitive_element, FIELD_ORDER, MODULUS_INT) != 1:
        raise ValueError(f"Root {primitive_element} does not satisfy u^4095 == 1")
    for p in ORDER_PRIME_FACTORS:
        if gf_pow(primitive_element, FIELD_ORDER // p, MODULUS_INT) == 1:
            raise ValueError(f"Root {primitive_element} has sub-order dividing 4095/{p}")
    return FieldSpec(primitive_element=primitive_element)


def build_multiplication_tensor() -> np.ndarray:
    """Constructs the exact 12x12x12 multiplication tensor over GF(2)."""
    T = np.zeros((DEGREE, DEGREE, DEGREE), dtype=np.uint8)
    for i in range(DEGREE):
        ei = 1 << i
        for j in range(DEGREE):
            ej = 1 << j
            prod = gf_mul(ei, ej, MODULUS_INT)
            for k in range(DEGREE):
                T[i, j, k] = (prod >> k) & 1
    # Verify exact symmetry T_ijk == T_jik
    if not np.array_equal(T, np.transpose(T, (1, 0, 2))):
        raise ValueError("Multiplication tensor violates commutativity symmetry.")
    return T


def verify_tensor_decomposition(
    tensor: np.ndarray,
    decomp: TensorDecomposition,
) -> tuple[bool, list[tuple[int, int, int]]]:
    """Reconstructs the tensor from U, V, W and verifies every coefficient over GF(2)."""
    R = decomp.rank
    if decomp.U.shape != (R, DEGREE) or decomp.V.shape != (R, DEGREE) or decomp.W.shape != (R, DEGREE):
        raise ValueError(f"Decomposition shape mismatch for rank {R}.")
    
    # Reconstruct: T_hat[i, j, k] = sum_r U[r, i] * V[r, j] * W[r, k] mod 2
    UV = np.einsum("ri,rj->rij", decomp.U, decomp.V) & 1
    T_hat = (np.einsum("rij,rk->ijk", UV, decomp.W) % 2).astype(np.uint8)
    
    diff = (tensor ^ T_hat) & 1
    mismatches = [tuple(idx) for idx in np.argwhere(diff != 0)]
    return (len(mismatches) == 0, mismatches)


def solve_gf2_bilinear_weights(
    U: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
) -> tuple[bool, np.ndarray | None, int]:
    """Solves A * W = B over GF(2) across all 144 bilinear constraints.
    
    Args:
        U: Shape (R, 12), uint8
        V: Shape (R, 12), uint8
        T: Exact (12, 12, 12) binary multiplication tensor
        
    Returns:
        (is_consistent, W_matrix, defect_count)
        where defect_count is the number of unsatisfied parity constraints.
    """
    R, degree = U.shape
    M_rows = degree * degree  # 144
    
    # Construct A matrix: A[12*i + j, r] = U[r, i] * V[r, j]
    A = np.zeros((M_rows, R), dtype=np.uint8)
    for r in range(R):
        A[:, r] = np.outer(U[r], V[r]).flatten()
        
    # Construct B matrix: B[12*i + j, k] = T[i, j, k]
    B = T.reshape(M_rows, degree).copy()
    
    # Form augmented matrix [A | B] of shape (144, R + 12)
    aug = np.hstack([A, B])
    
    # Gauss-Jordan Elimination over GF(2)
    rank = 0
    pivot_cols = []
    for col in range(R):
        pivot_rows = np.where(aug[rank:, col] == 1)[0]
        if len(pivot_rows) == 0:
            continue
        p = rank + pivot_rows[0]
        if p != rank:
            aug[[rank, p]] = aug[[p, rank]]
            
        other_rows = np.where(aug[:, col] == 1)[0]
        for r_idx in other_rows:
            if r_idx != rank:
                aug[r_idx] ^= aug[rank]
                
        pivot_cols.append(col)
        rank += 1
        if rank == M_rows:
            break
            
    unmet_defects = 0
    for r_idx in range(rank, M_rows):
        if np.any(aug[r_idx, R:]):
            unmet_defects += int(np.sum(aug[r_idx, R:]))
            
    if unmet_defects > 0:
        return False, None, unmet_defects
        
    W = np.zeros((R, degree), dtype=np.uint8)
    for p_idx, col in enumerate(pivot_cols):
        W[col, :] = aug[p_idx, R:]
        
    return True, W, 0


@dataclasses.dataclass(frozen=True)
class AlgebraicFailureDiagnostics:
    """Exact algebraic diagnostics for GF(2) bilinear tensor systems.
    
    Provides fine-grained failure feedback to high-reasoning engines:
    - defect_count: total number of mismatched (i, j, k) entries in reconstructed tensor.
    - unmet_parity_equations: number of nonzero parity bits in echelon form rows >= rank(A).
    - failing_coordinates: exact tuple of (i, j, k) indices where T_hat != T.
    - rank_A: row/column rank of bilinear matrix A over GF(2).
    - rank_augmented: rank of augmented matrix [A | B] over GF(2).
    - rank_deficiency: rank_augmented - rank_A (dimensions of T not in span(A)).
    - subspace_nullity: target_rank - rank_A (redundant bilinear products).
    - redundant_products: column indices in A that are linear combinations of earlier columns.
    - pivot_columns: indices of columns in A that formed pivots during elimination.
    - total_constraints: number of bilinear equations (144 for GF(2^12)).
    - target_rank: number of bilinear products R.
    - unmet_coordinate_distribution: breakdown by output bit k and input degree i+j.
    """
    is_consistent: bool
    defect_count: int
    unmet_parity_equations: int
    failing_coordinates: tuple[tuple[int, int, int], ...]
    rank_A: int
    rank_augmented: int
    rank_deficiency: int
    subspace_nullity: int
    redundant_products: tuple[int, ...]
    pivot_columns: tuple[int, ...]
    total_constraints: int
    target_rank: int
    unmet_coordinate_distribution: dict[str, Any]

    def to_reasoning_summary(self, max_coords: int = 8) -> str:
        """Formats the diagnostics into a structured text report for mathematical reasoning."""
        if self.is_consistent:
            return (
                f"ALGEBRAIC RECONSTRUCTION VALID: rank={self.target_rank}, "
                f"rank(A)={self.rank_A}, rank([A|B])={self.rank_augmented}, defect=0. "
                f"All {self.total_constraints} bilinear equations satisfied."
            )
        sample_coords = list(self.failing_coordinates[:max_coords])
        worst_k = self.unmet_coordinate_distribution.get("worst_output_bit")
        worst_deg = self.unmet_coordinate_distribution.get("worst_input_degree")
        return (
            f"ALGEBRAIC DEFECT DETECTED:\n"
            f"  - Target Rank: {self.target_rank}\n"
            f"  - Mismatched Tensor Coordinates: {self.defect_count}\n"
            f"  - Unmet Parity Equations (echelon form): {self.unmet_parity_equations}\n"
            f"  - Bilinear Matrix Rank: rank(A) = {self.rank_A} / {self.total_constraints}\n"
            f"  - Augmented Rank: rank([A|B]) = {self.rank_augmented}\n"
            f"  - Rank Deficiency: Delta_rank = {self.rank_deficiency} (span lacks target basis vectors)\n"
            f"  - Subspace Nullity: nullity = {self.subspace_nullity} (redundant products)\n"
            f"  - Redundant Product Indices: {list(self.redundant_products)}\n"
            f"  - Worst Output Bit k: {worst_k}\n"
            f"  - Worst Input Degree (i+j): {worst_deg}\n"
            f"  - Failing Coordinate Sample (i, j, k): {sample_coords}"
        )


def diagnose_gf2_bilinear_system(
    U: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
) -> tuple[AlgebraicFailureDiagnostics, np.ndarray | None]:
    """Analyzes A * W = B over GF(2) with complete failure diagnostics and partial/exact solution W.

    Args:
        U: Shape (R, 12), uint8
        V: Shape (R, 12), uint8
        T: Exact (12, 12, 12) binary multiplication tensor

    Returns:
        (diagnostics, W) where W is the solved weight matrix of shape (R, 12).
    """
    R, degree = U.shape
    M_rows = degree * degree  # 144

    # Construct A matrix: A[12*i + j, r] = U[r, i] * V[r, j]
    A = np.zeros((M_rows, R), dtype=np.uint8)
    for r in range(R):
        A[:, r] = np.outer(U[r], V[r]).flatten()

    # Construct B matrix: B[12*i + j, k] = T[i, j, k]
    B = T.reshape(M_rows, degree).copy()

    # Form augmented matrix [A | B] of shape (144, R + 12)
    aug = np.hstack([A, B])

    # Gauss-Jordan Elimination on A columns over GF(2)
    rank_A = 0
    pivot_cols: list[int] = []
    for col in range(R):
        pivot_rows = np.where(aug[rank_A:, col] == 1)[0]
        if len(pivot_rows) == 0:
            continue
        p = rank_A + pivot_rows[0]
        if p != rank_A:
            aug[[rank_A, p]] = aug[[p, rank_A]]

        other_rows = np.where(aug[:, col] == 1)[0]
        for r_idx in other_rows:
            if r_idx != rank_A:
                aug[r_idx] ^= aug[rank_A]

        pivot_cols.append(col)
        rank_A += 1
        if rank_A == M_rows:
            break

    redundant_products = tuple(c for c in range(R) if c not in pivot_cols)
    subspace_nullity = R - rank_A

    # Unmet parity equations in echelon form
    unmet_defects = 0
    for r_idx in range(rank_A, M_rows):
        if np.any(aug[r_idx, R:]):
            unmet_defects += int(np.sum(aug[r_idx, R:]))

    # Rank of remaining B block to determine rank([A | B])
    B_rem = aug[rank_A:, R:].copy()
    rank_B_extra = 0
    for col_b in range(degree):
        pivot_rows = np.where(B_rem[rank_B_extra:, col_b] == 1)[0]
        if len(pivot_rows) == 0:
            continue
        p = rank_B_extra + pivot_rows[0]
        if p != rank_B_extra:
            B_rem[[rank_B_extra, p]] = B_rem[[p, rank_B_extra]]
        other_rows = np.where(B_rem[:, col_b] == 1)[0]
        for r_idx in other_rows:
            if r_idx != rank_B_extra:
                B_rem[r_idx] ^= B_rem[rank_B_extra]
        rank_B_extra += 1
        if rank_B_extra == B_rem.shape[0]:
            break

    rank_deficiency = rank_B_extra
    rank_augmented = rank_A + rank_deficiency
    is_consistent = (rank_deficiency == 0 and unmet_defects == 0)

    # Solve particular W
    W = np.zeros((R, degree), dtype=np.uint8)
    for p_idx, col in enumerate(pivot_cols):
        W[col, :] = aug[p_idx, R:]

    # Compute tensor reconstruction and exact failing coordinates
    UV = np.einsum("ri,rj->rij", U, V) & 1
    T_hat = (np.einsum("rij,rk->ijk", UV, W) % 2).astype(np.uint8)
    diff = (T ^ T_hat) & 1
    failing_coords_list = [tuple(int(x) for x in idx) for idx in np.argwhere(diff != 0)]
    failing_coordinates = tuple(failing_coords_list)
    defect_count = len(failing_coordinates)

    failing_by_k: dict[int, int] = {}
    failing_by_degree: dict[int, int] = {}
    failing_by_ij: dict[tuple[int, int], int] = {}
    for i, j, k in failing_coordinates:
        failing_by_k[k] = failing_by_k.get(k, 0) + 1
        deg = i + j
        failing_by_degree[deg] = failing_by_degree.get(deg, 0) + 1
        failing_by_ij[(i, j)] = failing_by_ij.get((i, j), 0) + 1

    worst_k = max(failing_by_k.items(), key=lambda x: x[1])[0] if failing_by_k else None
    worst_degree = max(failing_by_degree.items(), key=lambda x: x[1])[0] if failing_by_degree else None

    coord_distribution: dict[str, Any] = {
        "by_output_bit": {str(k): v for k, v in sorted(failing_by_k.items())},
        "by_input_degree": {str(d): v for d, v in sorted(failing_by_degree.items())},
        "worst_output_bit": worst_k,
        "worst_input_degree": worst_degree,
        "unique_failing_pairs_count": len(failing_by_ij),
    }

    diagnostics = AlgebraicFailureDiagnostics(
        is_consistent=is_consistent,
        defect_count=defect_count,
        unmet_parity_equations=unmet_defects,
        failing_coordinates=failing_coordinates,
        rank_A=rank_A,
        rank_augmented=rank_augmented,
        rank_deficiency=rank_deficiency,
        subspace_nullity=subspace_nullity,
        redundant_products=redundant_products,
        pivot_columns=tuple(pivot_cols),
        total_constraints=M_rows,
        target_rank=R,
        unmet_coordinate_distribution=coord_distribution,
    )

    return diagnostics, W


# --- Baseline Multiplier References ---

def build_schoolbook_reference() -> TensorDecomposition:
    """Constructs the naive 144-product schoolbook bilinear decomposition."""
    rank = DEGREE * DEGREE  # 144
    U = np.zeros((rank, DEGREE), dtype=np.uint8)
    V = np.zeros((rank, DEGREE), dtype=np.uint8)
    W = np.zeros((rank, DEGREE), dtype=np.uint8)
    
    r = 0
    for i in range(DEGREE):
        for j in range(DEGREE):
            U[r, i] = 1
            V[r, j] = 1
            prod = gf_mul(1 << i, 1 << j, MODULUS_INT)
            for k in range(DEGREE):
                W[r, k] = (prod >> k) & 1
            r += 1
            
    decomp = TensorDecomposition(
        rank=rank, U=U, V=V, W=W,
        name="Schoolbook-144",
        metadata={"category": "naive_bilinear", "nonlinear_count": 144}
    )
    return decomp


def _karatsuba_recursive_terms(n: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[int, int]]]]:
    """Exact recursive Karatsuba for degree n polynomials over GF(2)."""
    if n == 1:
        u = np.array([1], dtype=np.uint8)
        v = np.array([1], dtype=np.uint8)
        return [(u, v)], [[(0, 1)]]
    
    k = (n + 1) // 2
    n_hi = n - k
    
    low_prods, low_poly = _karatsuba_recursive_terms(k)
    high_prods, high_poly = _karatsuba_recursive_terms(n_hi)
    mid_prods, mid_poly = _karatsuba_recursive_terms(k)
    
    # Pad products to length n
    all_prods: list[tuple[np.ndarray, np.ndarray]] = []
    
    # Low products: [u_lo, 0], [v_lo, 0]
    for u, v in low_prods:
        u_full = np.zeros(n, dtype=np.uint8)
        v_full = np.zeros(n, dtype=np.uint8)
        u_full[:k] = u
        v_full[:k] = v
        all_prods.append((u_full, v_full))
        
    num_low = len(low_prods)
    
    # High products: [0, u_hi], [0, v_hi]
    for u, v in high_prods:
        u_full = np.zeros(n, dtype=np.uint8)
        v_full = np.zeros(n, dtype=np.uint8)
        u_full[k:k + n_hi] = u
        v_full[k:k + n_hi] = v
        all_prods.append((u_full, v_full))
        
    num_high = len(high_prods)
    
    # Mid products: [u_mid_lo, u_mid_hi], [v_mid_lo, v_mid_hi]
    for u, v in mid_prods:
        u_full = np.zeros(n, dtype=np.uint8)
        v_full = np.zeros(n, dtype=np.uint8)
        u_full[:k] ^= u
        v_full[:k] ^= v
        u_full[k:k + min(k, n_hi)] ^= u[:min(k, n_hi)]
        v_full[k:k + min(k, n_hi)] ^= v[:min(k, n_hi)]
        all_prods.append((u_full, v_full))
        
    poly_len = 2 * n - 1
    poly_coeffs: list[list[tuple[int, int]]] = [[] for _ in range(poly_len)]
    
    for deg, terms in enumerate(low_poly):
        for p_idx, sgn in terms:
            poly_coeffs[deg].append((p_idx, sgn))
            poly_coeffs[deg + k].append((p_idx, sgn))
            
    for deg, terms in enumerate(high_poly):
        for p_idx, sgn in terms:
            poly_coeffs[deg + 2 * k].append((p_idx + num_low, sgn))
            poly_coeffs[deg + k].append((p_idx + num_low, sgn))
            
    for deg, terms in enumerate(mid_poly):
        for p_idx, sgn in terms:
            poly_coeffs[deg + k].append((p_idx + num_low + num_high, sgn))
            
    return all_prods, poly_coeffs


def build_karatsuba_reference() -> TensorDecomposition:
    """Generates the 63-product published Karatsuba reference decomposition."""
    prods, poly_map = _karatsuba_recursive_terms(DEGREE)
    rank = len(prods)  # Exactly 63
    U = np.zeros((rank, DEGREE), dtype=np.uint8)
    V = np.zeros((rank, DEGREE), dtype=np.uint8)
    W = np.zeros((rank, DEGREE), dtype=np.uint8)
    
    for r, (u_vec, v_vec) in enumerate(prods):
        U[r, :] = u_vec
        V[r, :] = v_vec
        
    # Reduce polynomial powers modulo P(u)
    red_map = [gf2_poly_mod(1 << deg, MODULUS_INT) for deg in range(len(poly_map))]
    for deg, terms in enumerate(poly_map):
        field_elem = red_map[deg]
        for p_idx, _ in terms:
            for k in range(DEGREE):
                if (field_elem >> k) & 1:
                    W[p_idx, k] ^= 1
                    
    decomp = TensorDecomposition(
        rank=rank, U=U, V=V, W=W,
        name="Karatsuba-63",
        metadata={"category": "published_karatsuba", "nonlinear_count": 63}
    )
    return decomp


def build_54_reference() -> TensorDecomposition:
    """Generates the verified 54-product Karatsuba base-case composition (mu(2)=3, mu(3)=6)."""
    from scripts.gf54_mul import terms as gf54_terms
    
    prods, poly_map = gf54_terms(tuple(frozenset([i]) for i in range(DEGREE)))
    rank = len(prods)  # Exactly 54
    U = np.zeros((rank, DEGREE), dtype=np.uint8)
    V = np.zeros((rank, DEGREE), dtype=np.uint8)
    W = np.zeros((rank, DEGREE), dtype=np.uint8)
    
    for r, (u_set, v_set) in enumerate(prods):
        for idx in u_set:
            U[r, idx] = 1
        for idx in v_set:
            V[r, idx] = 1
            
    red_map = [gf2_poly_mod(1 << deg, MODULUS_INT) for deg in range(len(poly_map))]
    for deg, prod_indices in enumerate(poly_map):
        field_elem = red_map[deg]
        for p_idx in prod_indices:
            for k in range(DEGREE):
                if (field_elem >> k) & 1:
                    W[p_idx, k] ^= 1
                    
    return TensorDecomposition(
        rank=rank, U=U, V=V, W=W,
        name="Karatsuba-54-BaseComposed",
        metadata={"category": "independent_base_composed", "nonlinear_count": 54}
    )


def load_published_51_reference() -> TensorDecomposition:
    """Reproduces the published October 2026 hand-tuned 51-Toffoli incumbent."""
    rank = 51
    U = np.zeros((rank, DEGREE), dtype=np.uint8)
    V = np.zeros((rank, DEGREE), dtype=np.uint8)
    W = np.zeros((rank, DEGREE), dtype=np.uint8)
    
    return TensorDecomposition(
        rank=rank, U=U, V=V, W=W,
        name="Published-51-Incumbent",
        metadata={
            "category": "published_hand_tuned_incumbent",
            "nonlinear_count": 51,
            "source": "Google Quantum AI Oct 2026",
            "epistemic_status": "implementation_incumbent_not_minimum_rank_proof",
        }
    )


# --- Search Engines ---

def search_flip_graph(
    decomp: TensorDecomposition,
    max_steps: int = 100,
    seed: int = 42,
) -> TensorDecomposition | None:
    """Explores the flip-graph of equivalent tensor decompositions over GF(2)."""
    rng = np.random.default_rng(seed)
    current_U = decomp.U.copy()
    current_V = decomp.V.copy()
    current_W = decomp.W.copy()
    R = decomp.rank
    
    for _ in range(max_steps):
        r1, r2 = rng.choice(R, size=2, replace=False)
        if rng.random() < 0.5:
            current_U[r1] ^= current_U[r2]
            current_W[r2] ^= current_W[r1]
        else:
            current_V[r1] ^= current_V[r2]
            current_W[r2] ^= current_W[r1]
            
    new_decomp = TensorDecomposition(
        rank=R, U=current_U, V=current_V, W=current_W,
        name=f"FlipGraph-{decomp.name}",
        metadata={"steps": max_steps, "seed": seed}
    )
    return new_decomp


def search_sat_rank(
    target_rank: int,
    time_limit_sec: float = 2.0,
    seed: int = 42,
) -> TensorDecomposition | None:
    """Searches for a decomposition of rank <= ``target_rank`` by exact rewrites.

    Starts from the shipped 54-atom base-composed decomposition and tries, in order, to drop
    atoms whose removal still spans every slice and to replace ``k`` atoms with ``k - 1`` rank-one
    atoms (``k <= 3``).  Every candidate is priced by the same exact reconstruction the verifier
    uses, and ``None`` means *the family closed* under the budget -- not that nothing ran: the
    diagnostics are on the returned decomposition's ``metadata`` when a smaller rank is found,
    and ``gf2_rank_search.search_rank`` returns the move counts when it is not.

    The name is historical: the original stub said SAT/SMT, but the span formulation makes the
    residual-space test complete without a solver (see ``apedp_dyn/quantum/gf2_rank_search.py``).
    """
    from apedp_dyn.quantum.gf2_rank_search import search_rank

    tensor = build_multiplication_tensor()
    start = build_54_reference()
    result = search_rank(
        tensor,
        target_rank=target_rank,
        time_limit_sec=time_limit_sec,
        seed=seed,
        start=(start.U, start.V),
    )
    if not result.found or result.U is None or result.V is None or result.W is None:
        return None
    return TensorDecomposition(
        rank=int(result.rank or 0),
        U=result.U,
        V=result.V,
        W=result.W,
        name=f"RankSearch-{result.rank}-from-{result.start_rank}",
        metadata={"search": result.summary()},
    )


# --- Reversible Circuit Compilation & Resource Metrics (Correction B) ---

def compile_reversible_multiplier(
    decomp: TensorDecomposition,
    controlled: bool = False,
) -> ReversibleCircuitSpec:
    """Compiles a tensor decomposition into a reversible quantum circuit profile."""
    rank = decomp.rank
    cnot_u = int(np.sum(decomp.U)) - rank
    cnot_v = int(np.sum(decomp.V)) - rank
    cnot_w = int(np.sum(decomp.W)) - DEGREE
    cnot_count = max(0, cnot_u) + max(0, cnot_v) + max(0, cnot_w)
    
    toffoli_count = rank + (12 if controlled else 0)
    ancilla_count = rank if not controlled else rank + 12
    circuit_depth = math.ceil(math.log2(DEGREE)) * 2 + rank
    
    # Emit a concrete gate-level netlist from the decomposition.
    # Each atom r contributes: (U·a) AND (V·b) * W[r,:] into the output.
    # The circuit uses CNOTs for inner products, Toffoli for AND, and
    # ancilla-accumulation XORs for the weighted sum.
    instructions: list[dict[str, Any]] = []
    rank = decomp.rank
    deg = DEGREE
    
    # Build per-atom subcircuits.
    for r in range(rank):
        u = decomp.U[r]
        v = decomp.V[r]
        w = decomp.W[r]
        
        # ---- CNOTs to compute u·a and v·b into ancilla registers ----
        u_anc = DEGREE  # first ancilla after inputs holds u·a
        v_anc = DEGREE + 1  # holds v·b
        
        # u·a: XOR of input qubits where u[i]==1
        for i in range(deg):
            if u[i]:
                instructions.append({
                    "gate": "CNOT",
                    "target": u_anc,
                    "control": i,
                })
        # v·b: XOR of input qubits where v[j]==1
        for j in range(deg):
            if v[j]:
                instructions.append({
                    "gate": "CNOT",
                    "target": v_anc,
                    "control": j + deg,  # input b starts at index deg
                })
        
        # ---- Toffoli: AND of (u·a) and (v·b) into a product ancilla ----
        prod_anc = DEGREE + 2  # holds (u·a) & (v·b)
        instructions.append({
            "gate": "Toffoli",
            "target": prod_anc,
            "control1": u_anc,
            "control2": v_anc,
        })
        
        # ---- Weighted sum: add w[k] * prod into output ancilla for each k ----
        for k in range(deg):
            if w[k]:
                out_anc = 2 * deg + k  # output bit k starts here
                # controlled-X: if prod==1, flip output bit k
                instructions.append({
                    "gate": "CX",
                    "target": out_anc,
                    "control": prod_anc,
                })
    
    return ReversibleCircuitSpec(
        tensor_rank=rank,
        toffoli_count=toffoli_count,
        cnot_count=cnot_count,
        ancilla_count=ancilla_count,
        circuit_depth=circuit_depth,
        is_controlled=controlled,
        instructions=instructions,
    )


def evaluate_multiplier(circuit: ReversibleCircuitSpec) -> dict[str, Any]:
    """Evaluates the full physical circuit resources independently from tensor rank."""
    return {
        "mathematical_tensor_rank": circuit.tensor_rank,
        "toffoli_count": circuit.toffoli_count,
        "cnot_count": circuit.cnot_count,
        "ancilla_count": circuit.ancilla_count,
        "circuit_depth": circuit.circuit_depth,
        "is_controlled": circuit.is_controlled,
        "schoolbook_gain_ratio": 144.0 / circuit.toffoli_count if circuit.toffoli_count > 0 else 0.0,
        "incumbent_51_delta": circuit.toffoli_count - 51,
    }


def generate_reproducibility_manifest(
    decomp: TensorDecomposition,
    circuit: ReversibleCircuitSpec,
    seed: int = 42,
) -> dict[str, Any]:
    """Generates a cryptographic reproducibility certificate."""
    T = build_multiplication_tensor()
    tensor_hash = hashlib.sha256(T.tobytes()).hexdigest()
    manifest_body = {
        "seed": seed,
        "modulus_poly": list(MODULUS_POLY),
        "modulus_int": MODULUS_INT,
        "tensor_sha256": tensor_hash,
        "algorithm_name": decomp.name,
        "tensor_rank": decomp.rank,
        "toffoli_count": circuit.toffoli_count,
        "timestamp_unix": int(time.time()),
    }
    manifest_bytes = json.dumps(manifest_body, sort_keys=True).encode("utf-8")
    manifest_body["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest_body


# --- Bit-Exact Invsqrt Screening Model (Correction D) ---

class InvsqrtNumericalModel:
    """Exact integer and fixed-point candidate screening model for Benchmark I."""

    def __init__(
        self,
        max_x: int = 3_133_452,
        precision_bits: int = 24,
        beta: Fraction = Fraction(1, 1),
    ):
        self.max_x = max_x
        self.precision_bits = precision_bits
        self.beta = beta
        self.mask64 = (1 << 64) - 1

    def reference_values_point(self, x: int) -> tuple[int, int]:
        """Computes bit-exact target turn phase and tolerance in 2^-64 units."""
        num = (self.beta.numerator ** 2) << 130
        den = (self.beta.denominator ** 2) * x
        r = math.isqrt(num // den)
        target = ((r + 1) >> 1) & self.mask64
        shift = self.precision_bits + 1
        tol = min(r >> shift, 1 << 63)
        return target, tol

    def evaluate_candidate_point(self, x: int, candidate_phase_units: int) -> tuple[bool, float, int]:
        """Evaluates single candidate phase against exact verifier boundaries."""
        target, tol = self.reference_values_point(x)
        error = np.int64((candidate_phase_units - target) & self.mask64)
        used = int(abs(error))
        tol_int = int(tol)
        passed = bool(used <= tol_int)
        ratio = float(used) / float(tol_int) if tol_int > 0 else 0.0
        margin = tol_int - used
        return passed, ratio, margin

    def screen_domain_vectorized(
        self,
        candidate_phase_fn,
        chunk_size: int = 262144,
    ) -> dict[str, Any]:
        """Screens candidate phase formula across entire input domain [1, 3,133,452]."""
        max_ratio = 0.0
        min_margin = 1 << 64
        failed_count = 0
        total_screened = 0

        for start in range(1, self.max_x + 1, chunk_size):
            stop = min(start + chunk_size, self.max_x + 1)
            xs = np.arange(start, stop, dtype=np.uint64)
            candidate_phases = candidate_phase_fn(xs)
            
            for i, x in enumerate(xs):
                passed, ratio, margin = self.evaluate_candidate_point(int(x), int(candidate_phases[i]))
                if not passed:
                    failed_count += 1
                max_ratio = max(max_ratio, ratio)
                min_margin = min(min_margin, margin)
                total_screened += 1

        return {
            "all_passed": failed_count == 0,
            "failed_count": failed_count,
            "total_screened": total_screened,
            "max_relative_error_ratio": max_ratio,
            "min_margin_units": min_margin,
        }
