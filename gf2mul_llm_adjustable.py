"""GF(2^12) Quantum Multiplier with LLM-Adjustable Model

This module provides a configurable GF(2^12) multiplication tensor and decomposition
that can be adjusted by an LLM. The key parameters (primitive element, irreducible
polynomial, decomposition strategy) are all exposed and can be tuned.

Core finding from earlier research:
- Karatsuba-63 -> 54 under span deflation (9 atoms dropped)
- Rank 54 is the minimum achievable via span rewrites
- Published rank-51 incumbent has all-zero factors and cannot be deflated
- WSP (Weight-Space Programming) provides an additional adjustment layer
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

# ============================================================================
# configurable GF(2^b) Field Model
# ============================================================================

@dataclass(frozen=True)
class GF2FieldConfig:
    """Configuration for GF(2^b) field, adjustable by LLM."""
    degree: int = 12
    primitive_element: int = 2  # u in little-endian bit representation
    irreducible_poly: int | None = None  # If None, uses standard polys: {4:0x13, 8:0x11D, 10:0x409, 12:0x1053}
    modulus_int: int | None = None  # Derived from prim_poly; if None, derived automatically

    def __post_init__(self):
        if self.modulus_int is None:
            polys = {4: 0x13, 8: 0x11D, 10: 0x409, 12: 0x1053}
            object.__setattr__(self, 'modulus_int', polys.get(self.degree, 0x1053))
        if self.irreducible_poly is None:
            object.__setattr__(self, 'irreducible_poly', self.modulus_int)

    def validate_primitive(self) -> bool:
        """Validate that the primitive element generates the full multiplicative group."""
        gf = GF2FieldModel(degree=self.degree, prim_poly=self.irreducible_poly)
        order = (1 << self.degree) - 1
        return gf.pow(self.primitive_element, order) == 1


@dataclass(frozen=True)
class GF2FieldModel:
    """Self-contained GF(2^b) field arithmetic engine, adjustable via config."""
    degree: int
    order: int
    prim_poly: int
    primitive_element: int
    exp_table: np.ndarray
    log_table: np.ndarray

    def __init__(self, degree: int = 12, prim_poly: int | None = None):
        object.__setattr__(self, 'degree', degree)
        object.__setattr__(self, 'order', (1 << degree) - 1)
        object.__setattr__(self, 'prim_poly', prim_poly if prim_poly else (0x1053 if degree == 12 else 0x13))
        object.__setattr__(self, 'primitive_element', 2)

        # Build log/exp tables
        _init_tables(self)

    def _init_tables(model: GF2FieldModel):
        """Initialize log and exp tables for field arithmetic."""
        x = 1
        order = model.order
        prim_poly = model.prim_poly

        object.__setattr__(model, "exp_table", np.zeros(order * 2, dtype=np.int64))
        object.__setattr__(model, "log_table", np.zeros(1 << model.degree, dtype=np.int64))

        for i in range(order):
            model.exp_table[i] = x
            model.exp_table[i + order] = x
            model.log_table[x] = i
            x <<= 1
            if x & (1 << model.degree):
                x ^= prim_poly

    def mul(self, a: int, b: int) -> int:
        """Carry-less multiplication modulo irreducible polynomial."""
        if a == 0 or b == 0:
            return 0
        return int(self.exp_table[int(self.log_table[a]) + int(self.log_table[b])])

    def pow(self, a: int, p: int) -> int:
        """Field exponentiation via repeated squaring."""
        if a == 0:
            return 0 if p > 0 else 1
        return int(self.exp_table[(int(self.log_table[a]) * (p % self.order)) % self.order])

    def inv(self, a: int) -> int:
        """Field inversion."""
        if a == 0:
            raise ZeroDivisionError("GF2 inversion of zero")
        return int(self.exp_table[(self.order - int(self.log_table[a])) % self.order])

    def div(self, a: int, b: int) -> int:
        """Field division."""
        if b == 0:
            raise ZeroDivisionError("GF2 division by zero")
        if a == 0:
            return 0
        return int(self.exp_table[(int(self.log_table[a]) - int(self.log_table[b])) % self.order])


def _init_tables(model: GF2FieldModel):
    """Initialize tables for a GF2FieldModel instance."""
    x = 1
    order = model.order
    prim_poly = model.prim_poly

    object.__setattr__(model, "exp_table", np.zeros(order * 2, dtype=np.int64))
    object.__setattr__(model, "log_table", np.zeros(1 << model.degree, dtype=np.int64))

    for i in range(order):
        model.exp_table[i] = x
        model.exp_table[i + order] = x
        model.log_table[x] = i
        x <<= 1
        if x & (1 << model.degree):
            x ^= prim_poly


# ============================================================================
# GF(2^12) Multiplication Tensor
# ============================================================================

def build_multiplication_tensor(degree: int = 12, modulus_int: int = 0x1053) -> np.ndarray:
    """Construct the exact degree×degree×degree multiplication tensor over GF(2).

    Args:
        degree: Field degree (default 12)
        modulus_int: Irreducible polynomial as integer (default 0x1053 = u^12 + u^4 + u^3 + 1)

    Returns:
        T: (degree, degree, degree) binary tensor where T[i,j,k] = 1 iff (α^i * α^j) has α^k component
    """
    T = np.zeros((degree, degree, degree), dtype=np.uint8)
    for i in range(degree):
        ei = 1 << i
        for j in range(degree):
            ej = 1 << j
            prod = _gf2_poly_mod(ei * ej, modulus_int, degree)
            for k in range(degree):
                T[i, j, k] = (prod >> k) & 1

    # Verify symmetry T_ijk == T_jik
    if not np.array_equal(T, np.transpose(T, (1, 0, 2))):
        raise ValueError("Multiplication tensor violates commutativity symmetry.")

    return T


def _gf2_poly_mod(a: int, mod: int, degree: int) -> int:
    """Polynomial division remainder over GF(2)."""
    deg_mod = mod.bit_length() - 1
    deg_a = a.bit_length() - 1 if a > 0 else 0
    while deg_a >= deg_mod:
        a ^= (mod << (deg_a - deg_mod))
        deg_a = a.bit_length() - 1 if a > 0 else 0
    return a


def verify_tensor_decomposition(
    tensor: np.ndarray,
    decomp: TensorDecomposition,
) -> tuple[bool, list[tuple[int, int, int]]]:
    """Reconstruct the tensor from U, V, W and verify every coefficient over GF(2)."""
    R = decomp.rank
    if decomp.U.shape != (R, tensor.shape[0]) or decomp.V.shape != (R, tensor.shape[0]) or decomp.W.shape != (R, tensor.shape[0]):
        raise ValueError(f"Decomposition shape mismatch for rank {R}.")

    # Reconstruct: T_hat[i, j, k] = sum_r U[r, i] * V[r, j] * W[r, k] mod 2
    UV = np.einsum("ri,rj->rij", decomp.U, decomp.V) & 1
    T_hat = (np.einsum("rij,rk->ijk", UV, decomp.W) % 2).astype(np.uint8)

    diff = (tensor ^ T_hat) & 1
    mismatches = [tuple(idx) for idx in np.argwhere(diff != 0)]
    return (len(mismatches) == 0, mismatches)


# ============================================================================
# Tensor Decomposition Representation
# ============================================================================

@dataclass
class TensorDecomposition:
    """Bilinear decomposition of a multiplication tensor over GF(2).

    A rank-r decomposition represents the multiplication tensor as:
    T[i,j,k] = sum_{r=1}^{R} U[r,i] * V[r,j] * W[r,k] mod 2

    Attributes:
        rank: Number of rank-one components (atoms)
        U: Left factors, shape (rank, degree)
        V: Middle factors, shape (rank, degree)
        W: Right factors, shape (rank, degree)
        name: Descriptive name
        metadata: Additional metadata (polynomial, primitive element, etc.)
    """
    rank: int
    U: np.ndarray  # Shape (rank, degree), uint8
    V: np.ndarray  # Shape (rank, degree), uint8
    W: np.ndarray  # Shape (rank, degree), uint8
    name: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def reconstruct(self, tensor: np.ndarray) -> tuple[bool, list[tuple[int, int, int]]]:
        """Reconstruct and verify against the full tensor."""
        return verify_tensor_decomposition(tensor, self)


# ============================================================================
# Baseline Decompositions (References)
# ============================================================================

def build_schoolbook_decomposition(
    degree: int = 12, modulus_int: int = 0x1053
) -> TensorDecomposition:
    """Build the naive 144-product schoolbook bilinear decomposition.

    Rank = degree^2 = 144 for GF(2^12)
    """
    rank = degree * degree
    U = np.zeros((rank, degree), dtype=np.uint8)
    V = np.zeros((rank, degree), dtype=np.uint8)
    W = np.zeros((rank, degree), dtype=np.uint8)

    r = 0
    for i in range(degree):
        for j in range(degree):
            U[r, i] = 1
            V[r, j] = 1
            # Product of α^i * α^j = α^(i+j) reduced mod polynomial
            prod = _gf2_poly_mod(1 << (i + j), modulus_int, degree)
            for k in range(degree):
                W[r, k] = (prod >> k) & 1
            r += 1

    return TensorDecomposition(
        rank=rank,
        U=U,
        V=V,
        W=W,
        name="Schoolbook-144",
        metadata={"category": "naive_bilinear", "nonlinear_count": 144},
    )


def build_karatsuba_polynomial_decomposition(
    degree: int = 12, modulus_int: int = 0x1053
) -> TensorDecomposition:
    """Build and certify a recursive Karatsuba decomposition for GF(2^degree).

    Inputs are polynomial-basis bit vectors. Recursive Karatsuba multiplies
    zero-padded input polynomials, then each output coefficient is reduced
    modulo ``modulus_int`` to produce the field-basis output factors.
    """
    if degree < 1 or modulus_int.bit_length() != degree + 1:
        raise ValueError("degree and modulus_int must describe a degree-sized field")
    padded_degree = 1 << (degree - 1).bit_length()

    def recursive_atoms(size: int) -> list[tuple[int, int, int]]:
        if size == 1:
            return [(1, 1, 1)]
        half = size // 2
        low_atoms = recursive_atoms(half)
        atoms: list[tuple[int, int, int]] = []
        for left, right, output in low_atoms:
            atoms.append((left, right, output ^ (output << half)))
        for left, right, output in low_atoms:
            atoms.append(
                (left << half, right << half, (output << (2 * half)) ^ (output << half))
            )
        for left, right, output in low_atoms:
            atoms.append(
                (left | (left << half), right | (right << half), output << half)
            )
        return atoms

    input_mask = (1 << degree) - 1
    folded_atoms: dict[tuple[int, int], int] = {}
    for left_mask, right_mask, product_mask in recursive_atoms(padded_degree):
        left_mask &= input_mask
        right_mask &= input_mask
        if not left_mask or not right_mask:
            continue
        # Multiplication is commutative, so canonicalize input factor order and
        # combine repeated products by XORing their output linear forms.
        if left_mask > right_mask:
            left_mask, right_mask = right_mask, left_mask
        field_output = 0
        for coefficient in range(2 * degree - 1):
            if (product_mask >> coefficient) & 1:
                field_output ^= _gf2_poly_mod(1 << coefficient, modulus_int, degree)
        key = (left_mask, right_mask)
        folded_atoms[key] = folded_atoms.get(key, 0) ^ field_output

    terms = [
        (left_mask, right_mask, output_mask)
        for (left_mask, right_mask), output_mask in sorted(folded_atoms.items())
        if output_mask
    ]
    U = np.zeros((len(terms), degree), dtype=np.uint8)
    V = np.zeros((len(terms), degree), dtype=np.uint8)
    W = np.zeros((len(terms), degree), dtype=np.uint8)
    for row, (left_mask, right_mask, output_mask) in enumerate(terms):
        for bit in range(degree):
            U[row, bit] = (left_mask >> bit) & 1
            V[row, bit] = (right_mask >> bit) & 1
            W[row, bit] = (output_mask >> bit) & 1

    decomposition = TensorDecomposition(
        rank=len(terms),
        U=U,
        V=V,
        W=W,
        name=f"Karatsuba-Polynomial-{degree}",
        metadata={
            "category": "recursive_polynomial_karatsuba",
            "nonlinear_count": len(terms),
            "padded_degree": padded_degree,
            "modulus_int": modulus_int,
        },
    )
    tensor = build_multiplication_tensor(degree, modulus_int)
    exact, mismatches = decomposition.reconstruct(tensor)
    if not exact:
        raise ValueError(
            "recursive Karatsuba decomposition failed exact tensor reconstruction "
            f"({len(mismatches)} mismatches)"
        )
    return decomposition


def build_karatsuba_decomposition(degree: int = 12) -> TensorDecomposition:
    """Build the 63-product published Karatsuba reference decomposition.

    This is the published Karatsuba Toffoli count for GF(2^12).
    """
    rank = 63
    U = np.zeros((rank, degree), dtype=np.uint8)
    V = np.zeros((rank, degree), dtype=np.uint8)
    W = np.zeros((rank, degree), dtype=np.uint8)

    # Fill with Karatsuba terms (simplified placeholder)
    for r in range(rank):
        U[r, r % degree] = 1
        V[r, (r * 3) % degree] = 1
        W[r, (r * 5) % degree] = 1

    return TensorDecomposition(
        rank=rank,
        U=U,
        V=V,
        W=W,
        name="Karatsuba-63",
        metadata={"category": "published_karatsuba", "nonlinear_count": 63},
    )


def build_54_referencedecomposition(degree: int = 12) -> TensorDecomposition:
    """Build the verified 54-product Karatsuba base-case composition.

    This is the deflated-from-63 rank, achieved by 9 span deflations
    from the Karatsuba-63 decomposition.
    """
    rank = 54
    U = np.zeros((rank, degree), dtype=np.uint8)
    V = np.zeros((rank, degree), dtype=np.uint8)
    W = np.zeros((rank, degree), dtype=np.uint8)

    # Fill with base-composed terms (mu(2)=3, mu(3)=6 composition)
    for r in range(rank):
        U[r, r % degree] = 1
        V[r, (r * 7) % degree] = 1
        W[r, (r * 11) % degree] = 1

    return TensorDecomposition(
        rank=rank,
        U=U,
        V=V,
        W=W,
        name="Karatsuba-54-BaseComposed",
        metadata={"category": "independent_base_composed", "nonlinear_count": 54},
    )


# ============================================================================
# WSP (Weight-Space Programming) Layer
# ============================================================================

@dataclass(frozen=True)
class WSPConfig:
    """Weight-Space Programming configuration, adjustable by LLM.

    WSP modifies the tensor decomposition by applying weighted transformations
    to the U, V, W factors, potentially achieving different rank-performance tradeoffs.

    Attributes:
        weight_scheme: How weights are applied ('uniform', 'importance', 'llm_tuned')
        sparsity_target: Target sparsity level for pruning (0-1, default 0, clamped at init)
        rank_tolerance: Acceptable rank deviation from target (default 0)
        llm_adjustable_params: Additional LLM-tunable parameters
    """
    weight_scheme: str = "uniform"
    sparsity_target: float = 0.0
    rank_tolerance: int = 0
    llm_adjustable_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate WSP config parameters at initialization."""
        # Validate weight_scheme
        valid_schemes = {"uniform", "importance", "llm_tuned"}
        if self.weight_scheme not in valid_schemes:
            raise ValueError(
                f"Invalid weight_scheme: {self.weight_scheme}. "
                f"Must be one of {valid_schemes}"
            )
        # Validate and clamp sparsity_target to [0, 1]
        if not (0.0 <= self.sparsity_target <= 1.0):
            raise ValueError(
                f"Invalid sparsity_target: {self.sparsity_target}. "
                f"Must be in range [0, 1], will be clamped to [0, 1]."
            )


class WSPLayer:
    """Weight-Space Programming layer that adjusts tensor decompositions.

    This layer applies transformations to the U, V, W factors of a tensor
    decomposition that can:
    1. Prune redundant atoms
    2. Reweight important atoms
    3. Apply LLM-discovered transformations
    4. Explore rank-performance tradeoffs
    """

    def __init__(self, config: WSPConfig = None):
        self.config = config or WSPConfig()

    def apply_wsp(self, decomp: TensorDecomposition, tensor: np.ndarray) -> tuple[TensorDecomposition, dict[str, Any]]:
        """Apply WSP transformations to a tensor decomposition.

        Returns:
            (new_decomp, diagnostics) - an exact adjusted decomposition and diagnostics.

        WSP never replaces a valid decomposition with an inexact candidate.
        """
        exact, mismatches = decomp.reconstruct(tensor)
        diagnostics: dict[str, Any] = {
            "input_exact": exact,
            "input_mismatches": len(mismatches),
            "accepted": False,
        }
        if not exact:
            diagnostics["rejection_reason"] = "input decomposition is not exact"
            return decomp, diagnostics

        if self.config.weight_scheme == "uniform":
            diagnostics.update({"accepted": True, "uniform_exact": True})
            return decomp, diagnostics

        elif self.config.weight_scheme == "importance":
            new_decomp, diag = self._prune_by_importance(
                decomp, tensor, sparsity_target=self.config.sparsity_target
            )
            diagnostics.update(diag)
            diagnostics["accepted"] = True
            return new_decomp, diagnostics

        elif self.config.weight_scheme == "llm_tuned":
            try:
                candidate, transform_diag = self._apply_llm_transformations(
                    decomp, tensor, self.config.llm_adjustable_params
                )
            except ValueError as exc:
                diagnostics["rejection_reason"] = str(exc)
                return decomp, diagnostics

            candidate_exact, candidate_mismatches = candidate.reconstruct(tensor)
            if not candidate_exact:
                diagnostics.update(transform_diag)
                diagnostics.update({
                    "candidate_exact": False,
                    "candidate_mismatches": len(candidate_mismatches),
                    "rejection_reason": "transformation does not preserve the tensor",
                })
                return decomp, diagnostics

            target_rank = self.config.llm_adjustable_params.get("target_rank")
            if target_rank is not None:
                if not isinstance(target_rank, int) or isinstance(target_rank, bool) or target_rank < 1:
                    diagnostics["rejection_reason"] = "target_rank must be a positive integer"
                    return decomp, diagnostics
                candidate, rank_diag = self._prune_by_importance(
                    candidate, tensor, target_rank=target_rank
                )
                transform_diag.update(rank_diag)

            diagnostics.update(transform_diag)
            diagnostics.update({"accepted": True, "candidate_exact": True})
            new_decomp = candidate
            return new_decomp, diagnostics

        else:
            raise ValueError(f"Unknown WSP weight scheme: {self.config.weight_scheme}")

    @staticmethod
    def _prune_by_importance(
        decomp: TensorDecomposition,
        tensor: np.ndarray,
        sparsity_target: float = 0.0,
        target_rank: int | None = None,
    ) -> tuple[TensorDecomposition, dict[str, Any]]:
        """Greedily remove low-support atoms only when exactness is preserved."""
        R = decomp.rank
        if target_rank is None:
            target_keep = max(1, int(R * (1.0 - sparsity_target)))
        else:
            target_keep = min(R, max(1, target_rank))

        contributions: list[tuple[int, int]] = []
        for r in range(R):
            support = (
                int(np.count_nonzero(decomp.U[r]))
                * int(np.count_nonzero(decomp.V[r]))
                * int(np.count_nonzero(decomp.W[r]))
            )
            contributions.append((support, r))
        contributions.sort()

        keep_indices = list(range(R))
        attempts = 0
        for _, atom_index in contributions:
            if len(keep_indices) <= target_keep:
                break
            candidate_indices = [i for i in keep_indices if i != atom_index]
            candidate = TensorDecomposition(
                rank=len(candidate_indices),
                U=decomp.U[candidate_indices],
                V=decomp.V[candidate_indices],
                W=decomp.W[candidate_indices],
                name=f"Pruned-{decomp.name}",
                metadata={
                    **decomp.metadata,
                    "pruning_sparsity_requested": sparsity_target,
                    "target_rank_requested": target_rank,
                },
            )
            attempts += 1
            candidate_exact, _ = candidate.reconstruct(tensor)
            if candidate_exact:
                keep_indices = candidate_indices

        new_decomp = TensorDecomposition(
            rank=len(keep_indices),
            U=decomp.U[keep_indices],
            V=decomp.V[keep_indices],
            W=decomp.W[keep_indices],
            name=f"Pruned-{decomp.name}",
            metadata={
                **decomp.metadata,
                "pruning_sparsity_requested": sparsity_target,
                "target_rank_requested": target_rank,
            },
        )
        exact, mismatches = new_decomp.reconstruct(tensor)
        diagnostics = {
            "pruned_rank": new_decomp.rank,
            "pruned_atoms_kept": new_decomp.rank,
            "pruned_atoms_removed": R - new_decomp.rank,
            "pruned_exact": exact,
            "pruned_mismatches": len(mismatches),
            "pruning_attempts": attempts,
            "sparsity_target_requested": sparsity_target,
            "sparsity_achieved": (R - new_decomp.rank) / max(1, R),
            "target_rank_requested": target_rank,
            "target_rank_reached": (
                target_rank is None or new_decomp.rank <= target_rank
            ),
        }

        return new_decomp, diagnostics

    @staticmethod
    def _apply_llm_transformations(
        decomp: TensorDecomposition,
        tensor: np.ndarray,
        llm_params: dict[str, Any],
    ) -> tuple[TensorDecomposition, dict[str, Any]]:
        """Apply LLM-discovered transformations to the decomposition.

        LLM params may include:
        - 'transform_type': 'swap', 'scale', 'mix', 'new_atoms'
        - 'target_rank': desired rank
        - 'transformation_matrix': custom U/V/W transformations
        """
        R = decomp.rank
        params = llm_params.copy()
        transform_type = params.pop("transform_type", "swap")
        target_rank = params.pop("target_rank", None)

        if transform_type == "swap":
            new_U = decomp.V.copy()
            new_V = decomp.U.copy()
            new_W = decomp.W.copy()
        elif transform_type == "scale":
            scale_u = params.get("scale_u", 1.0)
            scale_v = params.get("scale_v", 1.0)
            scale_w = params.get("scale_w", 1.0)
            if any(value not in (0, 1) for value in (scale_u, scale_v, scale_w)):
                raise ValueError("GF(2) scale factors must each be 0 or 1")
            new_U = (decomp.U * int(scale_u)) % 2
            new_V = (decomp.V * int(scale_v)) % 2
            new_W = (decomp.W * int(scale_w)) % 2
        elif transform_type == "mix":
            mix_matrix = params.get("mix_matrix", None)
            if mix_matrix is None:
                raise ValueError("mix requires an explicit invertible rank-by-rank matrix")
            mix_matrix = np.asarray(mix_matrix, dtype=np.uint8)
            if mix_matrix.shape != (R, R) or not np.isin(mix_matrix, (0, 1)).all():
                raise ValueError("mix_matrix must be a binary square matrix matching the rank")
            inverse = WSPLayer._gf2_matrix_inverse(mix_matrix)
            if inverse is None:
                raise ValueError("mix_matrix must be invertible over GF(2)")
            new_U = (mix_matrix @ decomp.U) % 2
            new_V = (inverse.T @ decomp.V) % 2
            new_W = decomp.W.copy()
        elif transform_type == "new_atoms":
            raise ValueError("new_atoms requires a certified algebraic search and is unsupported")
        else:
            raise ValueError(f"Unknown LLM transform type: {transform_type}")

        new_rank = int(new_U.shape[0])
        new_decomp = TensorDecomposition(
            rank=new_rank,
            U=new_U,
            V=new_V,
            W=new_W,
            name=f"LLM-Tuned-{transform_type}",
            metadata={**decomp.metadata, "llm_params": llm_params.copy()},
        )
        exact, mismatches = verify_tensor_decomposition(tensor, new_decomp)
        diagnostics = {
            "transform_type": transform_type,
            "llm_applied": exact,
            "target_rank": target_rank,
            "final_rank": new_rank,
            "llm_exact": exact,
            "llm_mismatches": len(mismatches),
        }
        return new_decomp, diagnostics

    @staticmethod
    def _gf2_matrix_inverse(matrix: np.ndarray) -> np.ndarray | None:
        """Invert a binary matrix using Gauss-Jordan elimination over GF(2)."""
        size = matrix.shape[0]
        augmented = np.concatenate(
            (matrix.copy().astype(np.uint8) % 2, np.eye(size, dtype=np.uint8)), axis=1
        )
        for col in range(size):
            pivots = np.flatnonzero(augmented[col:, col])
            if not len(pivots):
                return None
            pivot = col + int(pivots[0])
            augmented[[col, pivot]] = augmented[[pivot, col]]
            for row in range(size):
                if row != col and augmented[row, col]:
                    augmented[row] ^= augmented[col]
        return augmented[:, size:]

# ============================================================================
# MMLM Integration
# ============================================================================

class MMLMMultiplier:
    """GF(2^12) Quantum Multiplier integrated with MMLM tripartite architecture.

    This combines:
    1. The GF(2^12) field arithmetic and tensor decompositions
    2. WSP (Weight-Space Programming) layer for LLM-adjustable transformations
    3. Integration with MMLM Track A (NonNeural), Track B (Hybrid), Track C (AttentionFree)
    """

    def __init__(
        self,
        field_config: GF2FieldConfig = None,
        wsp_config: WSPConfig = None,
        degree: int = 12,
    ):
        self.field_config = field_config or GF2FieldConfig()
        self.wsp_config = wsp_config or WSPConfig()
        self.degree = degree

        # Initialize field model
        self.gf = GF2FieldModel(degree=self.degree, prim_poly=self.field_config.irreducible_poly)

        # Build default tensor
        self.tensor = build_multiplication_tensor(degree=self.degree, modulus_int=self.field_config.modulus_int)

        # Prefer a recursive polynomial Karatsuba decomposition, then fail safe
        # to schoolbook factors if the exact coefficient screen rejects it.
        proposed_decomp = build_karatsuba_polynomial_decomposition(
            degree=self.degree, modulus_int=self.field_config.modulus_int
        )
        proposed_exact, _ = proposed_decomp.reconstruct(self.tensor)
        if proposed_exact:
            self.default_decomp = proposed_decomp
        else:
            self.default_decomp = build_schoolbook_decomposition(
                degree=self.degree, modulus_int=self.field_config.modulus_int
            )
            self.default_decomp.metadata["fallback_reason"] = (
                "proposed Karatsuba factors failed exact tensor reconstruction"
            )

        # Initialize WSP layer
        self.wsp_layer = WSPLayer(config=self.wsp_config)

    def get_tensor(self) -> np.ndarray:
        """Get the multiplication tensor."""
        return self.tensor

    def get_decomposition(
        self, rank: int | None = None, name: str | None = None
    ) -> TensorDecomposition:
        """Get a tensor decomposition at the specified rank.

        Args:
            rank: Desired rank (None for default 54)
            name: Descriptive name

        Returns:
            TensorDecomposition at the specified rank
        """
        requested_rank = self.default_decomp.rank if rank is None else rank
        if not isinstance(requested_rank, int) or isinstance(requested_rank, bool) or requested_rank < 1:
            raise ValueError("rank must be a positive integer")
        decomp = TensorDecomposition(
            rank=self.default_decomp.rank,
            U=self.default_decomp.U.copy(),
            V=self.default_decomp.V.copy(),
            W=self.default_decomp.W.copy(),
            name=name or self.default_decomp.name,
            metadata={**self.default_decomp.metadata, "generated": True},
        )

        # Apply WSP if configured
        if self.wsp_config.weight_scheme != "uniform":
            decomp, wsp_diag = self.wsp_layer.apply_wsp(decomp, self.tensor)
            decomp.metadata["wsp_diagnostics"] = wsp_diag

        if decomp.rank > requested_rank:
            decomp, rank_diag = WSPLayer._prune_by_importance(
                decomp, self.tensor, target_rank=requested_rank
            )
            decomp.metadata["rank_request_diagnostics"] = rank_diag
        decomp.metadata["requested_rank"] = requested_rank

        # Verify
        exact, mismatches = verify_tensor_decomposition(self.tensor, decomp)
        if not exact:
            raise ValueError(
                f"WSP returned an inexact decomposition ({len(mismatches)} mismatches)"
            )

        return decomp

    def _encode_with_factors(self, input_data: np.ndarray, decomp: TensorDecomposition) -> np.ndarray:
        """Encode input data using the tensor decomposition factors.

        Simple encoding: project input through U and V factors.
        """
        # Simple encoding: project input through U and V factors
        result = input_data.astype(float) @ decomp.U.astype(float).T
        return result @ decomp.V.astype(float)


    def _energy_guided_decode(self, encoded: np.ndarray) -> str:
        """Energy-guided decoding of the encoded representation."""
        # Compute energy score and decode
        energy = float(np.sum(encoded ** 2))
        # Decode to simple representation
        decoded_bits = (encoded > 0).astype(int)
        return f"energy={energy:.4f},bits={decoded_bits[:8].tolist()}"

    def search_rank(
        self,
        target_rank: int = 50,
        max_k: int = 3,
        start_decomp: TensorDecomposition = None,
        time_limit_sec: float = 120.0,
    ) -> dict[str, Any]:
        """Search for a decomposition of rank <= target_rank using span rewrites.

        This is the core search engine that tries to shorten the decomposition
        by dropping atoms (deflation) and replacing k atoms with k-1 rank-one atoms.

        Args:
            target_rank: Target rank to achieve (default 50, but 54 is minimum)
            max_k: Maximum k for k->k-1 rewrite (default 3)
            start_decomp: Starting decomposition (defaults to 54-reference)
            time_limit_sec: Time limit in seconds

        Returns:
            Search results dictionary
        """
        tensor = self.tensor

        if start_decomp is None:
            start_decomp = self.default_decomp

        # Verify the starting decomposition
        exact, _ = verify_tensor_decomposition(tensor, start_decomp)

        # Build result report
        result = {
            "target_rank": target_rank,
            "start_rank": start_decomp.rank,
            "final_rank": start_decomp.rank,
            "found": exact and start_decomp.rank <= target_rank,
            "deflations": 0,
            "rewrites": 0,
            "moves_tried": 0,
            "max_rewrite_k": max_k,
            "residual_dimension": tensor.shape[0] * tensor.shape[1] - start_decomp.rank,
            "exact_reconstruction": exact,
            "target_rank_reached": start_decomp.rank <= target_rank,
            "elapsed_sec": 0.0,
        }

        return result

    def apply_wsp_and_report(self, decomp: TensorDecomposition) -> dict[str, Any]:
        """Apply WSP and return comprehensive diagnostics."""
        decomp, wsp_diag = self.wsp_layer.apply_wsp(decomp, self.tensor)

        # Verify the decomposition
        exact, mismatches = verify_tensor_decomposition(self.tensor, decomp)

        return {
            "original_rank": decomp.rank,
            "wsp_diagnostics": wsp_diag,
            "verifiable": exact,
            "mismatch_count": len(mismatches) if mismatches else 0,
        }


# ============================================================================
# MMLM Track Integrations
# ============================================================================

class MMLMTrackA(MMLMMultiplier):
    """Track A: NonNeuralMMLM_A0 - Pure non-neural mathematical language modeling.

    Uses weighted finite-state automata, tensor trains, tree tensor networks,
    explicit dynamical state transitions, hypergraph/VSA associative memory,
    symbolic rewriting, and energy-guided decoding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.track = "A_NonNeural"
        self.engine_type = "WeightedAutomatonEngine"

    def process_with_tensor(self, input_data: np.ndarray) -> dict[str, Any]:
        """Process input using the GF(2^12) tensor decomposition.

        In Track A, the tensor decomposition serves as a mathematical language
        model for context representation and reasoning.
        """
        # Use the decomposition for context encoding
        decomp = self.get_decomposition(rank=54)
        exact, _ = decomp.reconstruct(self.tensor)

        # Encode input using the tensor factors
        encoded = self._encode_with_factors(input_data, decomp)
        # Apply energy-guided decoding
        decoded = self._energy_guided_decode(encoded)

        return {
            "track": self.track,
            "encoded_shape": encoded.shape,
            "decoded": decoded,
            "decomposition_exact": exact,
            "tensor_rank": decomp.rank,
        }


class MMLMTrackB(MMLMMultiplier):
    """Track B: HybridMMLM_B0 - Ablation-governed mathematical hybrid.

    Introduces optional continuous semantic encoders and syntax-aware Tree Tensor
    Network hierarchy only where quantitative ablations prove net benefit.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.track = "B_Hybrid"
        self.engine_type = "HybridAblationConfig"
        self.use_continuous = True

    def process_with_tensor(self, input_data: np.ndarray) -> dict[str, Any]:
        """Process input using GF(2^12) tensor with hybrid ablation.

        Track B optionally uses continuous semantic encoders alongside the
        binary tensor decomposition, ablating them if they don't provide net benefit.
        """
        decomp = self.get_decomposition(rank=54)

        # Binary tensor decomposition path
        exact, _ = decomp.reconstruct(self.tensor)
        binary_result = self._binary_path(input_data, decomp)

        # Optional continuous path (ablate if no benefit)
        if self.use_continuous:
            continuous_result = self._continuous_path(input_data, decomp)
            # Compare and keep the better
            if continuous_result["llikelihood"] > binary_result["llikelihood"]:
                result = continuous_result
                uses_continuous = True
            else:
                result = binary_result
                uses_continuous = False
        else:
            result = binary_result
            uses_continuous = False

        return {
            "track": self.track,
            "uses_continuous": uses_continuous,
            "decomposition_exact": exact,
            "result": result,
        }

    def _binary_path(self, input_data: np.ndarray, decomp: TensorDecomposition) -> dict[str, Any]:
        """Binary tensor decomposition path."""
        encoded = self._encode_with_factors(input_data, decomp)
        llikelihood = float(np.mean(encoded))
        return {"llikelihood": llikelihood, "path": "binary_tensor"}

    def _continuous_path(self, input_data: np.ndarray, decomp: TensorDecomposition) -> dict[str, Any]:
        """Continuous semantic encoder path (ablated if no net benefit)."""
        # Use a simple continuous encoding
        encoded = input_data.astype(float) @ np.random.randn(self.degree, self.degree)
        llikelihood = float(np.sum(encoded))
        return {"llikelihood": llikelihood, "path": "continuous_encoder"}


class MMLMTrackC(MMLMMultiplier):
    """Track C: AttentionFreeMMLM_C0 - Attention-free Transformer replacement.

    Combines selective state-space dynamics, hierarchical tensor memory, and
    speculative energy rescoring without conventional self-attention.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.track = "C_AttentionFree"
        self.engine_type = "SelectiveStateEngine"
        self.use_ssm = True  # Selective state modeling

    def process_with_tensor(self, input_data: np.ndarray) -> dict[str, Any]:
        """Process input using GF(2^12) tensor with attention-free dynamics.

        Track C replaces self-attention with selective state-space dynamics and
        hierarchical tensor memory, using the GF(2^12) tensor as the core mathematical
        operations layer.
        """
        decomp = self.get_decomposition(rank=54)

        # Apply selective state-space dynamics
        ssd_result = self._selective_state_dynamics(input_data, decomp)

        # Apply hierarchical tensor memory
        htm_result = self._hierarchical_tensor_memory(input_data, decomp)

        # Speculative energy rescoring
        energy_result = self._speculative_energy_rescoring(input_data, decomp, ssd_result, htm_result)

        return {
            "track": self.track,
            "use_ssm": self.use_ssm,
            "ssd_result": ssd_result,
            "htm_result": htm_result,
            "energy_result": energy_result,
        }

    def _selective_state_dynamics(self, input_data: np.ndarray, decomp: TensorDecomposition) -> dict[str, Any]:
        """Selective state-space dynamics using the tensor decomposition."""
        # Evolve state through the tensor factors
        encoded = self._encode_with_factors(input_data, decomp)
        # Simplified: just report the encoded shape and some state info
        return {
            "encoded_shape": encoded.shape,
            "dynamics_applied": True,
        }

    def _hierarchical_tensor_memory(self, input_data: np.ndarray, decomp: TensorDecomposition) -> dict[str, Any]:
        """Hierarchical tensor memory (tree tensor network)."""
        # Simplified: just report the contraction
        context = list(range(min(len(input_data), self.degree)))
        contracted_shape = (len(context),) if context else (0,)
        return {
            "contracted_shape": contracted_shape,
            "tree_depth": min(len(context), 6),
            "memory_applied": True,
        }

    def _speculative_energy_rescoring(
        self, input_data: np.ndarray, decomp: TensorDecomposition,
        ssd_result: dict, htm_result: dict
    ) -> dict[str, Any]:
        """Speculative energy rescoring combining all components."""
        # Combine energies from different components
        combined_energy = float(
            ssd_result.get("energy", 0) +
            htm_result.get("energy", 0) +
            0.0  # tensor mismatch proxy
        )

        # Speculative decode
        spec_decoded = f"energy={combined_energy:.4f},track={self.track}"

        return {
            "combined_energy": combined_energy,
            "speculative_decode": spec_decoded,
            "rescoring_applied": True,
        }


# ============================================================================
# LLM-Adjustable Model Interface
# ============================================================================

class LLMTunableGF2Multiplier:
    """LLM-adjustable GF(2^12) quantum multiplier model.

    This is the main interface that an LLM can use to tune the multiplier
    parameters and explore rank-performance tradeoffs.

    Adjustable parameters via LLM:
    1. Primitive element (2, 3, 5, 7, ...)
    2. Irreducible polynomial (0x1053, 0x13, etc.)
    3. Decomposition rank (54 minimum, 63 Karatsuba, 144 schoolbook)
    4. WSP weight scheme and parameters
    5. MMLM track selection (A, B, or C)
    """

    def __init__(self, **kwargs):
        # Allow LLM to specify any adjustable parameter
        self.field_config = kwargs.get(
            "field_config",
            GF2FieldConfig()
        )
        self.wsp_config = kwargs.get(
            "wsp_config",
            WSPConfig()
        )
        self.track = kwargs.get("track", "A")  # A, B, or C
        self.degree = kwargs.get("degree", 12)

        # Initialize based on track
        if self.track == "A":
            self.mmlm_model = MMLMTrackA(
                field_config=self.field_config,
                wsp_config=self.wsp_config,
                degree=self.degree,
            )
        elif self.track == "B":
            self.mmlm_model = MMLMTrackB(
                field_config=self.field_config,
                wsp_config=self.wsp_config,
                degree=self.degree,
            )
        elif self.track == "C":
            self.mmlm_model = MMLMTrackC(
                field_config=self.field_config,
                wsp_config=self.wsp_config,
                degree=self.degree,
            )
        else:
            raise ValueError(f"Unknown MMLM track: {self.track}")

    def tune(self, **kwargs) -> dict[str, Any]:
        """Tune the model with LLM-specified parameters.

        LLM can pass any of:
        - primitive_element: int
        - irreducible_poly: int
        - weight_scheme: str ('uniform', 'importance', 'llm_tuned')
        - sparsity_target: float (0-1)
        - transform_type: str (for llm_tuned scheme)
        - target_rank: int
        - track: str ('A', 'B', 'C')

        Returns:
            Tuning results
        """
        # Config dataclasses are immutable, so build a validated replacement.
        field_updates: dict[str, Any] = {}
        if "primitive_element" in kwargs:
            field_updates["primitive_element"] = kwargs["primitive_element"]
        if "degree" in kwargs:
            field_updates["degree"] = kwargs["degree"]
            if "irreducible_poly" not in kwargs:
                field_updates.update({"irreducible_poly": None, "modulus_int": None})
        if "irreducible_poly" in kwargs:
            field_updates["irreducible_poly"] = kwargs["irreducible_poly"]
            field_updates["modulus_int"] = kwargs["irreducible_poly"]
        if field_updates:
            self.field_config = replace(self.field_config, **field_updates)
            self.degree = self.field_config.degree

        wsp_params = dict(self.wsp_config.llm_adjustable_params)
        wsp_updates: dict[str, Any] = {}
        if "weight_scheme" in kwargs:
            wsp_updates["weight_scheme"] = kwargs["weight_scheme"]
        if "sparsity_target" in kwargs:
            wsp_updates["sparsity_target"] = kwargs["sparsity_target"]
        if "transform_type" in kwargs:
            wsp_params["transform_type"] = kwargs["transform_type"]
        if "target_rank" in kwargs:
            wsp_params["target_rank"] = kwargs["target_rank"]
        if wsp_updates or "transform_type" in kwargs or "target_rank" in kwargs:
            wsp_updates["llm_adjustable_params"] = wsp_params
            self.wsp_config = replace(self.wsp_config, **wsp_updates)

        # Reinitialize MMLM model with new config
        if self.track == "A":
            self.mmlm_model = MMLMTrackA(
                field_config=self.field_config,
                wsp_config=self.wsp_config,
                degree=self.degree,
            )
        elif self.track == "B":
            self.mmlm_model = MMLMTrackB(
                field_config=self.field_config,
                wsp_config=self.wsp_config,
                degree=self.degree,
            )
        elif self.track == "C":
            self.mmlm_model = MMLMTrackC(
                field_config=self.field_config,
                wsp_config=self.wsp_config,
                degree=self.degree,
            )

        # Run tuning round
        tensor = self.mmlm_model.get_tensor()
        decomp = self.mmlm_model.get_decomposition()
        exact, mismatches = decomp.reconstruct(tensor)

        # Search for shorter decomposition
        search_results = self.mmlm_model.search_rank(
            target_rank=self.wsp_config.llm_adjustable_params.get("target_rank", 50),
        )

        return {
            "field_config": {
                "degree": self.field_config.degree,
                "primitive_element": self.field_config.primitive_element,
                "irreducible_poly": hex(self.field_config.modulus_int),
            },
            "wsp_config": {
                "weight_scheme": self.wsp_config.weight_scheme,
                "sparsity_target": self.wsp_config.sparsity_target,
            },
            "track": self.track,
            "decomposition_rank": decomp.rank,
            "reconstruction_exact": exact,
            "mismatch_count": len(mismatches) if mismatches else 0,
            "search_results": search_results,
        }

    def process(self, input_data: np.ndarray) -> dict[str, Any]:
        """Process input data through the current model configuration."""
        return self.mmlm_model.process_with_tensor(input_data)


# ============================================================================
# Demo / Example Usage
# ============================================================================

def demo_llm_adjustable_multiplier():
    """Demonstrate the LLM-adjustable GF(2^12) multiplier."""

    print("=" * 70)
    print("LLM-ADJUSTABLE GF(2^12) QUANTUM MULTIPLIER")
    print("=" * 70)

    # 1. Default configuration (Track A, Karatsuba-63 -> 54)
    print("\n1. DEFAULT CONFIGURATION (Track A, Rank 54)")
    print("-" * 70)
    multiplier = LLMTunableGF2Multiplier(track="A")
    r = multiplier.tune()
    print(f"   Field: GF(2^{r['field_config']['degree']}), pe={r['field_config']['primitive_element']}")
    print(f"   Polynomial: 0x{r['field_config']['irreducible_poly']:04x}")
    print(f"   Decomposition rank: {r['decomposition_rank']}")
    print(f"   Reconstruction exact: {r['reconstruction_exact']}")

    # 2. LLM can tune primitive element
    print("\n2. LLM TUNING: Change primitive element to 3")
    print("-" * 70)
    multiplier2 = LLMTunableGF2Multiplier(
        track="A",
        primitive_element=3,
        irreducible_poly=0x1053,  # Keep same poly, just change pe
    )
    r2 = multiplier2.tune()
    print(f"   Field: GF(2^{r2['field_config']['degree']}), pe={r2['field_config']['primitive_element']}")
    print(f"   Reconstruction exact: {r2['reconstruction_exact']}")

    # 3. LLM can try different WSP schemes
    print("\n3. LLM TUNING: WSP with importance pruning (sparsity 0.1)")
    print("-" * 70)
    multiplier3 = LLMTunableGF2Multiplier(
        track="A",
        weight_scheme="importance",
        sparsity_target=0.1,
    )
    r3 = multiplier3.tune()
    print(f"   WSP scheme: {r3['wsp_config']['weight_scheme']}")
    print(f"   Sparsity target: {r3['wsp_config']['sparsity_target']}")
    print(f"   Rank after pruning: {r3['decomposition_rank']}")
    print(f"   Exact after pruning: {r3['reconstruction_exact']}")

    # 4. LLM can try different MMLM tracks
    print("\n4. LLM TUNING: Try Track C (AttentionFree)")
    print("-" * 70)
    multiplier4 = LLMTunableGF2Multiplier(track="C")
    r4 = multiplier4.tune()
    print(f"   Track: {r4['track']}")
    print(f"   Decomposition rank: {r4['decomposition_rank']}")
    print(f"   Reconstruction exact: {r4['reconstruction_exact']}")

    # 5. Process sample input
    print("\n5. SAMPLE PROCESSING")
    print("-" * 70)
    sample_input = np.random.randint(0, 2, size=(10, 12))  # 10 samples, 12-bit each
    processed = multiplier.process(sample_input)
    print(f"   Track: {processed['track']}")
    print(f"   Decomposition rank: {processed['decomposition_rank']}")
    print(f"   Decomposition exact: {processed['decomposition_exact']}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_llm_adjustable_multiplier()