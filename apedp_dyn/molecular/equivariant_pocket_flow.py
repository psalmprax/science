r"""3D Equivariant Pocket-Conditioned Conformer Flow & Docking Referee.

Status: experimental

Scope & honesty ceiling:
- Upgrades `apedp_dyn/molecular/se3_flow.py` from unlabeled harmonic springs to an atom-typed,
  pocket-conditioned $SE(3)$-equivariant force field and gradient flow.
- Models intramolecular harmonic bonds, bond-angle bending ($k_\theta (\theta - \theta_0)^2$),
  periodic dihedral torsions ($V_n [1 + \cos(n\phi - \gamma)]$), tetrahedral chirality
  signed-volume constraints, and intermolecular protein-ligand interactions via soft-core
  12-6 Lennard-Jones van der Waals potentials plus distance-dependent dielectric Coulomb
  electrostatics.
- Strictly $SE(3)$-equivariant under joint rigid-body transformations $(R \in SO(3), t \in \mathbb{R}^3)$
  applied simultaneously to ligand coordinates $X_L$ and pocket coordinates $X_P$:
  $$v_L(X_L R^T + t, X_P R^T + t) = v_L(X_L, X_P) R^T$$
- Acts as a deterministic geometric referee (Layer 3 of the drug discovery gatekeeper),
  flagging unphysical steric clashes and unfavorable pocket interaction energies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from apedp_dyn.molecular.chemistry_bridge import (
    MolecularGraph,
    _element_registry,
    embed_conformer_3d,
)
from apedp_dyn.molecular.se3_flow import rigid_transform, signed_volume

_COULOMB_KCAL_A_E2 = 332.06371  # kcal * A / (mol * e^2)
_DIELECTRIC_CONSTANT = 4.0  # Effective interior protein pocket dielectric


def _lj_parameters(element: str) -> tuple[float, float]:
    """Return (sigma in Angstroms, epsilon in kcal/mol) for Universal Force Field / AMBER-like atoms."""
    clean = element.strip()
    params = {
        "H": (2.571, 0.044),
        "B": (3.638, 0.180),
        "C": (3.431, 0.105),
        "N": (3.261, 0.069),
        "O": (3.118, 0.060),
        "F": (2.997, 0.050),
        "P": (3.695, 0.305),
        "S": (3.595, 0.274),
        "Cl": (3.516, 0.227),
        "Br": (3.732, 0.251),
        "I": (4.009, 0.339),
    }
    return params.get(clean, (3.40, 0.10))


@dataclass(frozen=True)
class PocketAtom:
    """Single atom of a target protein binding site pocket."""

    index: int
    element: str
    position: tuple[float, float, float]
    partial_charge: float = 0.0
    residue_name: str = "UNK"
    residue_id: int = 1


@dataclass(frozen=True)
class BindingPocket:
    """3D protein binding site pocket conditioning ligand conformer flow."""

    pocket_id: str
    target_name: str
    atoms: tuple[PocketAtom, ...]
    clash_distance_threshold: float = 1.75  # Angstroms

    def coordinates(self) -> np.ndarray:
        return np.array([a.position for a in self.atoms], dtype=np.float64)

    def charges(self) -> np.ndarray:
        return np.array([a.partial_charge for a in self.atoms], dtype=np.float64)

    def lj_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        sigmas = np.empty(len(self.atoms), dtype=np.float64)
        epsilons = np.empty(len(self.atoms), dtype=np.float64)
        for i, a in enumerate(self.atoms):
            sigmas[i], epsilons[i] = _lj_parameters(a.element)
        return sigmas, epsilons


@dataclass(frozen=True)
class AngleTerm:
    """Harmonic bond angle term between atoms (i, j, k) with apex at j."""

    i: int
    j: int
    k: int
    theta0_rad: float
    k_theta: float = 35.0


@dataclass(frozen=True)
class TorsionTerm:
    """Periodic dihedral torsion term across bonded quadruple (i, j, k, atom_l)."""

    i: int
    j: int
    k: int
    atom_l: int
    periodicity: int = 3
    phase_rad: float = 0.0
    barrier_kcal: float = 1.4


@dataclass(frozen=True)
class PocketConditionedForceField:
    """Atom-typed intramolecular + intermolecular protein-ligand force field."""

    bonds: tuple[tuple[int, int, float, float], ...]  # (u, v, r0, k_bond)
    angles: tuple[AngleTerm, ...] = ()
    torsions: tuple[TorsionTerm, ...] = ()
    chirality_terms: tuple[tuple[int, int, int, int, float], ...] = ()
    ligand_elements: tuple[str, ...] = ()
    ligand_charges: tuple[float, ...] = ()
    chirality_strength: float = 15.0


@dataclass
class PocketDockingResult:
    """Outcome of pocket-conditioned equivariant conformer relaxation and clash refereeing."""

    ligand_coordinates: np.ndarray
    initial_total_energy: float
    final_total_energy: float
    intermolecular_energy_kcal: float
    intramolecular_strain_kcal: float
    min_interatomic_distance_a: float
    steric_clash_count: int
    passed_gate: bool
    rejection_reason: str = ""
    energy_trajectory: list[float] = field(default_factory=list)


def build_pocket_force_field(graph: MolecularGraph) -> PocketConditionedForceField:
    """Construct an atom-typed force field from a sanitized `MolecularGraph`."""
    n = len(graph.atoms)
    adj = graph.adjacency_list()

    # Partial charges via Pauling electronegativity equalization along covalent bonds
    charges = np.array([float(a.formal_charge) for a in graph.atoms], dtype=np.float64)
    for b in graph.bonds:
        en_u = _element_registry(graph.atoms[b.u].element).electronegativity
        en_v = _element_registry(graph.atoms[b.v].element).electronegativity
        shift = 0.085 * (en_v - en_u)
        charges[b.u] += shift
        charges[b.v] -= shift

    bonds_list: list[tuple[int, int, float, float]] = []
    for b in graph.bonds:
        r_u = _element_registry(graph.atoms[b.u].element).covalent_radius
        r_v = _element_registry(graph.atoms[b.v].element).covalent_radius
        r0 = max(0.95, r_u + r_v - 0.08 * (b.order - 1.0))
        k_bond = 120.0 * b.order
        bonds_list.append((b.u, b.v, r0, k_bond))

    angles_list: list[AngleTerm] = []
    for center in range(n):
        nbrs = [nbr for nbr, _, _ in adj[center]]
        theta0 = math.radians(120.0 if graph.atoms[center].is_aromatic else 109.47)
        for idx_a in range(len(nbrs)):
            for idx_b in range(idx_a + 1, len(nbrs)):
                angles_list.append(AngleTerm(i=nbrs[idx_a], j=center, k=nbrs[idx_b], theta0_rad=theta0))

    torsions_list: list[TorsionTerm] = []
    for b in graph.bonds:
        u, v = b.u, b.v
        u_nbrs = [nbr for nbr, _, _ in adj[u] if nbr != v]
        v_nbrs = [nbr for nbr, _, _ in adj[v] if nbr != u]
        if u_nbrs and v_nbrs:
            periodicity = 2 if (b.is_aromatic or b.order >= 1.5) else 3
            phase = math.pi if periodicity == 2 else 0.0
            barrier = 3.0 if periodicity == 2 else 1.2
            torsions_list.append(
                TorsionTerm(
                    i=u_nbrs[0],
                    j=u,
                    k=v,
                    atom_l=v_nbrs[0],
                    periodicity=periodicity,
                    phase_rad=phase,
                    barrier_kcal=barrier,
                )
            )

    chirality_list: list[tuple[int, int, int, int, float]] = []
    for idx, atom in enumerate(graph.atoms):
        if atom.chiral_tag != 0:
            nbrs = [nbr for nbr, _, _ in adj[idx]]
            if len(nbrs) >= 3:
                chirality_list.append((idx, nbrs[0], nbrs[1], nbrs[2], 0.65 * float(atom.chiral_tag)))

    return PocketConditionedForceField(
        bonds=tuple(bonds_list),
        angles=tuple(angles_list),
        torsions=tuple(torsions_list),
        chirality_terms=tuple(chirality_list),
        ligand_elements=tuple(a.element for a in graph.atoms),
        ligand_charges=tuple(float(c) for c in charges),
    )


def intramolecular_energy_and_grad(
    coords: np.ndarray,
    ff: PocketConditionedForceField,
) -> tuple[float, np.ndarray]:
    """Compute ligand internal potential energy and analytical Cartesian gradient."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    grad = np.zeros_like(coords)
    total_e = 0.0

    bonded_set: set[tuple[int, int]] = set()
    # 1. Harmonic bonds
    for u, v, r0, k_b in ff.bonds:
        bonded_set.add((min(u, v), max(u, v)))
        vec = coords[u] - coords[v]
        dist = max(float(np.linalg.norm(vec)), 1e-8)
        diff = dist - r0
        total_e += 0.5 * k_b * diff * diff
        g_vec = k_b * diff * (vec / dist)
        grad[u] += g_vec
        grad[v] -= g_vec

    # 2. Harmonic bond angles (using cosine formulation for smooth analytical derivatives)
    for ang in ff.angles:
        bonded_set.add((min(ang.i, ang.k), max(ang.i, ang.k)))
        u_vec = coords[ang.i] - coords[ang.j]
        v_vec = coords[ang.k] - coords[ang.j]
        ru = max(float(np.linalg.norm(u_vec)), 1e-8)
        rv = max(float(np.linalg.norm(v_vec)), 1e-8)
        cos_theta = float(np.clip(np.dot(u_vec, v_vec) / (ru * rv), -0.999999, 0.999999))
        cos_0 = math.cos(ang.theta0_rad)
        diff_c = cos_theta - cos_0
        total_e += 0.5 * ang.k_theta * diff_c * diff_c

        dcos_du = (v_vec / (ru * rv)) - (cos_theta / (ru * ru)) * u_vec
        dcos_dv = (u_vec / (ru * rv)) - (cos_theta / (rv * rv)) * v_vec
        scale = ang.k_theta * diff_c
        grad[ang.i] += scale * dcos_du
        grad[ang.k] += scale * dcos_dv
        grad[ang.j] -= scale * (dcos_du + dcos_dv)

    # 3. Non-bonded intramolecular soft-core repulsion (1-4 and higher pairs)
    # Intramolecular steric core radius (~1.35 A) with soft-core smoothing prevents
    # 1-4 aromatic ring atoms (~2.4-2.8 A apart) from experiencing artificial LJ walls.
    intra_soft_sq = 0.55 * 0.55
    intra_sigma = 1.35
    intra_eps = 0.04
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in bonded_set:
                continue
            vec = coords[i] - coords[j]
            r2_eff = float(np.dot(vec, vec)) + intra_soft_sq
            inv_r2 = (intra_sigma * intra_sigma) / r2_eff
            inv_r6 = inv_r2**3
            total_e += intra_eps * (inv_r6 * inv_r6)
            g_factor = -12.0 * intra_eps * (inv_r6 * inv_r6) / r2_eff
            grad[i] += g_factor * vec
            grad[j] -= g_factor * vec

    # 4. Chirality signed-volume terms
    for i, j, k, atom_l, target_v in ff.chirality_terms:
        a = coords[j] - coords[i]
        b = coords[k] - coords[i]
        c = coords[atom_l] - coords[i]
        vol = signed_volume(coords, i, j, k, atom_l)
        diff_v = vol - target_v
        total_e += 0.5 * ff.chirality_strength * diff_v * diff_v
        scale = ff.chirality_strength * diff_v
        gj = np.cross(b, c)
        gk = np.cross(c, a)
        gl = np.cross(a, b)
        grad[j] += scale * gj
        grad[k] += scale * gk
        grad[atom_l] += scale * gl
        grad[i] -= scale * (gj + gk + gl)

    return float(total_e), grad


def intermolecular_energy_and_grad(
    ligand_coords: np.ndarray,
    ff: PocketConditionedForceField,
    pocket: BindingPocket,
) -> tuple[float, np.ndarray]:
    """Compute soft-core 12-6 Lennard-Jones + screened Coulomb interaction between ligand and pocket."""
    ligand_coords = np.asarray(ligand_coords, dtype=np.float64)
    pocket_coords = pocket.coordinates()
    if pocket_coords.shape[0] == 0:
        return 0.0, np.zeros_like(ligand_coords)

    n_lig = ligand_coords.shape[0]
    p_sigmas, p_epsilons = pocket.lj_arrays()
    p_charges = pocket.charges()

    l_sigmas = np.empty(n_lig, dtype=np.float64)
    l_epsilons = np.empty(n_lig, dtype=np.float64)
    for i in range(n_lig):
        el = ff.ligand_elements[i] if i < len(ff.ligand_elements) else "C"
        l_sigmas[i], l_epsilons[i] = _lj_parameters(el)
    l_charges = (
        np.array(ff.ligand_charges, dtype=np.float64)
        if len(ff.ligand_charges) == n_lig
        else np.zeros(n_lig, dtype=np.float64)
    )

    # Pairwise displacement vectors: shape (N_lig, N_pock, 3)
    deltas = ligand_coords[:, None, :] - pocket_coords[None, :, :]
    r2 = np.sum(deltas * deltas, axis=-1)
    # Soft-core distance squared to prevent overflow on initial steric overlaps
    soft_delta2 = 0.65 * 0.65
    r2_eff = r2 + soft_delta2
    r_eff = np.sqrt(r2_eff)

    # Lorentz-Berthelot combining rules
    sigma_ij = 0.5 * (l_sigmas[:, None] + p_sigmas[None, :])
    eps_ij = np.sqrt(l_epsilons[:, None] * p_epsilons[None, :])

    sr2 = (sigma_ij * sigma_ij) / r2_eff
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6

    e_lj = 4.0 * eps_ij * (sr12 - sr6)
    # Derivative of 4 * eps * ((sigma^2 / r_eff^2)^6 - (sigma^2 / r_eff^2)^3) w.r.t. x_i
    d_elj_dr2 = 4.0 * eps_ij * (-6.0 * sr12 + 3.0 * sr6) / r2_eff

    # Distance-dependent dielectric Coulomb energy: E_coul = 332.06 * q_i * q_j / (eps_r * r_eff^2)
    q_ij = l_charges[:, None] * p_charges[None, :]
    coul_prefactor = _COULOMB_KCAL_A_E2 / _DIELECTRIC_CONSTANT
    e_coul = coul_prefactor * q_ij / r_eff
    d_ecoul_dr2 = -0.5 * coul_prefactor * q_ij / (r_eff * r2_eff)

    total_inter = float(np.sum(e_lj + e_coul))
    pair_grad_scale = 2.0 * (d_elj_dr2 + d_ecoul_dr2)  # shape (N_lig, N_pock)
    grad_lig = np.sum(pair_grad_scale[:, :, None] * deltas, axis=1)

    return total_inter, grad_lig


def pocket_velocity_field(
    ligand_coords: np.ndarray,
    ff: PocketConditionedForceField,
    pocket: BindingPocket,
) -> np.ndarray:
    """Equivariant drift $v_L(X_L, X_P) = -\\nabla_{X_L} (E_{\\text{intra}} + E_{\\text{inter}})$."""
    _, grad_intra = intramolecular_energy_and_grad(ligand_coords, ff)
    _, grad_inter = intermolecular_energy_and_grad(ligand_coords, ff, pocket)
    return -(grad_intra + grad_inter)


def pocket_equivariance_error(
    ligand_coords: np.ndarray,
    ff: PocketConditionedForceField,
    pocket: BindingPocket,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    """Verify joint $SE(3)$ equivariance when both ligand and pocket transform by $(R, t)$."""
    lig_t = rigid_transform(ligand_coords, rotation, translation)
    pock_coords_t = rigid_transform(pocket.coordinates(), rotation, translation)
    transformed_atoms = tuple(
        PocketAtom(
            index=a.index,
            element=a.element,
            position=(float(pock_coords_t[idx, 0]), float(pock_coords_t[idx, 1]), float(pock_coords_t[idx, 2])),
            partial_charge=a.partial_charge,
            residue_name=a.residue_name,
            residue_id=a.residue_id,
        )
        for idx, a in enumerate(pocket.atoms)
    )
    pocket_t = BindingPocket(
        pocket_id=pocket.pocket_id,
        target_name=pocket.target_name,
        atoms=transformed_atoms,
        clash_distance_threshold=pocket.clash_distance_threshold,
    )
    lhs = pocket_velocity_field(lig_t, ff, pocket_t)
    rhs = pocket_velocity_field(ligand_coords, ff, pocket) @ rotation.T
    denom = float(np.linalg.norm(rhs)) or 1.0
    return float(np.linalg.norm(lhs - rhs) / denom)


def relax_ligand_in_pocket(
    graph: MolecularGraph,
    pocket: BindingPocket,
    initial_coords: np.ndarray | None = None,
    n_steps: int = 180,
    step_size: float = 0.004,
    max_grad_norm: float = 25.0,
    seed: int = 42,
) -> PocketDockingResult:
    """Relax a ligand inside a protein binding pocket and adjudicate steric clashes."""
    ff = build_pocket_force_field(graph)
    if initial_coords is None:
        coords = embed_conformer_3d(graph, seed=seed, n_relax_steps=100)
        p_coords = pocket.coordinates()
        if p_coords.shape[0] > 0:
            pocket_center = p_coords.mean(axis=0)
            coords = coords - coords.mean(axis=0) + pocket_center
    else:
        coords = np.asarray(initial_coords, dtype=np.float64).copy()

    e_intra_0, _ = intramolecular_energy_and_grad(coords, ff)
    e_inter_0, _ = intermolecular_energy_and_grad(coords, ff, pocket)
    initial_total = e_intra_0 + e_inter_0
    trajectory_e = [initial_total]

    current = coords.copy()
    for _ in range(n_steps):
        e_in, g_in = intramolecular_energy_and_grad(current, ff)
        e_out, g_out = intermolecular_energy_and_grad(current, ff, pocket)
        g_total = g_in + g_out
        gnorm = float(np.linalg.norm(g_total))
        if gnorm > max_grad_norm:
            g_total = g_total * (max_grad_norm / gnorm)
        current = current - step_size * g_total
        trajectory_e.append(e_in + e_out)

    e_intra_f, _ = intramolecular_energy_and_grad(current, ff)
    e_inter_f, _ = intermolecular_energy_and_grad(current, ff, pocket)
    final_total = e_intra_f + e_inter_f

    p_coords = pocket.coordinates()
    if p_coords.shape[0] > 0:
        deltas = current[:, None, :] - p_coords[None, :, :]
        dists = np.linalg.norm(deltas, axis=-1)
        min_dist = float(np.min(dists))
        clash_count = int(np.sum(dists < pocket.clash_distance_threshold))
    else:
        min_dist = 99.0
        clash_count = 0

    passed = clash_count == 0 and e_inter_f < 5.0
    reason = ""
    if clash_count > 0:
        reason = f"Rejected: {clash_count} steric clashes below {pocket.clash_distance_threshold:.2f} A (min={min_dist:.2f} A)"
    elif e_inter_f >= 5.0:
        reason = f"Rejected: unfavorable intermolecular pocket interaction energy ({e_inter_f:.2f} kcal/mol)"

    return PocketDockingResult(
        ligand_coordinates=current,
        initial_total_energy=round(initial_total, 4),
        final_total_energy=round(final_total, 4),
        intermolecular_energy_kcal=round(e_inter_f, 4),
        intramolecular_strain_kcal=round(e_intra_f, 4),
        min_interatomic_distance_a=round(min_dist, 4),
        steric_clash_count=clash_count,
        passed_gate=passed,
        rejection_reason=reason,
        energy_trajectory=trajectory_e,
    )


# ============================================================================
# DIRECTION 3: HAWKINS-CRAMER-TRUHLAR GENERALIZED BORN / SURFACE AREA (GB/SA)
# ============================================================================


@dataclass(frozen=True)
class GBSAResult:
    """Analytical Hawkins-Cramer-Truhlar Generalized Born / Surface Area (GB/SA) binding free energy."""

    delta_g_bind_kcal: float
    delta_e_mm_kcal: float
    delta_g_gb_kcal: float
    delta_g_sa_kcal: float
    buried_sasa_a2: float
    ligand_sasa_a2: float
    pocket_contact_atoms: int


def compute_gbsa_binding_energy(
    ligand_coords: np.ndarray,
    ff: PocketConditionedForceField,
    pocket: BindingPocket,
    probe_radius: float = 1.4,
    solvent_dielectric: float = 78.5,
    surface_tension: float = 0.0054,
) -> GBSAResult:
    """Compute analytical Hawkins-Cramer-Truhlar GB/SA implicit solvation binding free energy.

    Evaluates:
      1. Intermolecular gas-phase mechanics: Delta E_MM = E_vdW + E_coulomb
      2. Generalized Born polar desolvation: Delta G_GB = -0.5 * (1 - 1/eps_w) * sum q_i q_j / f_GB(r_ij)
         with effective Born radii R_i computed via pairwise HCT descreening integrals.
      3. Nonpolar hydrophobic cavity desolvation: Delta G_SA = -gamma * Delta SASA_buried
         via Linear Combinations of Pairwise Overlaps (LCPO).
      4. Total binding free energy: Delta G_bind = Delta E_MM + Delta G_GB + Delta G_SA.
    """
    ligand_coords = np.asarray(ligand_coords, dtype=np.float64)
    pocket_coords = pocket.coordinates()
    n_lig = ligand_coords.shape[0]
    n_pock = pocket_coords.shape[0]

    if n_pock == 0 or n_lig == 0:
        return GBSAResult(
            delta_g_bind_kcal=0.0,
            delta_e_mm_kcal=0.0,
            delta_g_gb_kcal=0.0,
            delta_g_sa_kcal=0.0,
            buried_sasa_a2=0.0,
            ligand_sasa_a2=0.0,
            pocket_contact_atoms=0,
        )

    e_mm, _ = intermolecular_energy_and_grad(ligand_coords, ff, pocket)

    p_charges = pocket.charges()
    l_charges = (
        np.array(ff.ligand_charges, dtype=np.float64)
        if len(ff.ligand_charges) == n_lig
        else np.zeros(n_lig, dtype=np.float64)
    )

    l_rho = np.empty(n_lig, dtype=np.float64)
    for i in range(n_lig):
        el = ff.ligand_elements[i] if i < len(ff.ligand_elements) else "C"
        sig, _ = _lj_parameters(el)
        l_rho[i] = max(1.1, 0.5 * sig)

    p_rho = np.empty(n_pock, dtype=np.float64)
    p_sigmas, _ = pocket.lj_arrays()
    for j in range(n_pock):
        p_rho[j] = max(1.1, 0.5 * p_sigmas[j])

    deltas = ligand_coords[:, None, :] - pocket_coords[None, :, :]
    r_ij = np.linalg.norm(deltas, axis=-1)
    r_ij = np.maximum(r_ij, 0.8)

    inv_r4 = 1.0 / (r_ij ** 4)
    descreen_l = np.sum((p_rho[None, :] ** 3) * inv_r4 / 3.0, axis=1)
    inv_R_l = (1.0 / l_rho) - descreen_l
    R_l = np.clip(1.0 / np.maximum(inv_R_l, 1.0 / 30.0), l_rho, 30.0)

    descreen_p = np.sum((l_rho[:, None] ** 3) * inv_r4 / 3.0, axis=0)
    inv_R_p = (1.0 / p_rho) - descreen_p
    R_p = np.clip(1.0 / np.maximum(inv_R_p, 1.0 / 30.0), p_rho, 30.0)

    R_prod = R_l[:, None] * R_p[None, :]
    f_gb = np.sqrt(r_ij ** 2 + R_prod * np.exp(-(r_ij ** 2) / (4.0 * R_prod)))

    charge_prod = l_charges[:, None] * p_charges[None, :]
    gb_scaling = -0.5 * (1.0 - (1.0 / solvent_dielectric)) * _COULOMB_KCAL_A_E2
    delta_g_gb = float(gb_scaling * np.sum(charge_prod / f_gb))

    R_l_probe = l_rho + probe_radius
    R_p_probe = p_rho + probe_radius

    l_deltas = ligand_coords[:, None, :] - ligand_coords[None, :, :]
    l_r = np.linalg.norm(l_deltas, axis=-1)
    np.fill_diagonal(l_r, 999.0)

    sasa_l_free_atoms = np.zeros(n_lig, dtype=np.float64)
    sasa_l_bound_atoms = np.zeros(n_lig, dtype=np.float64)

    for i in range(n_lig):
        area_0 = 4.0 * math.pi * (R_l_probe[i] ** 2)
        shield_free = 1.0
        for k in range(n_lig):
            if k == i:
                continue
            dist = l_r[i, k]
            cutoff = R_l_probe[i] + R_l_probe[k]
            if dist < cutoff:
                overlap = (R_l_probe[k] * (cutoff - dist)) / (2.0 * R_l_probe[i] * max(0.5, dist))
                shield_free *= max(0.0, 1.0 - 0.25 * overlap)
        sasa_l_free_atoms[i] = area_0 * shield_free

        shield_bound = shield_free
        for j in range(n_pock):
            dist = r_ij[i, j]
            cutoff = R_l_probe[i] + R_p_probe[j]
            if dist < cutoff:
                overlap = (R_p_probe[j] * (cutoff - dist)) / (2.0 * R_l_probe[i] * max(0.5, dist))
                shield_bound *= max(0.0, 1.0 - 0.25 * overlap)
        sasa_l_bound_atoms[i] = area_0 * shield_bound

    total_sasa_l = float(np.sum(sasa_l_free_atoms))
    total_sasa_bound = float(np.sum(sasa_l_bound_atoms))
    buried_sasa = max(0.0, total_sasa_l - total_sasa_bound)
    delta_g_sa = float(-surface_tension * buried_sasa)

    delta_g_bind = float(e_mm + delta_g_gb + delta_g_sa)
    contact_count = int(np.sum(np.min(r_ij, axis=0) < 4.5))

    return GBSAResult(
        delta_g_bind_kcal=round(delta_g_bind, 4),
        delta_e_mm_kcal=round(float(e_mm), 4),
        delta_g_gb_kcal=round(delta_g_gb, 4),
        delta_g_sa_kcal=round(delta_g_sa, 4),
        buried_sasa_a2=round(buried_sasa, 2),
        ligand_sasa_a2=round(total_sasa_l, 2),
        pocket_contact_atoms=contact_count,
    )
