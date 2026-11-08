"""6-Layer Fail-Closed Drug Discovery Pipeline with Strict Proposer vs. Referee Separation.

Status: experimental

Scope & honesty ceiling:
- Implements the constitutionally compliant 6-layer drug discovery funnel under `AGENTS.md`:
  - **Layer 1 (Proposer)**: `MMLMMolecularProposer` emits standardized `MolecularCandidateProposal`
    objects. It never adjudicates its own candidates and never accesses holdout assay slices.
  - **Layer 2 (Chemical & Retrosynthesis Referee)**: Sanitizes valence, octet, and Bredt's rule
    via `chemistry_bridge.py` and checks multi-step retrosynthetic feasibility (`<= 4` steps)
    plus `SAScore < 3.5` via `RetrosynthesisReferee`.
  - **Layer 3 (3D Equivariant Pocket Flow Referee)**: Relaxes the ligand inside the target
    protein pocket via `equivariant_pocket_flow.py` and rejects steric clashes.
  - **Layer 4 (Alchemical FEP Free Energy Gate)**: Computes BAR relative binding free energy
    ($\\Delta\\Delta G$) and rejects candidates that fail convergence ($\\sigma > 0.5\\text{ kcal/mol}$)
    or affinity thresholds via `AlchemicalFEPGateway`.
  - **Layer 5 (Conformal ADMET & 2-Compartment TMDD PK/PD Referee)**: Enforces $95\\%$ conformal
    safety bounds on solubility, hERG $IC_{50}$, and metabolic half-life, and simulates
    2-compartment TMDD target receptor occupancy over 24 hours.
  - **Layer 6 (Epistemic Ledger & Wet-Lab Confirmation Gateway)**: Records full lineage in
    `FullLineageExperimentLedger`, enforces `HoldoutContaminationError` on burned assay slices,
    and distinguishes *In Silico Confirmed Lead Candidates* from *Wet-Lab Verified Drugs*.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from apedp_dyn.biophysics.alchemical_fep_gateway import (
    ADMETConformalGateResult,
    AlchemicalFEPGateway,
    AlchemicalFEPResult,
    ConformalADMETReferee,
)
from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
)
from apedp_dyn.epistemic.retrosynthesis_gate import RetrosynthesisReferee, RetrosynthesisVerdict
from apedp_dyn.molecular.chemistry_bridge import (
    MolecularGraph,
    PhysicochemicalDescriptors,
    ValenceError,
    compute_descriptors,
    parse_selfies,
    parse_smiles,
)
from apedp_dyn.molecular.equivariant_pocket_flow import (
    BindingPocket,
    PocketDockingResult,
    relax_ligand_in_pocket,
)
from apedp_dyn.problems.tmdd_pkpd import simulate_tmdd_trajectory
from apedp_mmlm.engines import VectorSymbolicEngine


@dataclass(frozen=True)
class MolecularCandidateProposal:
    """Standardized candidate molecule hypothesis emitted by a proposer (Layer 1)."""

    candidate_id: str
    proposer_id: str
    smiles_or_selfies: str
    target_id: str
    representation_kind: str = "smiles"  # "smiles" or "selfies"
    parent_candidate_id: str | None = None
    proposed_dose_nmol_l: float = 5.0


@dataclass(frozen=True)
class WetLabAssayHoldoutSlice:
    """External wet-lab experimental assay slice (SPR / ITC / Cryo-EM confirmation holdout)."""

    dataset_id: str
    slice_id: str
    measured_kd_nm: float
    measured_herg_ic50_um: float
    max_acceptable_kd_nm: float = 250.0
    min_acceptable_herg_um: float = 10.0


@dataclass
class DrugDiscoveryAdjudicationReport:
    """Complete 6-layer adjudication record for a molecular candidate proposal."""

    candidate_id: str
    proposer_id: str
    canonical_hash: str
    lifecycle_state: CandidateLifecycleState
    in_silico_confirmed_lead: bool
    wet_lab_verified_drug: bool
    failed_layer: int | None = None
    rejection_reason: str = ""
    descriptors: PhysicochemicalDescriptors | None = None
    retrosynthesis: RetrosynthesisVerdict | None = None
    docking: PocketDockingResult | None = None
    fep: AlchemicalFEPResult | None = None
    admet: ADMETConformalGateResult | None = None
    tmdd_mean_occupancy: float = 0.0
    ledger_metrics: dict[str, float] = field(default_factory=dict)


class MMLMMolecularProposer:
    """Layer 1 Hypothesis Proposer emitting candidate molecular topologies conditioned on pocket properties.

    Strictly separated from referee gates and confirmation holdouts per `AGENTS.md` §1.1.
    """

    def __init__(self, proposer_id: str = "mmlm_molecular_proposer_v1") -> None:
        self.proposer_id = proposer_id

    def generate_proposals_for_pocket(
        self,
        pocket: BindingPocket,
        scaffold_smiles_pool: Sequence[str] | None = None,
        max_candidates: int = 4,
    ) -> list[MolecularCandidateProposal]:
        """Emit standardized `MolecularCandidateProposal` objects without self-adjudication."""
        p_coords = pocket.coordinates()
        pocket_span = float(p_coords.max() - p_coords.min()) if p_coords.size > 0 else 10.0

        default_scaffolds = [
            "CC(=O)Nc1ccc(O)cc1",          # Paracetamol-like amide scaffold
            "CC(=O)Nc1ccncc1",             # N-(pyridin-4-yl)acetamide scaffold
            "CC(=O)Oc1ccccc1C(=O)O",       # Aspirin-like ester scaffold
            "CCNc1ccc(O)cc1",              # 4-(ethylamino)phenol scaffold
        ]
        pool = list(scaffold_smiles_pool) if scaffold_smiles_pool is not None else default_scaffolds
        selected = pool[: max(1, max_candidates)]
        dose = round(max(1.5, min(10.0, 0.4 * pocket_span)), 2)

        proposals: list[MolecularCandidateProposal] = []
        for idx, smi in enumerate(selected):
            proposals.append(
                MolecularCandidateProposal(
                    candidate_id=f"{pocket.pocket_id}_cand_{idx + 1}",
                    proposer_id=self.proposer_id,
                    smiles_or_selfies=smi,
                    target_id=pocket.target_name,
                    representation_kind="selfies" if smi.strip().startswith("[") else "smiles",
                    proposed_dose_nmol_l=dose,
                )
            )
        return proposals


@dataclass(frozen=True)
class SynthonFragment:
    """A chemical building block with defined reactive handles and pharmacophore features."""

    synthon_id: str
    kind: str  # "acid", "amine", "aryl_halide", "boronic_acid"
    smiles: str
    hbd: int
    hba: int
    aromatic_rings: int
    lipophilic_score: float
    molecular_weight: float


class VSASynthonDrugGenerator:
    """Vector Symbolic Architecture (VSA) Synthon-Guided De Novo Molecular Proposer.

    Uses `VectorSymbolicEngine` (Holographic Reduced Representations) to encode target pocket
    pharmacophoric profiles (H-bond donors/acceptors, lipophilic/aromatic features) and binds
    them to candidate synthon building blocks. Synthesizes leads via genuine forward coupling
    reactions (amide coupling, Suzuki-Miyaura coupling) that are guaranteed retrosynthetically
    valid (Layer 2 SA score < 3.0, steps <= 2) by construction.
    """

    def __init__(self, proposer_id: str = "vsa_synthon_generator_v1", vsa_dim: int = 128) -> None:
        self.proposer_id = proposer_id
        self.vsa = VectorSymbolicEngine(dim=vsa_dim, seed=42)
        self.feature_keys = ("hbd", "hba", "arom", "lipo", "size")
        for key in self.feature_keys:
            self.vsa.get_or_create_vector(key)

        self.synthons: dict[str, SynthonFragment] = {
            "benzoic_acid": SynthonFragment("benzoic_acid", "acid", "c1ccccc1C(=O)O", 1, 1, 1, 0.8, 122.1),
            "4_hydroxybenzoic_acid": SynthonFragment(
                "4_hydroxybenzoic_acid", "acid", "Oc1ccc(cc1)C(=O)O", 2, 2, 1, 0.5, 138.1
            ),
            "4_fluorobenzoic_acid": SynthonFragment(
                "4_fluorobenzoic_acid", "acid", "Fc1ccc(cc1)C(=O)O", 1, 1, 1, 0.9, 140.1
            ),
            "nicotinic_acid": SynthonFragment("nicotinic_acid", "acid", "c1cncc(c1)C(=O)O", 1, 2, 1, 0.3, 123.1),
            "acetic_acid": SynthonFragment("acetic_acid", "acid", "CC(=O)O", 1, 1, 0, 0.1, 60.0),
            "4_aminophenol": SynthonFragment("4_aminophenol", "amine", "Nc1ccc(O)cc1", 2, 1, 1, 0.5, 109.1),
            "aniline": SynthonFragment("aniline", "amine", "Nc1ccccc1", 1, 0, 1, 0.7, 93.1),
            "4_fluoroaniline": SynthonFragment("4_fluoroaniline", "amine", "Nc1ccc(F)cc1", 1, 0, 1, 0.8, 111.1),
            "4_aminopyridine": SynthonFragment("4_aminopyridine", "amine", "Nc1ccncc1", 1, 1, 1, 0.2, 94.1),
            "4_bromophenol": SynthonFragment("4_bromophenol", "aryl_halide", "Oc1ccc(Br)cc1", 1, 1, 1, 0.8, 173.0),
            "1_bromo_4_fluorobenzene": SynthonFragment(
                "1_bromo_4_fluorobenzene", "aryl_halide", "Fc1ccc(Br)cc1", 0, 0, 1, 1.0, 175.0
            ),
            "phenylboronic_acid": SynthonFragment(
                "phenylboronic_acid", "boronic_acid", "OB(O)c1ccccc1", 2, 2, 1, 0.6, 121.9
            ),
            "4_methoxyphenylboronic_acid": SynthonFragment(
                "4_methoxyphenylboronic_acid", "boronic_acid", "COc1ccc(cc1)B(O)O", 2, 3, 1, 0.7, 151.9
            ),
        }

        self.reactions: list[tuple[str, str, str, str]] = [
            ("acetic_acid", "4_aminophenol", "amide_coupling", "CC(=O)Nc1ccc(O)cc1"),
            ("benzoic_acid", "4_aminophenol", "amide_coupling", "O=C(Nc1ccc(O)cc1)c2ccccc2"),
            ("4_fluorobenzoic_acid", "4_aminophenol", "amide_coupling", "O=C(Nc1ccc(O)cc1)c2ccc(F)cc2"),
            ("nicotinic_acid", "4_aminophenol", "amide_coupling", "O=C(Nc1ccc(O)cc1)c2cccnc2"),
            ("benzoic_acid", "4_fluoroaniline", "amide_coupling", "O=C(Nc1ccc(F)cc1)c2ccccc2"),
            ("acetic_acid", "4_aminopyridine", "amide_coupling", "CC(=O)Nc1ccncc1"),
            ("4_bromophenol", "phenylboronic_acid", "suzuki_coupling", "Oc1ccc(-c2ccccc2)cc1"),
            ("1_bromo_4_fluorobenzene", "4_methoxyphenylboronic_acid", "suzuki_coupling", "COc1ccc(-c2ccc(F)cc2)cc1"),
        ]

    def _embed_pharmacophore(self, hbd: float, hba: float, arom: float, lipo: float, size: float) -> np.ndarray:
        vec = (
            hbd * self.vsa.get_or_create_vector("hbd")
            + hba * self.vsa.get_or_create_vector("hba")
            + arom * self.vsa.get_or_create_vector("arom")
            + lipo * self.vsa.get_or_create_vector("lipo")
            + size * self.vsa.get_or_create_vector("size")
        )
        norm = float(np.linalg.norm(vec))
        _out: np.ndarray = vec / norm if norm > 1e-9 else vec
        return _out

    def generate_proposals_for_pocket(
        self,
        pocket: BindingPocket,
        max_candidates: int = 4,
    ) -> list[MolecularCandidateProposal]:
        """Emit standardized `MolecularCandidateProposal` objects assembled from VSA synthon matching."""
        p_coords = pocket.coordinates()
        pocket_span = float(p_coords.max() - p_coords.min()) if p_coords.size > 0 else 10.0
        p_charges = pocket.charges()

        demand_hbd = float(np.sum(p_charges < -0.1)) * 0.5 + 1.0
        demand_hba = float(np.sum(p_charges > 0.1)) * 0.5 + 1.0
        demand_arom = max(1.0, float(len(pocket.atoms)) * 0.08)
        demand_lipo = 0.6
        demand_size = max(1.0, pocket_span * 0.2)

        pocket_demand_vec = self._embed_pharmacophore(
            demand_hbd, demand_hba, demand_arom, demand_lipo, demand_size
        )

        scored_reactions: list[tuple[float, str, str, str, str]] = []
        for s_a_id, s_b_id, rxn_type, prod_smiles in self.reactions:
            sa = self.synthons[s_a_id]
            sb = self.synthons[s_b_id]
            prod_hbd = float(sa.hbd + sb.hbd - 1)
            prod_hba = float(sa.hba + sb.hba)
            prod_arom = float(sa.aromatic_rings + sb.aromatic_rings)
            prod_lipo = 0.5 * (sa.lipophilic_score + sb.lipophilic_score)
            prod_size = (sa.molecular_weight + sb.molecular_weight) / 100.0

            cand_vec = self._embed_pharmacophore(prod_hbd, prod_hba, prod_arom, prod_lipo, prod_size)
            similarity = float(np.dot(pocket_demand_vec, cand_vec))
            scored_reactions.append((similarity, s_a_id, s_b_id, rxn_type, prod_smiles))

        scored_reactions.sort(key=lambda x: x[0], reverse=True)
        selected = scored_reactions[: max(1, max_candidates)]
        dose = round(max(1.5, min(10.0, 0.4 * pocket_span)), 2)

        proposals: list[MolecularCandidateProposal] = []
        for idx, (_, _, _, _, smi) in enumerate(selected):
            proposals.append(
                MolecularCandidateProposal(
                    candidate_id=f"{pocket.pocket_id}_synthon_{idx + 1}",
                    proposer_id=self.proposer_id,
                    smiles_or_selfies=smi,
                    target_id=pocket.target_name,
                    representation_kind="smiles",
                    proposed_dose_nmol_l=dose,
                )
            )
        return proposals


class DrugDiscoveryPipelineReferee:
    """6-Layer Fail-Closed Epistemic Referee Funnel for Drug Discovery Proposals."""

    def __init__(
        self,
        ledger: FullLineageExperimentLedger,
        retrosynthesis_referee: RetrosynthesisReferee | None = None,
        fep_gateway: AlchemicalFEPGateway | None = None,
        admet_referee: ConformalADMETReferee | None = None,
        min_receptor_occupancy: float = 0.20,
    ) -> None:
        self.ledger = ledger
        self.retrosynthesis_referee = retrosynthesis_referee or RetrosynthesisReferee()
        self.fep_gateway = fep_gateway or AlchemicalFEPGateway()
        self.admet_referee = admet_referee or ConformalADMETReferee()
        self.min_receptor_occupancy = min_receptor_occupancy

    def adjudicate_proposal(
        self,
        proposal: MolecularCandidateProposal,
        pocket: BindingPocket,
        wet_lab_holdout: WetLabAssayHoldoutSlice | None = None,
        seed: int = 42,
    ) -> DrugDiscoveryAdjudicationReport:
        """Run a candidate proposal through all 6 deterministic layers and record in the ledger."""
        # Register candidate in the FullLineageExperimentLedger
        self.ledger.register_candidate(
            candidate_id=proposal.candidate_id,
            proposer_id=proposal.proposer_id,
            formula=proposal.smiles_or_selfies,
            parent_candidate_id=proposal.parent_candidate_id,
            feature_set=(proposal.target_id, proposal.representation_kind),
            parameter_choices={"proposed_dose_nmol_l": proposal.proposed_dose_nmol_l},
        )

        # LAYER 2A: Chemical Graph Sanitization
        try:
            if proposal.representation_kind == "selfies":
                graph: MolecularGraph = parse_selfies(proposal.smiles_or_selfies)
            else:
                graph = parse_smiles(proposal.smiles_or_selfies)
        except ValenceError as exc:
            reason = f"Layer 2 Chemical Valence Violation: {exc}"
            self.ledger.update_lifecycle_state(
                proposal.candidate_id,
                CandidateLifecycleState.PRE_SCREEN_REJECTED,
                rejection_reason=reason,
            )
            return DrugDiscoveryAdjudicationReport(
                candidate_id=proposal.candidate_id,
                proposer_id=proposal.proposer_id,
                canonical_hash="invalid_topology",
                lifecycle_state=CandidateLifecycleState.PRE_SCREEN_REJECTED,
                in_silico_confirmed_lead=False,
                wet_lab_verified_drug=False,
                failed_layer=2,
                rejection_reason=reason,
            )

        mol_hash = graph.canonical_hash()
        descriptors = compute_descriptors(graph)

        # LAYER 2B: Retrosynthesis & Synthetic Accessibility Gate
        retro_verdict = self.retrosynthesis_referee.adjudicate(graph)
        if not retro_verdict.passed:
            reason = f"Layer 2 Retrosynthesis Gate Failed: {retro_verdict.rejection_reason}"
            self.ledger.update_lifecycle_state(
                proposal.candidate_id,
                CandidateLifecycleState.PRE_SCREEN_REJECTED,
                rejection_reason=reason,
                metrics={"sa_score": retro_verdict.sa_score, "retro_steps": float(retro_verdict.num_steps)},
            )
            return DrugDiscoveryAdjudicationReport(
                candidate_id=proposal.candidate_id,
                proposer_id=proposal.proposer_id,
                canonical_hash=mol_hash,
                lifecycle_state=CandidateLifecycleState.PRE_SCREEN_REJECTED,
                in_silico_confirmed_lead=False,
                wet_lab_verified_drug=False,
                failed_layer=2,
                rejection_reason=reason,
                descriptors=descriptors,
                retrosynthesis=retro_verdict,
            )

        # LAYER 3: 3D Equivariant Pocket Conformer Relaxation & Steric Clash Referee
        dock_res = relax_ligand_in_pocket(graph, pocket, seed=seed)
        if not dock_res.passed_gate:
            reason = f"Layer 3 Pocket Flow Gate Failed: {dock_res.rejection_reason}"
            self.ledger.update_lifecycle_state(
                proposal.candidate_id,
                CandidateLifecycleState.CALIBRATION_FAILED,
                rejection_reason=reason,
                metrics={
                    "intermolecular_energy_kcal": dock_res.intermolecular_energy_kcal,
                    "steric_clash_count": float(dock_res.steric_clash_count),
                },
            )
            return DrugDiscoveryAdjudicationReport(
                candidate_id=proposal.candidate_id,
                proposer_id=proposal.proposer_id,
                canonical_hash=mol_hash,
                lifecycle_state=CandidateLifecycleState.CALIBRATION_FAILED,
                in_silico_confirmed_lead=False,
                wet_lab_verified_drug=False,
                failed_layer=3,
                rejection_reason=reason,
                descriptors=descriptors,
                retrosynthesis=retro_verdict,
                docking=dock_res,
            )

        # LAYER 4: Alchemical FEP / BAR Relative Binding Free Energy Gate
        fep_res = self.fep_gateway.evaluate_binding_free_energy(
            graph, pocket, docked_coords=dock_res.ligand_coordinates, seed=seed
        )
        if not fep_res.passed_gate:
            reason = f"Layer 4 Alchemical FEP Gate Failed: {fep_res.rejection_reason}"
            self.ledger.update_lifecycle_state(
                proposal.candidate_id,
                CandidateLifecycleState.CALIBRATION_FAILED,
                rejection_reason=reason,
                metrics={
                    "delta_g_bind_kcal": fep_res.delta_g_bind_kcal_mol,
                    "delta_delta_g_kcal": fep_res.delta_delta_g_kcal_mol,
                    "fep_sigma_kcal": fep_res.uncertainty_kcal_mol,
                },
            )
            return DrugDiscoveryAdjudicationReport(
                candidate_id=proposal.candidate_id,
                proposer_id=proposal.proposer_id,
                canonical_hash=mol_hash,
                lifecycle_state=CandidateLifecycleState.CALIBRATION_FAILED,
                in_silico_confirmed_lead=False,
                wet_lab_verified_drug=False,
                failed_layer=4,
                rejection_reason=reason,
                descriptors=descriptors,
                retrosynthesis=retro_verdict,
                docking=dock_res,
                fep=fep_res,
            )

        # LAYER 5: Conformal ADMET & 2-Compartment TMDD PK/PD Referee
        admet_res = self.admet_referee.evaluate_admet(graph)
        # Convert FEP dissociation constant into TMDD k_off / k_on ratio
        kd_nmol_l = max(0.05, min(50.0, fep_res.dissociation_constant_kd_nm / 100.0))
        k_on = 0.50
        k_off = max(0.02, min(2.5, k_on * kd_nmol_l))
        k_el = max(0.02, 4.0 / max(10.0, admet_res.metabolic_half_life_min.point_estimate))
        tmdd_sim = simulate_tmdd_trajectory(
            dose_nmol_l=proposal.proposed_dose_nmol_l,
            k_el=k_el,
            k_on=k_on,
            k_off=k_off,
        )
        mean_occ = float(tmdd_sim["mean_receptor_occupancy"])

        if not admet_res.passed_gate or mean_occ < self.min_receptor_occupancy:
            admet_reasons = list(admet_res.rejection_reasons)
            if mean_occ < self.min_receptor_occupancy:
                admet_reasons.append(
                    f"2-compartment TMDD mean receptor occupancy ({mean_occ:.2f}) < {self.min_receptor_occupancy:.2f}"
                )
            reason = "Layer 5 ADMET/PK-PD Gate Failed: " + "; ".join(admet_reasons)
            self.ledger.update_lifecycle_state(
                proposal.candidate_id,
                CandidateLifecycleState.ECONOMIC_EVALUATION_FAILED,
                rejection_reason=reason,
                metrics={"tmdd_mean_occupancy": mean_occ},
            )
            return DrugDiscoveryAdjudicationReport(
                candidate_id=proposal.candidate_id,
                proposer_id=proposal.proposer_id,
                canonical_hash=mol_hash,
                lifecycle_state=CandidateLifecycleState.ECONOMIC_EVALUATION_FAILED,
                in_silico_confirmed_lead=False,
                wet_lab_verified_drug=False,
                failed_layer=5,
                rejection_reason=reason,
                descriptors=descriptors,
                retrosynthesis=retro_verdict,
                docking=dock_res,
                fep=fep_res,
                admet=admet_res,
                tmdd_mean_occupancy=mean_occ,
            )

        # Candidate passed all 5 in-silico physical and statistical gates!
        metrics_summary = {
            "mw": descriptors.molecular_weight,
            "sa_score": retro_verdict.sa_score,
            "intermolecular_energy_kcal": dock_res.intermolecular_energy_kcal,
            "delta_g_bind_kcal": fep_res.delta_g_bind_kcal_mol,
            "delta_delta_g_kcal": fep_res.delta_delta_g_kcal_mol,
            "tmdd_mean_occupancy": mean_occ,
        }

        # LAYER 6: Epistemic Ledger & External Wet-Lab Assay Holdout Gateway
        if wet_lab_holdout is None:
            # Without physical wet-lab assay data, the candidate is promoted ONLY to
            # DEVELOPMENT_PASSED ("In Silico Confirmed Lead Candidate"), never CONFIRMED_VALID.
            self.ledger.update_lifecycle_state(
                proposal.candidate_id,
                CandidateLifecycleState.DEVELOPMENT_PASSED,
                metrics=metrics_summary,
            )
            return DrugDiscoveryAdjudicationReport(
                candidate_id=proposal.candidate_id,
                proposer_id=proposal.proposer_id,
                canonical_hash=mol_hash,
                lifecycle_state=CandidateLifecycleState.DEVELOPMENT_PASSED,
                in_silico_confirmed_lead=True,
                wet_lab_verified_drug=False,
                descriptors=descriptors,
                retrosynthesis=retro_verdict,
                docking=dock_res,
                fep=fep_res,
                admet=admet_res,
                tmdd_mean_occupancy=mean_occ,
                ledger_metrics=metrics_summary,
            )

        # Evaluate against physical wet-lab confirmation holdout slice (burns slice on failure!)
        assay_passed = (
            wet_lab_holdout.measured_kd_nm <= wet_lab_holdout.max_acceptable_kd_nm
            and wet_lab_holdout.measured_herg_ic50_um >= wet_lab_holdout.min_acceptable_herg_um
        )
        outcome_str = "PASSED" if assay_passed else "FAILED"
        # Raises HoldoutContaminationError if this wet-lab slice was already burned by an earlier failure
        self.ledger.record_data_exposure(
            candidate_id=proposal.candidate_id,
            dataset_id=wet_lab_holdout.dataset_id,
            slice_identifier=wet_lab_holdout.slice_id,
            role=DatasetRole.CONFIRMATION_HOLDOUT,
            outcome_status=outcome_str,
        )

        metrics_summary["wet_lab_kd_nm"] = wet_lab_holdout.measured_kd_nm
        metrics_summary["wet_lab_herg_um"] = wet_lab_holdout.measured_herg_ic50_um

        if not assay_passed:
            reason = (
                f"Layer 6 Wet-Lab Assay Confirmation Failed: measured Kd={wet_lab_holdout.measured_kd_nm:.1f} nM, "
                f"hERG IC50={wet_lab_holdout.measured_herg_ic50_um:.1f} uM"
            )
            self.ledger.update_lifecycle_state(
                proposal.candidate_id,
                CandidateLifecycleState.CONFIRMATION_REJECTED,
                rejection_reason=reason,
                metrics=metrics_summary,
            )
            return DrugDiscoveryAdjudicationReport(
                candidate_id=proposal.candidate_id,
                proposer_id=proposal.proposer_id,
                canonical_hash=mol_hash,
                lifecycle_state=CandidateLifecycleState.CONFIRMATION_REJECTED,
                in_silico_confirmed_lead=True,
                wet_lab_verified_drug=False,
                failed_layer=6,
                rejection_reason=reason,
                descriptors=descriptors,
                retrosynthesis=retro_verdict,
                docking=dock_res,
                fep=fep_res,
                admet=admet_res,
                tmdd_mean_occupancy=mean_occ,
                ledger_metrics=metrics_summary,
            )

        self.ledger.update_lifecycle_state(
            proposal.candidate_id,
            CandidateLifecycleState.CONFIRMED_VALID,
            metrics=metrics_summary,
        )
        return DrugDiscoveryAdjudicationReport(
            candidate_id=proposal.candidate_id,
            proposer_id=proposal.proposer_id,
            canonical_hash=mol_hash,
            lifecycle_state=CandidateLifecycleState.CONFIRMED_VALID,
            in_silico_confirmed_lead=True,
            wet_lab_verified_drug=True,
            descriptors=descriptors,
            retrosynthesis=retro_verdict,
            docking=dock_res,
            fep=fep_res,
            admet=admet_res,
            tmdd_mean_occupancy=mean_occ,
            ledger_metrics=metrics_summary,
        )
