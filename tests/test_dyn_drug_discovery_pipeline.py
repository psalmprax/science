"""Comprehensive Regression Suite for the 6-Layer Drug Discovery Pipeline & Biophysics Referees."""
from __future__ import annotations

import math

import numpy as np
import pytest

from apedp_dyn.biophysics.alchemical_fep_gateway import (
    AlchemicalFEPGateway,
    ConformalADMETReferee,
    bennett_acceptance_ratio,
    zwanzig_free_energy,
)
from apedp_dyn.epistemic.drug_discovery_pipeline import (
    DrugDiscoveryPipelineReferee,
    MMLMMolecularProposer,
    MolecularCandidateProposal,
    VSASynthonDrugGenerator,
    WetLabAssayHoldoutSlice,
)
from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.epistemic.retrosynthesis_gate import RetrosynthesisReferee
from apedp_dyn.molecular.chemistry_bridge import (
    AtomNode,
    BondEdge,
    MolecularGraph,
    ValenceError,
    compute_descriptors,
    embed_conformer_3d,
    parse_selfies,
    parse_smiles,
    sanitize_molecular_graph,
)
from apedp_dyn.molecular.equivariant_pocket_flow import (
    BindingPocket,
    PocketAtom,
    build_pocket_force_field,
    compute_gbsa_binding_energy,
    pocket_equivariance_error,
    relax_ligand_in_pocket,
)
from apedp_dyn.molecular.se3_flow import random_rotation
from apedp_dyn.problems.tmdd_pkpd import (
    load as load_tmdd_problem,
)
from apedp_dyn.problems.tmdd_pkpd import (
    simulate_tmdd_trajectory,
)


def _sample_binding_pocket() -> BindingPocket:
    """Create a well-spaced 3D target binding pocket shell."""
    atoms = (
        PocketAtom(0, "O", (-6.5, 0.0, 0.0), partial_charge=-0.45, residue_name="ASP", residue_id=85),
        PocketAtom(1, "N", (6.5, 0.0, 0.0), partial_charge=0.35, residue_name="LYS", residue_id=112),
        PocketAtom(2, "C", (0.0, -6.5, 0.0), partial_charge=0.05, residue_name="LEU", residue_id=140),
        PocketAtom(3, "C", (0.0, 6.5, 0.0), partial_charge=0.05, residue_name="PHE", residue_id=168),
        PocketAtom(4, "O", (0.0, 0.0, -6.5), partial_charge=-0.35, residue_name="SER", residue_id=195),
        PocketAtom(5, "N", (0.0, 0.0, 6.5), partial_charge=0.25, residue_name="HIS", residue_id=210),
    )
    return BindingPocket(pocket_id="POCK-KRAS-01", target_name="KRAS_G12C", atoms=atoms)


def test_smiles_and_selfies_parsing_and_valence_sanitization():
    """Verify real SMILES/SELFIES graph construction, descriptors, and valence/Bredt rejections."""
    paracetamol = parse_smiles("CC(=O)Nc1ccc(O)cc1")
    desc = compute_descriptors(paracetamol)
    assert paracetamol.num_atoms() == 11
    assert 150.0 <= desc.molecular_weight <= 152.5
    assert desc.aromatic_ring_count == 1
    assert desc.hbd >= 2
    assert desc.hba >= 2
    assert 1.0 <= desc.sa_score < 3.5

    # SELFIES round-trip parsing
    selfies_mol = parse_selfies("[C][C][=O]")
    assert selfies_mol.num_atoms() == 3

    # Pentavalent carbon must raise ValenceError
    with pytest.raises(ValenceError, match="exceeds maximum valence"):
        parse_smiles("C(C)(C)(C)(C)C")

    # Bredt's rule violation: bridgehead double bond in small bicyclic cage (norbornene bridgehead C=C)
    # Bicyclo[2.2.1]hept-1-ene: bridgehead atoms 0 and 3 connected by three bridges:
    # (0=1-2-3), (0-4-5-3), and (0-6-3). Bridgehead 0 has valence 2 + 1 + 1 = 4.
    atoms = [AtomNode(i, "C") for i in range(7)]
    bonds = [
        BondEdge(0, 1, 2.0),  # Bridgehead 0 has a strained double bond!
        BondEdge(1, 2, 1.0),
        BondEdge(2, 3, 1.0),
        BondEdge(0, 4, 1.0),
        BondEdge(4, 5, 1.0),
        BondEdge(5, 3, 1.0),
        BondEdge(0, 6, 1.0),
        BondEdge(6, 3, 1.0),
    ]
    with pytest.raises(ValenceError, match="Bredt's rule violation"):
        sanitize_molecular_graph(MolecularGraph(atoms=atoms, bonds=bonds))


def test_3d_conformer_embedding_and_pocket_se3_equivariance():
    """Verify 3D distance geometry embedding and exact joint SE(3) equivariance in binding pockets."""
    mol = parse_smiles("CC(=O)Nc1ccc(O)cc1")
    coords = embed_conformer_3d(mol, seed=7)
    assert coords.shape == (mol.num_atoms(), 3)
    assert np.all(np.isfinite(coords))

    pocket = _sample_binding_pocket()
    ff = build_pocket_force_field(mol)
    rng = np.random.default_rng(123)
    rot = random_rotation(rng)
    trans = rng.normal(0.0, 3.0, size=3)

    err = pocket_equivariance_error(coords, ff, pocket, rot, trans)
    assert err < 1e-10, f"Joint SE(3) pocket equivariance violated: error={err}"

    dock = relax_ligand_in_pocket(mol, pocket, seed=7)
    assert dock.final_total_energy <= dock.initial_total_energy
    assert dock.steric_clash_count == 0
    assert dock.passed_gate is True


def test_retrosynthesis_referee_disconnection_and_rejection():
    """Verify multi-step amide/biaryl retrosynthetic disconnection and complex macrocycle rejection."""
    referee = RetrosynthesisReferee(max_steps=4, max_sa_score=3.5)
    amide = parse_smiles("CC(=O)Nc1ccc(O)cc1")
    verdict = referee.adjudicate(amide)
    assert verdict.passed is True
    assert 1 <= verdict.num_steps <= 4
    assert any(step.reaction_id == "RXN-AMIDE" for step in verdict.route)

    # Overly complex poly-spiro structure exceeding SAScore gate
    complex_mol = parse_smiles("C1CC2(CCC1)CC3(CCC2)CC4(CCC3)CC5(CCC4)CC6(CCC5)CCC6")
    bad_verdict = referee.adjudicate(complex_mol)
    assert bad_verdict.passed is False
    assert "SAScore" in bad_verdict.rejection_reason


def test_alchemical_fep_bar_and_conformal_admet_gates():
    """Verify Bennett Acceptance Ratio (BAR), Zwanzig free energy, and 95% conformal ADMET bounds."""
    rng = np.random.default_rng(99)
    du_f = rng.normal(-1.2, 0.3, size=120)
    du_r = rng.normal(1.2, 0.3, size=120)
    dg_bar, sigma_bar = bennett_acceptance_ratio(du_f, du_r)
    dg_zw = zwanzig_free_energy(du_f)
    assert -1.5 <= dg_bar <= -0.9
    assert sigma_bar < 0.15
    assert math.isfinite(dg_zw)

    mol = parse_smiles("CC(=O)Nc1ccc(O)cc1")
    pocket = _sample_binding_pocket()
    fep_gate = AlchemicalFEPGateway(reference_delta_g_kcal_mol=2.0, target_delta_delta_g_threshold_kcal=-0.5)
    fep_res = fep_gate.evaluate_binding_free_energy(mol, pocket, seed=42)
    assert fep_res.converged is True
    assert fep_res.passed_gate is True

    admet_ref = ConformalADMETReferee(alpha=0.05)
    admet_res = admet_ref.evaluate_admet(mol)
    assert admet_res.passed_gate is True
    assert admet_res.herg_ic50_um.lower_bound >= 10.0


def test_tmdd_pkpd_problem_and_conservation_law():
    """Verify 2-compartment TMDD ODE specification and exact receptor mass conservation."""
    problem = load_tmdd_problem()
    assert problem.problem_id == "PS-TMDD"
    assert len(problem.states) == 4

    sim = simulate_tmdd_trajectory(dose_nmol_l=5.0, r_total=1.0, t_hours=24.0)
    assert sim["max_receptor_conservation_drift"] < 1e-12
    assert 0.25 <= sim["mean_receptor_occupancy"] <= 1.0


def test_end_to_end_6_layer_pipeline_and_wet_lab_holdout_burning():
    """Verify Proposer/Referee separation, 6-layer promotion, and anti-snooping HoldoutContaminationError."""
    pocket = _sample_binding_pocket()
    ledger = FullLineageExperimentLedger()
    proposer = MMLMMolecularProposer()
    fep_gate = AlchemicalFEPGateway(reference_delta_g_kcal_mol=2.0, target_delta_delta_g_threshold_kcal=-0.5)
    pipeline = DrugDiscoveryPipelineReferee(ledger=ledger, fep_gateway=fep_gate)

    proposals = proposer.generate_proposals_for_pocket(pocket, max_candidates=2)
    assert len(proposals) == 2

    # 1. Without wet-lab holdout, candidate is promoted ONLY to DEVELOPMENT_PASSED (in-silico lead)
    report_in_silico = pipeline.adjudicate_proposal(proposals[0], pocket)
    assert report_in_silico.in_silico_confirmed_lead is True
    assert report_in_silico.wet_lab_verified_drug is False
    assert report_in_silico.lifecycle_state == CandidateLifecycleState.DEVELOPMENT_PASSED

    # 2. Candidate failing wet-lab SPR/hERG holdout slice permanently burns that slice
    failing_holdout = WetLabAssayHoldoutSlice(
        dataset_id="SPR_KRAS_PANEL_2026",
        slice_id="BATCH_A_WELL_04",
        measured_kd_nm=950.0,  # Exceeds 250 nM threshold -> FAIL
        measured_herg_ic50_um=25.0,
    )
    report_failed_lab = pipeline.adjudicate_proposal(proposals[1], pocket, wet_lab_holdout=failing_holdout)
    assert report_failed_lab.wet_lab_verified_drug is False
    assert report_failed_lab.lifecycle_state == CandidateLifecycleState.CONFIRMATION_REJECTED

    # 3. Attempting to reuse the burned wet-lab holdout slice for a successor candidate MUST raise HoldoutContaminationError
    successor = MolecularCandidateProposal(
        candidate_id="POCK-KRAS-01_cand_3",
        proposer_id=proposer.proposer_id,
        smiles_or_selfies="CCNc1ccc(O)cc1",
        target_id=pocket.target_name,
        parent_candidate_id=proposals[1].candidate_id,
    )
    with pytest.raises(HoldoutContaminationError, match="DATA SNOOPING VIOLATION"):
        pipeline.adjudicate_proposal(successor, pocket, wet_lab_holdout=failing_holdout)


def test_direction_3_gbsa_implicit_solvation_and_vsa_synthon_generation():
    """Verify Direction 3: VSA synthon lead generation and analytical GB/SA implicit solvation."""
    pocket = _sample_binding_pocket()
    retro_ref = RetrosynthesisReferee()

    # 1. Generate de novo candidates via VSA synthon matching
    generator = VSASynthonDrugGenerator()
    proposals = generator.generate_proposals_for_pocket(pocket, max_candidates=3)
    assert len(proposals) == 3

    for prop in proposals:
        mol = parse_smiles(prop.smiles_or_selfies)
        assert len(mol.atoms) > 5

        # All generated candidates must pass Layer 2 RetrosynthesisGate by construction
        v = retro_ref.adjudicate(mol)
        assert v.passed is True, f"Candidate {prop.smiles_or_selfies} failed retrosynthesis gate"
        assert v.num_steps <= 2
        assert v.sa_score < 3.0

    # 2. Test analytical Hawkins-Cramer-Truhlar GB/SA implicit solvation on relaxed lead
    lead_mol = parse_smiles(proposals[0].smiles_or_selfies)
    dock_res = relax_ligand_in_pocket(lead_mol, pocket, n_steps=60)
    ff = build_pocket_force_field(lead_mol)

    gbsa = compute_gbsa_binding_energy(dock_res.ligand_coordinates, ff, pocket)
    assert gbsa.ligand_sasa_a2 > 50.0
    assert gbsa.buried_sasa_a2 >= 0.0
    assert gbsa.pocket_contact_atoms > 0
    assert abs(gbsa.delta_g_bind_kcal) < 1000.0, "GB/SA binding energy is finite and physical"
    # Hydrophobic burial in aqueous solvent provides a negative cavitation free energy
    assert gbsa.delta_g_sa_kcal <= 0.0

    # 3. Adjudicate candidate proposal through full 6-layer pipeline referee
    ledger = FullLineageExperimentLedger()
    fep_gate = AlchemicalFEPGateway(reference_delta_g_kcal_mol=2.0, target_delta_delta_g_threshold_kcal=-0.5)
    pipeline = DrugDiscoveryPipelineReferee(ledger=ledger, fep_gateway=fep_gate)
    report = pipeline.adjudicate_proposal(proposals[0], pocket)
    assert report.descriptors is not None
    assert report.retrosynthesis is not None and report.retrosynthesis.passed is True
    assert report.lifecycle_state in (
        CandidateLifecycleState.DEVELOPMENT_PASSED,
        CandidateLifecycleState.PRE_SCREEN_REJECTED,
    )
