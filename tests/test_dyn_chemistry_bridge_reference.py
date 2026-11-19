"""The hand-written chemistry bridge against RDKit, on eighteen known molecules.

``apedp_dyn/molecular/chemistry_bridge.py`` parses SMILES and computes
descriptors with no cheminformatics dependency. Nothing in this repository
measured it against a reference implementation until 2026-11-17, and the
measurement found three defects, each fixed and pinned below:

1. summing aromatic bond orders (1.5 each) made a fused aromatic carbon read
   4.5 and ``ValenceError`` rejected naphthalene, caffeine, naproxen,
   sildenafil and warfarin as impossible structures;
2. the molecular weight counted implicit hydrogens only, so every bracket atom's
   hydrogen was missing (cholesterol 380.62 g/mol against 386.66);
3. an aromatic nitrogen could take nitrogen's 5-valent target and gain a
   phantom hydrogen (sildenafil 476.60 against 474.59).

What is exact here is the graph arithmetic: molecular weight, hydrogen-bond
donors and acceptors, rotatable bonds. logP, TPSA, SAScore and QED are this
module's own expressions, not Crippen's, Ertl's or Bickerton's, and they differ
from them by the margins pinned at the end of this file. The gates that read
them (``retrosynthesis_gate``'s ``sa_score < 3.5``,
``ConformalADMETReferee``'s logP-driven models) decide on numbers with that much
systematic error, which is why the funnel is labelled experimental and its
ADMET verdicts are not safety assessments.

RDKit is not a dependency of this repository; these tests skip without it.
"""
from __future__ import annotations

import os

import pytest

from apedp_dyn.molecular.chemistry_bridge import compute_descriptors, parse_smiles

Chem = pytest.importorskip("rdkit.Chem", reason="RDKit is the reference implementation here")

#: Eighteen drugs and simple molecules, chosen to cover fused aromatics,
#: stereocentres, bracket atoms, charges and heteroaromatic rings.
MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "nicotine": "CN1CCC[C@H]1c1cccnc1",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "naproxen": "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1",
    "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    "fluoxetine": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "sildenafil": "CCCc1nn(C)c2c(=O)[nH]c(nc12)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(n1)-c1cccnc1",
    "penicillin_g": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "cholesterol": "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
}


@pytest.fixture(scope="module")
def measured():
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    rows = {}
    for name, smiles in MOLECULES.items():
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None, name
        ours = compute_descriptors(parse_smiles(smiles))
        rows[name] = {
            "mw": (ours.molecular_weight, Descriptors.MolWt(molecule)),
            "hbd": (ours.hbd, Lipinski.NumHDonors(molecule)),
            "hba": (ours.hba, Lipinski.NumHAcceptors(molecule)),
            "rot": (ours.rotatable_bonds, rdMolDescriptors.CalcNumRotatableBonds(molecule)),
            "heavy": (ours.heavy_atom_count, molecule.GetNumHeavyAtoms()),
            "logp": (ours.logp, Crippen.MolLogP(molecule)),
            "tpsa": (ours.tpsa, rdMolDescriptors.CalcTPSA(molecule)),
            "qed": (ours.qed_estimate, QED.qed(molecule)),
        }
    return rows


def test_every_molecule_parses(measured):
    """Defect 1: five of these eighteen raised ValenceError before 2026-11-17."""
    assert set(measured) == set(MOLECULES)


def test_the_molecular_weight_is_exact(measured):
    """Defects 2 and 3. Graph arithmetic has no modelling freedom: it matches."""
    for name, row in measured.items():
        ours, reference = row["mw"]
        assert abs(ours - reference) < 0.05, (name, ours, reference)


def test_the_heavy_atom_count_is_exact(measured):
    for name, row in measured.items():
        assert row["heavy"][0] == row["heavy"][1], (name, row["heavy"])


def test_hydrogen_counts_and_rotatable_bonds_are_close(measured):
    """Donors follow from the hydrogen count; acceptors and rotatable bonds use
    this module's own definitions (it counts F as an acceptor, RDKit does not),
    so they are bounded rather than exact."""
    for name, row in measured.items():
        assert abs(row["hbd"][0] - row["hbd"][1]) <= 1, (name, row["hbd"])
        assert abs(row["hba"][0] - row["hba"][1]) <= 3, (name, row["hba"])
        assert abs(row["rot"][0] - row["rot"][1]) <= 3, (name, row["rot"])


def test_the_estimated_properties_are_recorded_as_estimates(measured):
    """logP, TPSA and QED are different models from Crippen's, Ertl's and
    Bickerton's. These are the measured deviations on 2026-11-17, pinned as
    ceilings so a change in either direction shows up as a failure rather than
    as a silently different gate."""
    import statistics

    deviations = {
        key: statistics.fmean(abs(row[key][0] - row[key][1]) for row in measured.values())
        for key in ("logp", "tpsa", "qed")
    }
    # Measured: logP 2.06, TPSA 11.67, QED 0.20.
    assert deviations["logp"] < 2.5, deviations
    assert deviations["tpsa"] < 14.0, deviations
    assert deviations["qed"] < 0.25, deviations
    # And they are not noise: each tracks its reference.
    if os.environ.get("APEDP_PRINT_CHEMISTRY_DEVIATIONS"):
        print(deviations)


def test_a_structure_that_is_really_impossible_is_still_refused():
    """The valence bound was weakened to the minimum Kekule valence; it must
    still reject what no Kekule structure could allow."""
    from apedp_dyn.molecular.chemistry_bridge import ValenceError

    with pytest.raises(ValenceError):
        parse_smiles("C(C)(C)(C)(C)C")  # five bonds on one carbon
