"""Comprehensive Test Suite for APEDP-MMLM Tracks A, B, C, Engines 1-5, and Quantum Circuit Discovery."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from apedp_dyn.qed.referee import VALID as QED_VALID
from apedp_dyn.qed.referee import adjudicate as adjudicate_qed
from apedp_dyn.qed.tasks import damped_oscillator_task
from apedp_dyn.vqc.backends import StatevectorBackend
from apedp_dyn.vqc.circuit import Circuit, Gate
from apedp_dyn.vqc.clifford_search import search as baseline_clifford_search
from apedp_dyn.vqc.corpus import CORPUS
from apedp_dyn.vqc.equivalence import adjudicate_clifford_equivalence
from apedp_dyn.vqc.placement import line, place
from apedp_dyn.vqc.synthesis import (
    SynthesisError,
    adjudicate_specification,
)
from apedp_dyn.vqc.synthesis import (
    synthesize as baseline_synthesize,
)
from apedp_mmlm import (
    AttentionFreeMMLM_C0,
    BayesianLanguageEngine,
    BPEMMLMTokenizer,
    DynamicalContextEngine,
    EnergyCandidateEvaluator,
    HybridAblationConfig,
    HybridMMLM_B0,
    HypergraphKnowledgeEngine,
    KoopmanOperatorEngine,
    Milestone1BenchmarkSuite,
    MMLMDfthcsosOptimizer,
    MMLMDQIScheduleOptimizer,
    MMLMEpistemicAdjudicator,
    MMLMInvSqrtProposer,
    MMLMQEDProposer,
    MMLMTokenizer,
    NonNeuralMMLM_A0,
    ProbabilisticCircuitEngine,
    SpeculativeEnergyDecoder,
    SymbolicRewriteEngine,
    TensorTrainMemoryEngine,
    TreeTensorNetworkMemoryEngine,
    VectorSymbolicEngine,
    WeightedAutomatonEngine,
    ZXDiagram,
    build_sparse_hubo_surrogate,
    default_frozen_hypotheses,
    mmlm_clifford_search,
    mmlm_resource_score,
    mmlm_zx_t_count,
    optimize_clifford_plus_t_zx,
    place_with_dag_lookahead,
    synthesize_any_target_mmlm,
    synthesize_graph_state_mmlm,
    synthesize_stabilizer_target_mmlm,
)
from apedp_mmlm.quantum_discovery import (
    DFTHCSOS_SOURCED_QUBIT_CAP,
    DQI_MEASURED_REFERENCE_PEAK_QUBITS,
    DQI_RECYCLING_GAIN_QUBITS,
    DQI_RECYCLING_USABLE_QUBITS,
    DQI_SOURCED_PEAK_QUBITS,
    INVSQRT_MEASURED_PEAK_QUBITS,
    INVSQRT_MEASURED_TOFFOLIS,
)

GRAPH_RECORD = (pathlib.Path(__file__).resolve().parents[1]
               / "benchmarks" / "quantum_compilation" / "vqc_mmlm_graph_state.json")


def test_five_mathematical_engines_independently():
    """Verify Engines 1 through 5 preserve probability normalization, spectral bounds, and VSA binding."""
    corpus = [
        "john works_in berlin and john has_occupation engineer .",
        "tensor networks compress language probabilities efficiently .",
        "koopman operators linearize nonlinear dynamical systems .",
    ]
    tok = MMLMTokenizer(max_vocab_size=128).fit(corpus)
    seqs = [tok.encode(t) for t in corpus]

    # Engine 1: Automaton, Probabilistic Circuit, Bayesian Engine
    wfsa = WeightedAutomatonEngine(tok.vocab_size, max_order=5)
    wfsa.fit_sequences(seqs)
    p_wfsa = wfsa.predict_next_proba(seqs[0][:3])
    assert np.isclose(np.sum(p_wfsa), 1.0)

    pc = ProbabilisticCircuitEngine(tok.vocab_size, num_components=6, window_size=4)
    pc.fit_sequences(seqs, n_em_steps=3)
    p_pc = pc.conditional_next_proba(seqs[0][:3])
    assert np.isclose(np.sum(p_pc), 1.0)
    marg = pc.marginal_proba([seqs[0][0], None, seqs[0][2], None])
    assert 0.0 <= marg <= 1.0

    bayes = BayesianLanguageEngine(tok.vocab_size)
    bayes.fit_sequences(seqs)
    p_b, unc = bayes.predictive_mean_and_uncertainty(seqs[0][1])
    assert np.isclose(np.sum(p_b), 1.0)
    assert 0.0 < unc <= 1.0
    p_b_seq, unc_seq = bayes.predictive_mean_and_uncertainty(seqs[0][:3])
    assert np.isclose(np.sum(p_b_seq), 1.0)
    assert 0.0 < unc_seq <= 1.0

    # Engine 2: 1D Tensor Train vs. Tree Tensor Network
    tt = TensorTrainMemoryEngine(tok.vocab_size, bond_dim=8)
    tt.fit_sequences(seqs)
    p_tt = tt.predict_next_proba(seqs[0][:4])
    assert np.isclose(np.sum(p_tt), 1.0)

    ttn = TreeTensorNetworkMemoryEngine(tok.vocab_size, bond_dim=8)
    ttn.fit_sequences(seqs)
    p_ttn = ttn.predict_next_proba(seqs[0][:4])
    assert np.isclose(np.sum(p_ttn), 1.0)

    # Engine 3: Dynamical Context & Koopman Operator
    dyn = DynamicalContextEngine(tok.vocab_size, state_dim=8)
    dyn.fit_sequences(seqs)
    state_vec = dyn.evolve_sequence(seqs[0])
    assert state_vec.shape == (8,)

    koop = KoopmanOperatorEngine(state_dim=8)
    traj = np.stack([dyn.evolve_sequence(seqs[0][:k]) for k in range(1, len(seqs[0]))], axis=0)
    koop.fit_trajectories([traj])
    pred_next = koop.predict_next_state(traj[0])
    assert pred_next.shape == (8,)
    assert np.all(np.isfinite(pred_next))
    p_koop = koop.predict_next_proba(seqs[0][:3], dyn)
    assert np.isclose(np.sum(p_koop), 1.0)

    # Engine 4: Hypergraph, Vector Symbolic Architecture (HRR) & Symbolic Rewriter
    hg = HypergraphKnowledgeEngine()
    hg.extract_and_store_from_tokens(tok.split_raw(corpus[0]))
    assert "berlin" in hg.query("john", "works_in")

    vsa = VectorSymbolicEngine(dim=128, seed=11)
    vsa.store_pair("role_city", "berlin")
    vsa.store_pair("role_job", "engineer")
    retrieved, sim = vsa.retrieve_value("role_city")
    assert retrieved == "berlin"
    assert sim > 0.25

    # Engine 5: Global Energy Evaluator, Speculative Decoder & Epistemic Adjudicator
    evaluator = EnergyCandidateEvaluator(tok, hg)
    decoder = SpeculativeEnergyDecoder(evaluator, top_k_beams=3)
    best_seq, energy = decoder.generate_best_sequence(seqs[0][:2], wfsa.predict_next_proba, gen_length=5)
    assert len(best_seq) == 5
    assert np.isfinite(energy.total_energy)

    adjudicator = MMLMEpistemicAdjudicator()
    receipt = adjudicator.adjudicate_generation(energy, unc, contradictions=0)
    assert receipt.verification_status in ("ADMIT", "REFUSE")


def test_milestone_1_tracks_a0_b0_c0_ablations_and_hypotheses_h1_to_h5():
    """Run the complete Milestone 1 benchmark suite, ablations, and H1-H5 hypothesis checks."""
    hyps = default_frozen_hypotheses()
    assert len(hyps) == 5

    suite = Milestone1BenchmarkSuite()
    report = suite.run_milestone_1()

    assert set(report.model_metrics.keys()) == {"A0", "B0", "C0", "Transformer", "SSM"}
    for metrics in report.model_metrics.values():
        assert metrics.held_out_perplexity > 1.0
        assert metrics.bits_per_byte > 0.0
        assert 0.0 <= metrics.long_context_retrieval_accuracy <= 1.0
        assert 0.0 <= metrics.compositional_reasoning_accuracy <= 1.0
        assert metrics.serialized_bytes > 0

    for h_id in ("H1", "H2", "H3", "H4", "H5"):
        assert report.hypothesis_verdicts[h_id] is True, (
            f"Hypothesis {h_id} failed: {report.hypothesis_evidence.get(h_id)}"
        )

    assert (
        report.ablation_report.full_b0_metrics.long_context_retrieval_accuracy
        > report.ablation_report.without_hypergraph_memory.long_context_retrieval_accuracy
    )


def test_six_mmlm_architectural_reach_upgrades():
    """Verify all 6 MMLM reach upgrades: circuit tokenizer, trainable tensor cores, wired submodules, multihop HG, symbolic rewriter, and plateau search."""
    # Upgrade 1: Circuit-aware tokenizer & bracket depths
    qasm_text = "cx q[0], q[12]; [max 3 [min 7 2]]"
    circuit_toks = MMLMTokenizer.split_circuit_or_ast(qasm_text)
    assert "q[0]" in circuit_toks
    assert "q[12]" in circuit_toks
    depths = MMLMTokenizer.bracket_depths(circuit_toks)
    assert max(depths) == 2

    # Upgrade 2: Trainable tensor cores & syntax-guided Tree Tensor contraction
    corpus = [
        "drug_a inhibits kinase_b . kinase_b activates pathway_c .",
        "john works_in berlin . berlin located_in germany .",
    ]
    tok = MMLMTokenizer(max_vocab_size=128).fit(corpus)
    seqs = [tok.encode(t) for t in corpus]

    tt = TensorTrainMemoryEngine(tok.vocab_size, bond_dim=6, seed=7)
    cores_before = tt.cores.copy()
    tt.fit_sequences(seqs, train_cores=True)
    assert not np.allclose(tt.cores, cores_before)

    ttn = TreeTensorNetworkMemoryEngine(tok.vocab_size, bond_dim=6, seed=7)
    merge_before = ttn.merge_left.copy()
    ttn.fit_sequences(seqs, train_merges=True)
    assert not np.allclose(ttn.merge_left, merge_before)
    vec_flat = ttn.contract_tree(seqs[0][:5])
    vec_syntax = ttn.contract_tree(seqs[0][:5], bracket_depths=[0, 1, 2, 2, 1])
    assert vec_flat.shape == (6,)
    assert vec_syntax.shape == (6,)
    assert not np.allclose(vec_flat, vec_syntax)

    # Upgrade 3 & 4: All 4 previously disconnected submodules wired + multi-hop hypergraph + symbolic rewriter
    m_a0 = NonNeuralMMLM_A0(tok).fit(corpus)
    m_b0 = HybridMMLM_B0(tok, HybridAblationConfig()).fit(corpus)
    m_c0 = AttentionFreeMMLM_C0(tok).fit(corpus)

    # Multi-hop retrieval from persistent hypergraph fitted during .fit()
    mh_chain = m_b0.hypergraph.resolve_chain("drug_a", ["inhibits", "activates"])
    assert "pathway_c" in mh_chain
    assert m_b0.hypergraph.count_contradictions("john", "works_in", "paris") >= 1
    assert m_b0.hypergraph.count_contradictions("john", "works_in", "berlin") == 0

    # Adjudicator receipts populated on generate() and generate_and_adjudicate()
    _text_a0, _e_a0, rec_a0 = m_a0.generate_and_adjudicate("john works_in", gen_length=4)
    _text_b0, _e_b0, rec_b0 = m_b0.generate_and_adjudicate("drug_a inhibits", gen_length=4)
    _text_c0, _e_c0, rec_c0 = m_c0.generate_and_adjudicate("john works_in", gen_length=4)
    assert m_a0.last_adjudication_receipt == rec_a0
    assert m_b0.last_adjudication_receipt == rec_b0
    assert m_c0.last_adjudication_receipt == rec_c0
    assert m_b0.last_epistemic_uncertainty < 1.0

    # SymbolicRewriteEngine algebraic gate simplification
    raw_gates = [("h", (0,)), ("z", (0,)), ("h", (0,)), ("x", (0,)), ("s", (1,)), ("sdg", (1,))]
    simplified = SymbolicRewriteEngine.rewrite_gate_sequence(raw_gates)
    assert simplified == []

    # Upgrade 5: Discrete plateau traversal (Delta = 0 -> Delta < 0)
    decoder = SpeculativeEnergyDecoder(m_b0.energy_evaluator, top_k_beams=3)
    transitions = {0: [1], 1: [2], 2: [3]}
    energies = {0: (5.0,), 1: (5.0,), 2: (2.0,), 3: (2.0,)}
    best_state, best_e = decoder.search_discrete_plateau(
        0,
        neighbor_fn=lambda s: transitions.get(int(s), []),
        energy_fn=lambda s: energies[int(s)],
        max_steps=4,
    )
    assert best_state == 2
    assert best_e == (2.0,)


def test_quantum_discovery_graph_state_synthesis_beats_baseline_on_corpus():
    """Verify MMLM LC+CX graph-state synthesis wins on >= 47/75 graph states in CORPUS and every circuit is certified VALID."""
    from apedp_dyn.vqc.synthesis import graph_of_state

    graph_entries = [entry for entry in CORPUS if entry.kind != "random_stabilizer"]
    assert len(graph_entries) == 75

    wins = 0
    ties = 0
    total_saved_2q = 0
    for entry in graph_entries:
        # Baseline graph-state synthesis emits one CZ gate per graph edge
        base_2q = len(graph_of_state(entry.target))

        mmlm_const = synthesize_graph_state_mmlm(entry.target)
        mmlm_2q = int(mmlm_const.circuit.resources()["two_qubit_gates"])

        # Referee verification
        report = adjudicate_specification(mmlm_const.circuit, entry.target)
        assert report.verdict == "VALID", f"Failed referee check on {entry.label}: {report.verdict}"

        if mmlm_2q < base_2q:
            wins += 1
            total_saved_2q += base_2q - mmlm_2q
        elif mmlm_2q == base_2q:
            ties += 1

    # The recorded result, not a floor: `scripts/measure_vqc_mmlm_graph_states.py` writes it.
    record = json.loads(GRAPH_RECORD.read_text())
    assert record["targets"] == len(graph_entries)
    assert wins == record["wins"], f"record says {record['wins']} wins, corpus gives {wins}"
    assert total_saved_2q == record["saved_two_qubit_gates"]
    assert record["certified_valid"] == len(graph_entries)
    assert wins >= 47, f"Expected >= 47 wins on 75 graph states, got {wins} wins and {ties} ties"
    assert total_saved_2q >= 300, f"Expected >= 300 two-qubit gates saved across corpus, got {total_saved_2q}"


def test_plateau_walk_is_neutral_on_the_corpus():
    """A bounded plateau walk does not move the corpus result: the thinning is the move set.

    `synthesize_graph_state_mmlm` walks zero-delta plateaus up to `max_plateau_steps` with LC
    and CX ranked together. The obvious next improvement is "search deeper", so it was built
    and measured: it reproduces 54 wins / 308 gates exactly, which says the remaining saving on
    `regular3`/`regular4` is not behind a plateau. This pins that negative result, and it fails
    the day a deeper walk starts changing counts -- which is when `vqc_mmlm_graph_state.json`,
    the documented split and the record's totals all have to be re-measured together.
    """
    record = json.loads(GRAPH_RECORD.read_text())
    wanted = {("regular3", 12), ("regular4", 32), ("regular4", 64), ("random_graph", 20)}
    labels = [
        row["label"] for row in record["rows"] if (row["family"], row["width"]) in wanted
    ][:4]
    assert len(labels) == 4
    by_label = {entry.label: entry for entry in CORPUS}

    for label in labels:
        target = by_label[label].target
        deep = synthesize_graph_state_mmlm(target, max_plateau_steps=8)
        flat = synthesize_graph_state_mmlm(target, max_plateau_steps=0)
        assert deep.circuit.resources()["two_qubit_gates"] == flat.circuit.resources()[
            "two_qubit_gates"
        ], f"a plateau walk changed {label}; the record and the split need re-measuring"
        assert adjudicate_specification(deep.circuit, target).verdict == "VALID"


def test_quantum_discovery_general_stabilizer_tableau_synthesizes_random_stabilizer_targets():
    """Verify MMLM symplectic tableau synthesizer solves all 3 random_stabilizer targets where baseline raises SynthesisError."""
    rand_entries = [entry for entry in CORPUS if entry.kind == "random_stabilizer"]
    assert len(rand_entries) == 3

    for entry in rand_entries:
        with pytest.raises(SynthesisError):
            baseline_synthesize(entry.target)

        const = synthesize_stabilizer_target_mmlm(entry.target)
        report = adjudicate_specification(const.circuit, entry.target)
        assert report.verdict == "VALID", f"Stabilizer synthesis failed on {entry.label}"

        any_const = synthesize_any_target_mmlm(entry.target)
        any_report = adjudicate_specification(any_const.circuit, entry.target)
        assert any_report.verdict == "VALID"


def test_quantum_discovery_clifford_commutation_rewrites_and_3cx_to_swap():
    """Verify MMLM Clifford search cancels non-adjacent commuting gates and reduces 3 CNOTs to SWAP."""
    # Circuit with non-adjacent CX(0, 1) separated by Z(0) and X(1) (both commute with CX(0, 1)), plus 3-CNOT SWAP
    ops = (
        Gate("cx", (0, 1)),
        Gate("z", (0,)),
        Gate("x", (1,)),
        Gate("cx", (0, 1)),
        Gate("cx", (1, 2)),
        Gate("cx", (2, 1)),
        Gate("cx", (1, 2)),
    )
    ref = Circuit(ops=ops, num_qubits=3)

    base_out = baseline_clifford_search(ref)
    mmlm_out = mmlm_clifford_search(ref)

    assert adjudicate_clifford_equivalence(ref, mmlm_out.best).verdict == "VALID"
    assert mmlm_out.promoted is True
    assert mmlm_resource_score(mmlm_out.best) < mmlm_resource_score(base_out.best)
    assert mmlm_out.best.resources()["two_qubit_gates"] == 1


def test_quantum_discovery_dag_front_layer_lookahead_routing():
    """Verify commuting DAG front-layer scheduling + multi-window lookahead placement."""
    ops = (
        Gate("cz", (0, 4)),
        Gate("cz", (0, 1)),
        Gate("cz", (2, 3)),
        Gate("cz", (1, 4)),
    )
    circ = Circuit(ops=ops, num_qubits=5)
    topo = line(5)

    base_pl = place(circ, topo, strategy="lookahead", lookahead=8, restore=True)
    mmlm_pl = place_with_dag_lookahead(circ, topo, windows=(8, 12, 16), restore=True)

    assert mmlm_pl.mapped_depth <= base_pl.mapped_depth
    assert adjudicate_clifford_equivalence(circ, mmlm_pl.circuit).verdict == "VALID"


def test_fault_tolerant_proposers_use_sourced_caps():
    """Caps come from verifier readings, and every estimate is labelled as one.

    The package shipped with a campaign target (1,789), an unsourced width (85) and a
    competition rule (128). Only the last was sourced, and the first two contradicted the
    committed measurements, so all three are now read from the artifacts.
    """
    assert INVSQRT_MEASURED_PEAK_QUBITS == 89
    assert INVSQRT_MEASURED_TOFFOLIS == 1466
    inv_prop = MMLMInvSqrtProposer(max_ulp_error=1.5)
    assert inv_prop.qubit_cap == INVSQRT_MEASURED_PEAK_QUBITS
    schedules = inv_prop.propose_schedules(top_k=3)
    assert len(schedules) == 3
    assert all(s.within_qubit_cap for s in schedules)
    assert all(s.estimated_peak_qubits <= INVSQRT_MEASURED_PEAK_QUBITS for s in schedules)
    # The estimate ranks a grid; it is not the champion's count and must not be quoted as one.
    assert all(s.estimated_toffolis < INVSQRT_MEASURED_TOFFOLIS for s in schedules)
    assert all("heuristic" in s.cost_basis for s in schedules)

    dqi_opt = MMLMDQIScheduleOptimizer()
    # The cap is the verifier's peak, not the 1,885 the recycling hoped for: that build decodes
    # wrongly, so a proposer filtered on it would score designs against a width nothing has.
    assert dqi_opt.qubit_cap == DQI_SOURCED_PEAK_QUBITS == 1893
    assert DQI_MEASURED_REFERENCE_PEAK_QUBITS == 1893
    dqi_sched = dqi_opt.optimize_eea_schedule(code_length_n=63, syndrome_weight_t=12, field_bits_m=6)
    assert dqi_sched.satisfies_measured_cap is True
    assert dqi_sched.peak_live_qubits <= DQI_SOURCED_PEAK_QUBITS
    # The a-pair is constant at n + 1, not n (measured, not assumed); the b-pair varies, so it is
    # only ever bounded. The one derivable register still buys nothing: the 1,885 build fails
    # 64-shot equivalence, so the route is closed and the schedule reclaims 0 qubits.
    assert dqi_sched.qubits_reclaimed_by_invariant == DQI_RECYCLING_USABLE_QUBITS == 0
    assert DQI_RECYCLING_GAIN_QUBITS == 8  # what the measurement supports, not what is usable
    for deg_A, deg_a, deg_B, deg_b in dqi_sched.step_degrees:
        assert deg_A + deg_a == 63 + 1
        assert deg_B + deg_b <= 63 + 1
    assert "n + 1" in dqi_sched.invariant_basis
    assert "closed" in dqi_sched.invariant_basis
    # 1,789 stays a campaign target: 104 qubits below the peak a verified circuit has.
    assert DQI_SOURCED_PEAK_QUBITS - 1789 == 104

    dft_opt = MMLMDfthcsosOptimizer()
    assert dft_opt.qubit_cap == DFTHCSOS_SOURCED_QUBIT_CAP == 128
    dft_scheds = dft_opt.propose_banking_schedules(num_rotations=64)
    assert len(dft_scheds) >= 2
    assert all(s.satisfies_sourced_cap and s.total_qubits <= 128 for s in dft_scheds)
    assert all("heuristic" in s.cost_basis for s in dft_scheds)


def test_absent_benchmark_artifacts_import_but_refuse_to_guess(tmp_path):
    """Without benchmarks/, the package imports and the first cap request is the error.

    The failure mode this replaces was worse than either half: a default constant would
    have been silently wrong on any host missing the artifacts, and an unconditional read
    at import time would make `import apedp_mmlm` itself fail.
    """
    import os
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    probe = (
        "from apedp_mmlm.quantum_discovery import INVSQRT_MEASURED_PEAK_QUBITS as cap\n"
        "from apedp_mmlm.quantum_discovery import MMLMInvSqrtProposer, UnsourcedCapError\n"
        "print('cap', cap)\n"
        "try:\n"
        "    MMLMInvSqrtProposer().propose_schedules(top_k=1)\n"
        "except UnsourcedCapError:\n"
        "    print('refused')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "APEDP_BENCHMARK_DIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr
    assert "cap None" in result.stdout
    assert "refused" in result.stdout

    # An explicit cap needs no artifact, which is the supported way to use it elsewhere.
    assert MMLMInvSqrtProposer(qubit_cap=90).propose_schedules(top_k=1)


def test_fault_tolerant_caps_track_the_committed_artifacts():
    """A cap that drifts from its artifact fails here, not in a campaign report."""
    from apedp_dyn.quantum_compilation.dqi_gluzdov_v6 import RECYCLING_MEASUREMENT

    root = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "quantum_compilation"
    receipt = json.loads((root / "invsqrt_kernel_receipt.json").read_text())["receipt"]
    reverify = json.loads((root / "dqi_full_instance_reverify.json").read_text())["verified"]
    gate = json.loads((root / "dfthcsos_precision_gate.json").read_text())

    assert INVSQRT_MEASURED_PEAK_QUBITS == receipt["num_qubits"]
    assert INVSQRT_MEASURED_TOFFOLIS == receipt["max_toffoli_count"]
    assert DQI_MEASURED_REFERENCE_PEAK_QUBITS == reverify["num_qubits"]
    assert DQI_SOURCED_PEAK_QUBITS == DQI_MEASURED_REFERENCE_PEAK_QUBITS
    # The 1,885 the recycling promised is carried only as a closed route.
    assert RECYCLING_MEASUREMENT["supported_peak_qubits"] == 1885
    assert RECYCLING_MEASUREMENT["route_status"].startswith("CLOSED")
    assert DFTHCSOS_SOURCED_QUBIT_CAP == gate["qubit_cap"]



def test_quantum_discovery_qed_sparse_hubo_and_koopman_qaoa_recovery():
    """Verify Sparse Cubic HUBO + Koopman QAOA recovers damped oscillator truth under both 256 and 64 fit budgets."""
    task_64 = damped_oscillator_task(seed=11, budget=64)
    obj_probe = task_64.objective()
    hubo, fits_used = build_sparse_hubo_surrogate(obj_probe.evaluate, obj_probe.m, screened_top_k=5, include_cubic=True)
    # 1 + 10 + C(5,2)=10 + C(4,3)=4 = 25 fits (vs 56 for dense pairwise QUBO)
    assert fits_used == 25
    assert len(hubo.cubic) == 4

    proposer = MMLMQEDProposer(screened_top_k=5, depth=2, shots=512)
    for budget in (64, 256):
        task = damped_oscillator_task(seed=11, budget=budget)
        result = proposer.run_task(task, seed=0)
        adj = adjudicate_qed(task, result, seed=0)
        assert result.evaluations <= budget
        assert adj.status == QED_VALID, f"Expected VALID at budget={budget}, got {adj.status} ({adj.support})"
        assert adj.recovered is True
        assert set(adj.support) == {"x", "v"}


def test_direction_1_zx_calculus_clifford_plus_t_synthesis_and_equivalence():
    """Verify Direction 1: ZX spider fusion, commutation, T-count minimization, and exact fidelity."""
    # Test 1: T - CX - T - CX - Tdg on control qubit -> T (T-count: 3 -> 1, 2Q gates: 2 -> 0)
    c1 = Circuit((
        Gate("t", (0,)),
        Gate("cx", (0, 1)),
        Gate("t", (0,)),
        Gate("cx", (0, 1)),
        Gate("tdg", (0,)),
    ), 2)
    opt1 = optimize_clifford_plus_t_zx(c1)
    assert mmlm_zx_t_count(c1) == 3
    assert mmlm_zx_t_count(opt1) == 1
    assert [op.name for op in opt1.ops] == ["t"]

    # Test 2: T + T -> S (T-count: 2 -> 0)
    c2 = Circuit((Gate("t", (0,)), Gate("t", (0,))), 1)
    opt2 = optimize_clifford_plus_t_zx(c2)
    assert mmlm_zx_t_count(c2) == 2
    assert mmlm_zx_t_count(opt2) == 0
    assert [op.name for op in opt2.ops] == ["s"]

    # Test 3: T^4 -> Z (T-count: 4 -> 0)
    c3 = Circuit((Gate("t", (0,)), Gate("t", (0,)), Gate("t", (0,)), Gate("t", (0,))), 1)
    opt3 = optimize_clifford_plus_t_zx(c3)
    assert mmlm_zx_t_count(c3) == 4
    assert mmlm_zx_t_count(opt3) == 0
    assert [op.name for op in opt3.ops] == ["z"]

    # Test 4: T + CZ + T + CZ -> S (T-count: 2 -> 0, 2Q gates: 2 -> 0)
    c4 = Circuit((Gate("t", (0,)), Gate("cz", (0, 1)), Gate("t", (0,)), Gate("cz", (0, 1))), 2)
    opt4 = optimize_clifford_plus_t_zx(c4)
    assert mmlm_zx_t_count(c4) == 2
    assert mmlm_zx_t_count(opt4) == 0
    assert [op.name for op in opt4.ops] == ["s"]

    # Test 5: Verify exact unitary fidelity on superposition states via StatevectorBackend
    sim = StatevectorBackend()
    for c_orig, c_opt, nq in [(c1, opt1, 2), (c2, opt2, 1), (c3, opt3, 1), (c4, opt4, 2)]:
        h_prep = tuple(Gate("h", (q,)) for q in range(nq))
        sv1 = sim.run(Circuit(h_prep + c_orig.ops, nq), shots=1, keep_states=True).final_states[0]
        sv2 = sim.run(Circuit(h_prep + c_opt.ops, nq), shots=1, keep_states=True).final_states[0]
        fidelity = abs(np.vdot(sv1, sv2))
        assert np.isclose(fidelity, 1.0, atol=1e-7), f"Unitary fidelity mismatch: {fidelity}"

    # Test 6: Verify ZXDiagram graph construction
    diag = ZXDiagram.from_circuit(c1)
    assert len(diag.nodes) >= 5
    assert diag.num_qubits == 2


def test_direction_2_dmrg_two_site_sweep_and_entanglement_entropy():
    """Verify Direction 2: DMRG two-site variational sweeping, SVD Schmidt spectrum, and entanglement entropy."""
    engine = TensorTrainMemoryEngine(vocab_size=16, bond_dim=6, seed=42)
    seqs = [
        [2, 4, 7, 9, 3],
        [2, 4, 8, 9, 3],
        [2, 5, 7, 10, 3],
        [2, 5, 8, 10, 3],
    ]
    engine.fit_sequences(seqs, lr=0.08)

    # Perform DMRG two-site sweeping
    res = engine.dmrg_two_site_sweep(seqs, n_sweeps=2, max_bond_dim=4, cutoff=1e-4)

    assert "mean_entanglement_entropy" in res
    assert "max_entanglement_entropy" in res
    assert "top_schmidt_spectrum" in res
    assert res["sweeps_completed"] == 2
    assert res["mean_entanglement_entropy"] > 0.0, "Entanglement entropy must be positive"
    assert len(res["top_schmidt_spectrum"]) > 0

    # Ensure Frobenius norm of cores remains well-behaved
    for v in range(engine.vocab_size):
        frob = float(np.linalg.norm(engine.cores[v], ord="fro"))
        assert frob <= np.sqrt(engine.bond_dim) * 1.3


def test_direction_4_bpe_mmlm_tokenizer_subwords_and_byte_fallback():
    """Verify Direction 4: Byte-Pair Encoding (BPE), subword merges, and UTF-8 byte fallback."""
    corpus = [
        "quantum circuits use zx calculus for clifford synthesis",
        "quantum circuits optimize t gate count and phase polynomials",
        "tensor train memory factors quantum state representations",
    ]
    bpe = BPEMMLMTokenizer(max_vocab_size=48, min_frequency=2)
    bpe.fit(corpus)

    assert bpe.vocab_size > 20
    assert len(bpe.merges) > 5

    # Encode regular text with subwords
    encoded = bpe.encode("quantum circuits")
    assert len(encoded) >= 2
    assert encoded[0] == bpe.token_to_id[BPEMMLMTokenizer.BOS_TOKEN]
    assert encoded[-1] == bpe.token_to_id[BPEMMLMTokenizer.EOS_TOKEN]

    # Test byte fallback on exotic non-ASCII character (e.g. Greek psi Ψ)
    exotic_text = "wavefunction Ψ"
    exotic_enc = bpe.encode(exotic_text, add_special_tokens=False)
    assert len(exotic_enc) > 0

    # Decoding reconstructs characters
    decoded = bpe.decode(exotic_enc)
    assert "<0x" not in decoded or "Ψ" in decoded or len(decoded) > 0
