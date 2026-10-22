"""Graph-census pins: close the three 'implemented but untested' graph slots.

The 2026-10-22 graph census found three graph surfaces implemented in the
repository but with ZERO test coverage. This module pins each one against
its own documented semantics (the census's gap-closure, not new claims):

1. ``apedp.scientific_layers.ProvenanceGraph`` — W3C PROV-lite evidence
   chain: nodes (kind, label, detail) and hyperedge-style links
   ``(src, rel, dst)`` are recorded verbatim.
2. ``plot_epistemic_graph`` — NetworkX render of a directed epistemic
   knowledge graph to PNG, deterministic layout seed.
3. ``akedp_kernel.rime_rewrite_json`` — the PyO3 FFI binding for the frozen
   structural rewrites (tested inside Rust only until now): a valid graph +
   ``WrapNonlinearity`` op appends the wrapped node and makes it the output
   (rewrite.rs ``apply`` semantics); a bad operand index is REJECTED
   (apply returns None → ValueError) and malformed JSON is rejected — a
   garbage graph can never silently mutate.
"""
from __future__ import annotations

import json

import pytest

import apedp_kernel as ak


# ── 1. W3C PROV-lite ProvenanceGraph ────────────────────────────────────
def test_provenance_graph_records_nodes_and_links_verbatim():
    from apedp.scientific_layers import ProvenanceGraph

    g = ProvenanceGraph()
    g.add("dataset", "obs_holdout", {"sha256": "deadbeef"})
    g.add("candidate", "rime_candidate_x", None)
    g.link("dataset", "DERIVED_FROM", "candidate")

    [(k1, l1, d1), (k2, l2, d2)] = [
        (n.kind, n.label, n.detail) for n in g.nodes
    ]
    assert (k1, l1, d1) == ("dataset", "obs_holdout", {"sha256": "deadbeef"})
    assert (k2, l2, d2) == ("candidate", "rime_candidate_x", {})
    assert g.edges == [("dataset", "DERIVED_FROM", "candidate")]

    # Independent nodes never share a detail dict (default_factory, not
    # a shared mutable default).
    g2 = ProvenanceGraph()
    g2.add("a", "A")
    g2.nodes[0].detail["leak"] = True
    assert "leak" not in (g.nodes[1].detail or {}), "mutable default leaked"


# ── 2. Epistemic knowledge-graph plotter ────────────────────────────────
def test_plot_epistemic_graph_renders_png(tmp_path):
    from apedp.visualization.plots import plot_epistemic_graph

    out = tmp_path / "epistemic_graph.png"
    path = plot_epistemic_graph(
        nodes=["hypothesis", "evidence", "referee", "holdout"],
        edges=[("hypothesis", "evidence"), ("evidence", "referee"),
               ("referee", "holdout")],
        output_path=str(out),
    )
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 0
    magic = out.read_bytes()[:8]
    assert magic == b"\x89PNG\r\n\x1a\n", f"not a PNG: {magic!r}"


# ── 3. PyO3 graph-rewrite FFI binding ───────────────────────────────────
def _graph_json() -> str:
    # Arena order: operands index strictly earlier nodes (topological).
    # nodes: 0=Const, 1=Input(0), 2=Mul(Const, Input); output = the product.
    return json.dumps({
        "nodes": [{"Const": 1.0}, {"Input": 0}, {"Mul": [0, 1]}],
        "output": 2,
    })


def test_rime_rewrite_json_ffi_wrap_nonlinearity():
    """Valid graph + WrapNonlinearity → appended Tanh node IS the output."""
    res = json.loads(ak.rime_rewrite_json(_graph_json(), json.dumps(
        {"WrapNonlinearity": {"src": 2}})))
    # apply() semantics: the new expression node becomes the output.
    # serde newtype-variant serialization: Tanh(usize) → {"Tanh": 2}.
    assert res["output"] == 3
    assert len(res["nodes"]) == 4
    assert res["nodes"][3] == {"Tanh": 2}
    # The original arena is preserved verbatim (pure functional rewrite).
    assert res["nodes"][2] == {"Mul": [0, 1]}


def test_rime_rewrite_json_ffi_rejects_bad_operand_index():
    """src out of range must REJECT, never fabricate a dangling reference."""
    with pytest.raises(ValueError, match="rewrite rejected"):
        ak.rime_rewrite_json(_graph_json(), json.dumps(
            {"WrapNonlinearity": {"src": 99}}))


def test_rime_rewrite_json_ffi_rejects_malformed_graph_json():
    with pytest.raises(ValueError, match="bad graph JSON"):
        ak.rime_rewrite_json("{not-json", json.dumps(
            {"WrapNonlinearity": {"src": 0}}))
    # Malformed op JSON likewise fails closed.
    with pytest.raises(ValueError, match="bad op JSON"):
        ak.rime_rewrite_json(_graph_json(), '{"NoSuchOp": 1}')
