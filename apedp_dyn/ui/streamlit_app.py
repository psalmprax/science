"""Interactive Streamlit Epistemic Referee Studio & DAG Explorer for APEDP-X / CPMD.

Run with:
    streamlit run apedp_dyn/ui/streamlit_app.py
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import pandas as pd  # type: ignore[import-untyped]
    import streamlit as st  # type: ignore[import-untyped]
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    FullLineageExperimentLedger,
)
from apedp_dyn.ui.dag_explorer import (
    build_burned_holdouts_view,
    compute_pareto_frontier,
    extract_dag_lineage,
)


def render_dashboard(ledger_path: str | Path | None = None) -> None:
    """Render Streamlit web dashboard for experiment ledger."""
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Install with: pip install streamlit")
        return

    st.set_page_config(page_title="APEDP-X Epistemic Studio", layout="wide")
    st.title("🔬 APEDP-X Epistemic Discovery & Referee Studio")
    st.markdown("### Anti-Snooping Referee Adjudication, DAG Lineages & Pareto Frontiers")

    path = Path(ledger_path or os.environ.get("APEDP_STORAGE_DIR", "artifacts/storage")) / "experiment_ledger.jsonl"

    if not path.is_file():
        st.warning(f"No experiment ledger found at `{path}`. Please run discovery sweep first.")
        return

    ledger = FullLineageExperimentLedger(persistence_path=path)
    candidates = ledger.get_all_candidates()
    burned = ledger.get_burned_holdouts()

    col1, col2, col3, col4 = st.columns(4)
    promoted = [c for c in candidates if c.state == CandidateLifecycleState.CONFIRMED_VALID]
    rejected = [c for c in candidates if "REJECTED" in str(c.state) or "FAILED" in str(c.state)]

    col1.metric("Total Hypotheses", len(candidates))
    col2.metric("Promoted Active", len(promoted))
    col3.metric("Referee Refuted", len(rejected))
    col4.metric("Burned Holdout Slices", len(burned))

    tab1, tab2, tab3 = st.tabs(["🌳 Hypothesis Mutation DAG", "⚖️ Pareto Frontier", "🔥 Burned Holdouts Registry"])

    with tab1:
        st.subheader("Candidate Hypothesis Mutation Graph")
        dag = extract_dag_lineage(ledger)
        st.json(dag.to_dict())

    with tab2:
        st.subheader("Multi-Objective Pareto Trade-Offs")
        pareto = compute_pareto_frontier(candidates, ("dsr", "rmse"), (True, False))
        st.write(f"Found {len(pareto)} Pareto-optimal candidates (Maximizing DSR, Minimizing RMSE):")
        for p in pareto:
            st.markdown(f"- **[{p.candidate_id}]** `{p.formula}` (DSR: {p.metrics.get('dsr', 'N/A')}, RMSE: {p.metrics.get('rmse', 'N/A')})")

    with tab3:
        st.subheader("Burned Confirmation Holdout Slices (Anti-Snooping Firewall)")
        view = build_burned_holdouts_view(ledger)
        if view:
            df = pd.DataFrame(view)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No confirmation holdout slices have been burned.")


if __name__ == "__main__":
    render_dashboard()
