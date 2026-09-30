"""Plotting utilities for APEDP analysis results."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def plot_regression_fitness(
    complexity: list[int],
    r_squared: list[float],
    output_path: str = "regression_fitness.png"
) -> str:
    """Plot model complexity vs R² for symbolic regression."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(complexity, r_squared, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Expression Complexity", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Symbolic Regression: Fitness Landscape", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_uncertainty_propagation(
    values: list[float],
    uncertainties: list[float],
    output_path: str = "uncertainty_propagation.png"
) -> str:
    """Plot measured values with uncertainty bars."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(values))
    ax.errorbar(x, values, yerr=uncertainties, fmt="o", capsize=5, capthick=2, linewidth=1.5)
    ax.set_xlabel("Measurement Index", fontsize=12)
    ax.set_ylabel("Value ± Uncertainty", fontsize=12)
    ax.set_title("Uncertainty Propagation", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_calibration_curve(
    x_values: list[float],
    y_measured: list[float],
    y_ideal: list[float],
    output_path: str = "calibration_curve.png"
) -> str:
    """Plot measured vs ideal calibration values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, y_ideal, "g--", label="Ideal", linewidth=2)
    ax.scatter(x_values, y_measured, c="red", s=50, zorder=5, label="Measured")
    ax.plot(x_values, y_measured, "r-", linewidth=1, alpha=0.7)
    ax.set_xlabel("Input", fontsize=12)
    ax.set_ylabel("Output", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_dimensional_analysis(
    variables: list[str],
    dimensions: list[str],
    output_path: str = "dimensional_analysis.png"
) -> str:
    """Visualize dimensional structure of variables."""
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(variables))
    ax.barh(y_pos, [1] * len(variables), color="steelblue", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{v}\n({d})" for v, d in zip(variables, dimensions, strict=False)], fontsize=10)
    ax.set_xlabel("Dimensional Presence", fontsize=12)
    ax.set_title("Dimensional Analysis Map", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_epistemic_graph(
    nodes: list[str],
    edges: list[tuple[str, str]],
    output_path: str = "epistemic_graph.png"
) -> str:
    """Plot epistemic knowledge graph as a network."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue",
            node_size=2000, font_size=8, arrows=True, arrowsize=20,
            edge_color="gray", linewidths=1)
    ax.set_title("Epistemic Knowledge Graph", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_scenario_results(
    scenario_names: list[str],
    pass_fail: list[bool],
    output_path: str = "scenario_results.png"
) -> str:
    """Visualize scenario suite pass/fail results."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["green" if p else "red" for p in pass_fail]
    ax.bar(range(len(scenario_names)), [1] * len(scenario_names), color=colors, alpha=0.8)
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel("Pass/Fail", fontsize=12)
    ax.set_title("APEDP Scenario Suite Results", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path