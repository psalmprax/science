"""Enhanced plotting utilities for APEDP analysis results."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.bbox"] = "tight"


def plot_regression_fitness(
    complexity: list[int],
    r_squared: list[float],
    output_path: str = "regression_fitness.png",
) -> str:
    """Plot model complexity vs R² for symbolic regression."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(complexity, r_squared, "o-", linewidth=2.5, markersize=10, 
            color="#2E86AB", markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(complexity, r_squared, alpha=0.2, color="#2E86AB")
    ax.set_xlabel("Expression Complexity", fontsize=12, fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
    ax.set_title("Symbolic Regression: Fitness vs Complexity", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_uncertainty_propagation(
    values: list[float],
    uncertainties: list[float],
    labels: list[str] | None = None,
    output_path: str = "uncertainty_propagation.png",
) -> str:
    """Plot measured values with uncertainty bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(values))
    ax.errorbar(x, values, yerr=uncertainties, fmt="o", capsize=6, capthick=2, 
                linewidth=2, markersize=10, color="#A23B72", ecolor="#F18F01")
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Measurement", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value ± Uncertainty", fontsize=12, fontweight="bold")
    ax.set_title("Uncertainty Propagation Analysis", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_calibration_curve(
    x_values: list[float],
    y_measured: list[float],
    y_ideal: list[float] | None = None,
    output_path: str = "calibration_curve.png",
) -> str:
    """Plot measured vs ideal calibration values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    if y_ideal is None:
        y_ideal = x_values
    ax.plot(x_values, y_ideal, "--", label="Perfect Calibration", linewidth=2.5, color="#06A77D")
    ax.scatter(x_values, y_measured, s=80, zorder=5, label="Measured", color="#D62828", alpha=0.8)
    ax.plot(x_values, y_measured, "-", linewidth=1.5, alpha=0.6, color="#D62828")
    ax.set_xlabel("Predicted/Expected", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed/Measured", fontsize=12, fontweight="bold")
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_dimensional_analysis(
    variables: list[str],
    dimensions: list[str],
    output_path: str = "dimensional_analysis.png",
) -> str:
    """Visualize dimensional structure of variables."""
    fig, ax = plt.subplots(figsize=(12, max(6, len(variables) * 0.4)))
    y_pos = np.arange(len(variables))
    ax.barh(y_pos, [1] * len(variables), color=sns.color_palette("viridis", len(variables)), alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{v}  [{d}]" for v, d in zip(variables, dimensions, strict=False)], fontsize=10)
    ax.set_xlabel("Dimensional Consistency", fontsize=12, fontweight="bold")
    ax.set_title("Dimensional Analysis Map", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="x")
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_epistemic_graph(
    nodes: list[str],
    edges: list[tuple[str, str]],
    node_colors: dict[str, str] | None = None,
    output_path: str = "epistemic_graph.png",
) -> str:
    """Plot epistemic knowledge graph as a network."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    if node_colors:
        colors = [node_colors.get(n, "#87CEEB") for n in G.nodes()]
    else:
        colors = "#87CEEB"
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=3000, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#666666", arrows=True, 
                          arrowsize=25, width=2, alpha=0.6, ax=ax,
                          connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    
    ax.set_title("Epistemic Knowledge Graph", fontsize=16, fontweight="bold")
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_scenario_results(
    scenario_names: list[str],
    pass_fail: list[bool],
    scores: list[float] | None = None,
    output_path: str = "scenario_results.png",
) -> str:
    """Visualize scenario suite pass/fail results."""
    fig, ax = plt.subplots(figsize=(max(12, len(scenario_names) * 0.8), 7))
    colors = ["#06D6A0" if p else "#EF476F" for p in pass_fail]
    bars = ax.bar(range(len(scenario_names)), [1] * len(scenario_names), color=colors, alpha=0.85, edgecolor='black')
    
    if scores:
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_x() + bar.get_width()/2., 0.5, f'{score:.2f}', 
                    ha='center', va='center', fontweight='bold', fontsize=9)
    
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel("Status", fontsize=12, fontweight="bold")
    ax.set_title(f"APEDP Scenario Suite Results ({sum(pass_fail)}/{len(pass_fail)} Passed)", 
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_heatmap(
    data: np.ndarray | list[list[float]],
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
    title: str = "Correlation Heatmap",
    output_path: str = "heatmap.png",
    cmap: str = "coolwarm",
) -> str:
    """Plot correlation/heatmap visualization."""
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(max(10, data.shape[1] * 0.8), max(8, data.shape[0] * 0.6)))
    sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, center=0,
                xticklabels=x_labels, yticklabels=y_labels,
                cbar_kws={"label": "Value"}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_pareto_front(
    objectives_x: list[float],
    objectives_y: list[float],
    labels: list[str] | None = None,
    output_path: str = "pareto_front.png",
) -> str:
    """Plot multi-objective Pareto front."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(objectives_x, objectives_y, s=100, alpha=0.6, color="#457B9D", label="Candidates")
    
    # Find and highlight Pareto frontier
    points = sorted(zip(objectives_x, objectives_y), key=lambda p: p[0])
    pareto_x, pareto_y = [], []
    best_y = float('inf')
    for x, y in points:
        if y < best_y:
            pareto_x.append(x)
            pareto_y.append(y)
            best_y = y
    
    ax.plot(pareto_x, pareto_y, 'r-', linewidth=3, marker='o', markersize=12, 
            label="Pareto Frontier", markerfacecolor='white', markeredgewidth=2)
    
    if labels:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (objectives_x[i], objectives_y[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel("Objective 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("Objective 2", fontsize=12, fontweight="bold")
    ax.set_title("Pareto Frontier Analysis", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str = "3D Surface Plot",
    output_path: str = "surface3d.png",
) -> str:
    """Plot 3D surface visualization."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.9, 
                          linewidth=0, antialiased=True)
    ax.set_xlabel('X', fontsize=11, fontweight="bold")
    ax.set_ylabel('Y', fontsize=11, fontweight="bold")
    ax.set_zlabel('Z', fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_interactive_dashboard(
    data: dict,
    output_path: str = "dashboard.html",
) -> str:
    """Create interactive Plotly dashboard if available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly required for interactive dashboards. Install with: pip install plotly")
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Metric Trends', 'Distribution', 'Correlation', 'Performance'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "heatmap"}, {"type": "bar"}]]
    )
    
    # Add sample traces based on data
    if 'x' in data and 'y' in data:
        fig.add_trace(go.Scatter(x=data['x'], y=data['y'], mode='lines+markers', 
                                 name='Trend'), row=1, col=1)
    
    if 'values' in data:
        fig.add_trace(go.Histogram(x=data['values'], name='Distribution'), row=1, col=2)
    
    fig.update_layout(title_text="APEDP Interactive Dashboard", showlegend=False, 
                      height=800, template="plotly_white")
    fig.write_html(output_path)
    return output_path


def plot_violin_distribution(
    data: dict[str, list[float]],
    title: str = "Distribution Comparison",
    output_path: str = "violin.png",
) -> str:
    """Plot violin plots for distribution comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    df_data = []
    labels = []
    for label, values in data.items():
        df_data.extend(values)
        labels.extend([label] * len(values))
    
    import pandas as pd
    df = pd.DataFrame({'Value': df_data, 'Category': labels})
    sns.violinplot(data=df, x='Category', y='Value', ax=ax, palette='Set2')
    sns.stripplot(data=df, x='Category', y='Value', ax=ax, color='black', alpha=0.3, size=3)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
