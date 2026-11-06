"""Real-time and search process visualization for APEDP."""
from __future__ import annotations

import time
from collections import deque

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class SearchProcessVisualizer:
    """Real-time visualization of search/discovery process."""
    
    def __init__(self, max_points: int = 1000, update_interval: float = 0.1):
        self.max_points = max_points
        self.update_interval = update_interval
        self.iterations = deque(maxlen=max_points)
        self.best_fitness = deque(maxlen=max_points)
        self.avg_fitness = deque(maxlen=max_points)
        self.diversity = deque(maxlen=max_points)
        self.exploration_rate = deque(maxlen=max_points)
        self.start_time = time.time()
    
    def update(self, iteration: int, best: float, avg: float, 
               diversity: float = None, exploration: float = None):
        """Update with current search state."""
        self.iterations.append(iteration)
        self.best_fitness.append(best)
        self.avg_fitness.append(avg)
        if diversity is not None:
            self.diversity.append(diversity)
        if exploration is not None:
            self.exploration_rate.append(exploration)
    
    def plot_search_progress(self, output_path: str = "search_progress.png") -> str:
        """Plot search progress over time."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Best fitness
        axes[0, 0].plot(list(self.iterations), list(self.best_fitness), 
                        'b-', linewidth=2, label='Best Fitness')
        axes[0, 0].set_title('Best Fitness Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Average fitness
        axes[0, 1].plot(list(self.iterations), list(self.avg_fitness),
                        'g-', linewidth=2, label='Average Fitness')
        axes[0, 1].set_title('Average Fitness Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Fitness')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Diversity
        if self.diversity:
            axes[1, 0].plot(list(self.iterations)[-len(self.diversity):], 
                           list(self.diversity), 'r-', linewidth=2)
            axes[1, 0].set_title('Population Diversity', fontweight='bold')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Diversity')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Exploration rate
        if self.exploration_rate:
            axes[1, 1].plot(list(self.iterations)[-len(self.exploration_rate):],
                           list(self.exploration_rate), 'm-', linewidth=2)
            axes[1, 1].set_title('Exploration vs Exploitation', fontweight='bold')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Exploration Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path
    
    def create_live_dashboard_html(self, output_path: str = "search_live_dashboard.html") -> str:
        """Create interactive real-time dashboard."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for interactive dashboards")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Fitness Evolution', 'Convergence Rate', 'Population Diversity',
                'Search Trajectory', 'Solution Quality Distribution', 'Time & Efficiency'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "indicator"}]
            ]
        )
        
        iters = list(self.iterations)
        if iters:
            fig.add_trace(go.Scatter(x=iters, y=list(self.best_fitness), 
                                    mode='lines', name='Best', 
                                    line=dict(color='blue', width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=iters, y=list(self.avg_fitness),
                                    mode='lines', name='Average',
                                    line=dict(color='green', width=2)), row=1, col=1)
            
            # Convergence rate
            if len(self.best_fitness) > 1:
                conv = np.diff(list(self.best_fitness))
                fig.add_trace(go.Scatter(x=iters[1:], y=conv, mode='lines',
                                        name='Improvement', line=dict(color='orange')),
                             row=1, col=2)
        
        if self.diversity:
            fig.add_trace(go.Scatter(x=iters[-len(self.diversity):], 
                                    y=list(self.diversity), mode='lines',
                                    name='Diversity', line=dict(color='red')), row=2, col=1)
        
        elapsed = time.time() - self.start_time
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=len(self.iterations),
            title={'text': "Iterations"},
            gauge={'axis': {'range': [None, max(100, len(self.iterations) * 1.2)]}},
        ), row=3, col=2)
        
        fig.update_layout(
            title_text="APEDP Real-Time Search Process Monitor",
            showlegend=True,
            height=1000,
            template="plotly_dark"
        )
        fig.write_html(output_path, auto_play=True)
        return output_path


class ProcessMonitor:
    """Monitor and visualize long-running processes."""
    
    def __init__(self):
        self.stages = []
        self.timestamps = []
        self.status = []
    
    def log_stage(self, stage_name: str, status: str = "running"):
        """Log a process stage."""
        self.stages.append(stage_name)
        self.timestamps.append(time.time())
        self.status.append(status)
    
    def plot_timeline(self, output_path: str = "process_timeline.png") -> str:
        """Plot process timeline."""
        if not self.stages:
            return output_path
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(self.stages) * 0.5)))
        y_pos = np.arange(len(self.stages))
        colors = []
        for s in self.status:
            if s == "completed":
                colors.append('#06D6A0')
            elif s == "running":
                colors.append('#FFD166')
            elif s == "failed":
                colors.append('#EF476F')
            else:
                colors.append('#118AB2')
        
        ax.barh(y_pos, [1] * len(self.stages), color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.stages)
        ax.set_xlabel("Progress", fontweight='bold')
        ax.set_title("Process Execution Timeline", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        
        # Add status labels
        for i, (stage, status) in enumerate(zip(self.stages, self.status)):
            ax.text(0.5, i, status.upper(), ha='center', va='center', 
                   fontweight='bold', color='white')
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path


def plot_search_trajectory(
    positions: list[tuple[float, float]],
    fitness: list[float],
    output_path: str = "search_trajectory.png",
) -> str:
    """Plot 2D search trajectory."""
    fig, ax = plt.subplots(figsize=(12, 10))
    x, y = zip(*positions)
    
    scatter = ax.scatter(x, y, c=fitness, s=80, cmap='plasma', alpha=0.7)
    ax.plot(x, y, '--', alpha=0.4, color='gray')
    
    # Mark start and end
    ax.scatter(x[0], y[0], s=200, color='green', marker='o', 
               label='Start', edgecolors='black', linewidths=2)
    ax.scatter(x[-1], y[-1], s=200, color='red', marker='*',
               label='End (Best)', edgecolors='black', linewidths=2)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness', fontweight='bold')
    
    ax.set_xlabel('Dimension 1', fontweight='bold')
    ax.set_ylabel('Dimension 2', fontweight='bold')
    ax.set_title('Search Trajectory Visualization', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
