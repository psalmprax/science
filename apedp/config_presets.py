# Config presets for APEDP
PRESETS = {
    "auto": {
        "auto_select_proposer": True,
        "auto_resource_detect": True,
        "mcts": True,
        "neural_guidance": "auto",
        "pareto": True,
        "parallel": True,
        "gpu": "auto",
        "llm": False,
        "cost_optimization": True
    },
    "simple": {
        "auto_select_proposer": True,
        "mcts": False,
        "neural_guidance": False,
        "pareto": True,
        "parallel": True,
        "gpu": False,
        "llm": False
    },
    "balanced": {
        "auto_select_proposer": True,
        "mcts": True,
        "neural_guidance": False,
        "pareto": True,
        "parallel": True,
        "gpu": False,
        "llm": False
    },
    "thorough": {
        "auto_select_proposer": True,
        "mcts": True,
        "neural_guidance": True,
        "pareto": True,
        "parallel": True,
        "gpu": True,
        "llm": False
    }
}


def get_preset(name: str = "auto"):
    """Get preset configuration with fallback to auto."""
    return PRESETS.get(name, PRESETS["auto"])

