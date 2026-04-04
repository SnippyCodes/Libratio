"""
Task 3: Multi-Objective Optimization scenarios.
Constraints are calibrated to be achievable with the optimal precision strategy.

Physics calibration:
  Best achievable memory (FP32 emb+output, BF16 attn+layernorm, FP8 ffn):
    = (0.15*4 + 0.25*2 + 0.40*1 + 0.002*2 + 0.198*4) * params_B GB
    = 2.296 * params_B GB
  Best achievable time (normalized, 1 day per billion params at FP32 baseline):
    = params_B / avg_speedup (~1.87x for optimal mixed config)
    ≈ params_B * 0.535 days

All budgets set ABOVE minimum achievable to create a meaningful optimization challenge
without making the task impossible.
"""

LAYER_DISTRIBUTION = {
    "embedding": 0.150,
    "attention":  0.250,
    "ffn":        0.400,
    "layernorm":  0.002,
    "output":     0.198,
}

TASK3_SCENARIOS = {
    "datacenter_gpu_cluster": {
        "scenario_id": "datacenter_gpu_cluster",
        "description": "12B parameter model on A100 cluster — maximize efficiency within budget",
        "total_params": 12_000_000_000,
        # Best possible memory: 2.296 * 12 = 27.6GB  → budget set to 35GB (25% headroom)
        # Best possible time:  12 / 1.87 = 6.4 days  → budget set to 10 days
        "constraints": {
            "memory_budget_gb": 35.0,
            "time_budget_days": 10.0,
            "accuracy_threshold": 0.99,
        },
        "layer_distribution": LAYER_DISTRIBUTION,
        "max_iterations": 5,
    },
    "edge_device_8gb": {
        "scenario_id": "edge_device_8gb",
        "description": "3B parameter model for edge deployment — must fit in 8GB device memory",
        "total_params": 3_000_000_000,
        # Best possible memory: 2.296 * 3 = 6.9GB  → budget set to 8GB (tight!)
        # Best possible time:  3 / 1.87 = 1.6 days → budget set to 3 days
        "constraints": {
            "memory_budget_gb": 8.0,
            "time_budget_days": 3.0,
            "accuracy_threshold": 0.95,
        },
        "layer_distribution": LAYER_DISTRIBUTION,
        "max_iterations": 5,
    },
    "time_critical_rush": {
        "scenario_id": "time_critical_rush",
        "description": "5B model with strict 3-day deadline — minimize training time",
        "total_params": 5_000_000_000,
        # Best possible memory: 2.296 * 5 = 11.5GB → budget set to 20GB (comfortable)
        # Best possible time:  5 / 1.87 = 2.7 days → budget set to 3.5 days (tight!)
        "constraints": {
            "memory_budget_gb": 20.0,
            "time_budget_days": 3.5,
            "accuracy_threshold": 0.97,
        },
        "layer_distribution": LAYER_DISTRIBUTION,
        "max_iterations": 5,
    },
    "accuracy_paramount": {
        "scenario_id": "accuracy_paramount",
        "description": "7B medical AI model — accuracy is critical, resources are flexible",
        "total_params": 7_000_000_000,
        # Best possible memory: 2.296 * 7 = 16.1GB → budget set to 30GB (comfortable)
        # Best possible time:  7 / 1.87 = 3.7 days → budget set to 10 days (comfortable)
        "constraints": {
            "memory_budget_gb": 30.0,
            "time_budget_days": 10.0,
            "accuracy_threshold": 0.995,  # Very high accuracy requirement
        },
        "layer_distribution": LAYER_DISTRIBUTION,
        "max_iterations": 5,
    },
    "everything_tight": {
        "scenario_id": "everything_tight",
        "description": "4B model — memory, time, AND accuracy are all tightly constrained",
        "total_params": 4_000_000_000,
        # Best possible memory: 2.296 * 4 = 9.2GB  → budget set to 12GB (tight!)
        # Best possible time:  4 / 1.87 = 2.1 days → budget set to 2.8 days (tight!)
        "constraints": {
            "memory_budget_gb": 12.0,
            "time_budget_days": 2.8,
            "accuracy_threshold": 0.97,  # Medium-high accuracy requirement
        },
        "layer_distribution": LAYER_DISTRIBUTION,
        "max_iterations": 5,
    },
}
