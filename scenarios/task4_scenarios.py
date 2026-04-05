"""
Task 4: Precision Transfer scenarios.
The agent receives a working precision config from a SOURCE model and must
adapt it to work on a TARGET model with different size/constraints.

This tests generalization: can the agent reason about HOW precision choices
interact with model scale, rather than just memorizing rules?
"""

TASK4_SCENARIOS = {
    "small_to_large": {
        "scenario_id": "small_to_large",
        "description": "Transfer config from 7B to 70B — memory becomes the bottleneck",
        "source_model": {
            "name": "LLaMA-3-7B",
            "total_params": 7_000_000_000,
            "layer_distribution": {
                "embedding": 0.150, "attention": 0.250,
                "ffn": 0.400, "layernorm": 0.002, "output": 0.198,
            },
            "working_config": {
                "embedding": "FP32", "attention": "FP32",
                "ffn": "BF16", "layernorm": "FP32", "output": "FP32",
            },
            "metrics": {
                "memory_gb": 22.4, "accuracy": 0.998, "speedup_vs_fp32": 1.22,
                "training_days": 5.7,
            },
        },
        "target_model": {
            "name": "LLaMA-3-70B",
            "total_params": 70_000_000_000,
            "layer_distribution": {
                "embedding": 0.120, "attention": 0.300,
                "ffn": 0.420, "layernorm": 0.002, "output": 0.158,
            },
            "constraints": {
                "memory_budget_gb": 180.0,
                "time_budget_days": 25.0,
                "accuracy_threshold": 0.96,
            },
        },
        "max_iterations": 3,
    },
    "large_to_small": {
        "scenario_id": "large_to_small",
        "description": "Transfer config from 70B to 3B edge device — can afford higher precision",
        "source_model": {
            "name": "LLaMA-3-70B",
            "total_params": 70_000_000_000,
            "layer_distribution": {
                "embedding": 0.120, "attention": 0.300,
                "ffn": 0.420, "layernorm": 0.002, "output": 0.158,
            },
            "working_config": {
                "embedding": "FP32", "attention": "BF16",
                "ffn": "FP8", "layernorm": "BF16", "output": "FP32",
            },
            "metrics": {
                "memory_gb": 160.7, "accuracy": 0.991, "speedup_vs_fp32": 1.81,
                "training_days": 38.6,
            },
        },
        "target_model": {
            "name": "Gemma-2-3B",
            "total_params": 3_000_000_000,
            "layer_distribution": {
                "embedding": 0.180, "attention": 0.220,
                "ffn": 0.380, "layernorm": 0.003, "output": 0.217,
            },
            "constraints": {
                "memory_budget_gb": 10.0,
                "time_budget_days": 2.0,
                "accuracy_threshold": 0.98,
            },
        },
        "max_iterations": 3,
    },
    "same_family_upgrade": {
        "scenario_id": "same_family_upgrade",
        "description": "Transfer from LLaMA-2-13B to LLaMA-3-13B — subtle architecture changes",
        "source_model": {
            "name": "LLaMA-2-13B",
            "total_params": 13_000_000_000,
            "layer_distribution": {
                "embedding": 0.160, "attention": 0.240,
                "ffn": 0.390, "layernorm": 0.002, "output": 0.208,
            },
            "working_config": {
                "embedding": "FP32", "attention": "BF16",
                "ffn": "BF16", "layernorm": "BF16", "output": "FP32",
            },
            "metrics": {
                "memory_gb": 28.6, "accuracy": 0.995, "speedup_vs_fp32": 1.65,
                "training_days": 7.9,
            },
        },
        "target_model": {
            "name": "LLaMA-3-13B",
            "total_params": 13_000_000_000,
            "layer_distribution": {
                "embedding": 0.140, "attention": 0.280,
                "ffn": 0.400, "layernorm": 0.002, "output": 0.178,
            },
            "constraints": {
                "memory_budget_gb": 25.0,
                "time_budget_days": 6.0,
                "accuracy_threshold": 0.97,
            },
        },
        "max_iterations": 3,
    },
    "dense_to_moe": {
        "scenario_id": "dense_to_moe",
        "description": "Transfer from 13B dense to 47B MoE — FFN layer distribution shifts dramatically",
        "source_model": {
            "name": "LLaMA-3-13B-Dense",
            "total_params": 13_000_000_000,
            "layer_distribution": {
                "embedding": 0.150, "attention": 0.250,
                "ffn": 0.400, "layernorm": 0.002, "output": 0.198,
            },
            "working_config": {
                "embedding": "FP32", "attention": "BF16",
                "ffn": "FP8", "layernorm": "BF16", "output": "FP32",
            },
            "metrics": {
                "memory_gb": 29.8, "accuracy": 0.991, "speedup_vs_fp32": 1.81,
                "training_days": 7.0,
            },
        },
        "target_model": {
            "name": "Mixtral-47B-MoE",
            "total_params": 47_000_000_000,
            "layer_distribution": {
                "embedding": 0.080, "attention": 0.180,
                "ffn": 0.580, "layernorm": 0.002, "output": 0.160,
            },
            "constraints": {
                "memory_budget_gb": 100.0,
                "time_budget_days": 15.0,
                "accuracy_threshold": 0.95,
            },
        },
        "max_iterations": 3,
    },
}
