"""
Fleet Scenarios: Multi-model GPU cluster configurations.

Each fleet scenario defines:
- A shared GPU cluster (total GPUs, total memory)
- 2-3 models that must train simultaneously on the cluster
- Each model has its own param count, layer distribution, and priority
- Loss trajectories per model for the oversight monitoring task

The fleet tasks test multi-agent cooperation, resource negotiation,
and fleet-wide oversight — core capabilities for Theme 1 (Multi-Agent).
"""
import math
import random

# ── Standard layer distributions for common architectures ────────────────────

DENSE_DISTRIBUTION = {
    "embedding": 0.150,
    "attention": 0.250,
    "ffn": 0.400,
    "layernorm": 0.002,
    "output": 0.198,
}

MOE_DISTRIBUTION = {
    "embedding": 0.080,
    "attention": 0.180,
    "ffn": 0.600,       # MoE models have much more FFN (expert layers)
    "layernorm": 0.002,
    "output": 0.138,
}

CODE_DISTRIBUTION = {
    "embedding": 0.120,
    "attention": 0.300,  # Code models tend to have heavier attention
    "ffn": 0.380,
    "layernorm": 0.002,
    "output": 0.198,
}


# ── Loss trajectory generators (for oversight monitoring task) ───────────────

def _smooth_trajectory(steps, base_loss=4.5, decay=0.025, noise=0.02, seed=42):
    """Generate a healthy, smoothly decaying loss trajectory."""
    rng = random.Random(seed)
    return [base_loss * math.exp(-decay * i) + rng.gauss(0, noise) for i in range(steps)]


def _crash_trajectory(steps, crash_step, base_loss=4.5, seed=42):
    """Generate a trajectory that crashes (NaN) at a specific step."""
    traj = _smooth_trajectory(steps, base_loss=base_loss, seed=seed)
    traj[crash_step] = 1e8  # Spike before crash
    for i in range(crash_step + 1, steps):
        traj[i] = float('nan')
    return traj


def _slow_diverge_trajectory(steps, diverge_start, base_loss=4.5, seed=42):
    """Generate a trajectory that slowly diverges (loss increases) from a point."""
    rng = random.Random(seed)
    traj = _smooth_trajectory(steps, base_loss=base_loss, seed=seed)
    for i in range(diverge_start, steps):
        traj[i] = traj[diverge_start - 1] + 0.08 * (i - diverge_start) + rng.gauss(0, 0.03)
    return traj


# ── Fleet Scenario Definitions ──────────────────────────────────────────────

FLEET_SCENARIOS = {
    # ── Small Fleet: 2 models, modest cluster ──
    "small_fleet": {
        "scenario_id": "small_fleet",
        "description": "2 models sharing a small 4-GPU cluster — tight memory, must cooperate",
        "cluster": {
            "total_gpus": 4,
            "gpu_memory_gb": 80,          # H100 80GB per GPU
            "total_memory_gb": 320,       # 4 * 80GB
            "cost_per_gpu_hour_usd": 3.0, # H100 cloud rate
        },
        "models": [
            {
                "model_id": "model_a",
                "name": "LLaMA-3-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 2,       # higher = more important
                "priority_reason": "Production fine-tune — shipping this week",
            },
            {
                "model_id": "model_b",
                "name": "CodeLLaMA-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": CODE_DISTRIBUTION,
                "priority": 1,
                "priority_reason": "Research experiment — flexible timeline",
            },
        ],
        "max_steps_per_task": 5,
    },

    # ── Medium Fleet: 3 models, mid-size cluster ──
    "medium_fleet": {
        "scenario_id": "medium_fleet",
        "description": "3 models sharing an 8-GPU cluster — requires careful resource balancing",
        "cluster": {
            "total_gpus": 8,
            "gpu_memory_gb": 80,
            "total_memory_gb": 640,
            "cost_per_gpu_hour_usd": 3.0,
        },
        "models": [
            {
                "model_id": "model_a",
                "name": "LLaMA-3-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 3,
                "priority_reason": "Customer-facing model — top priority",
            },
            {
                "model_id": "model_b",
                "name": "LLaMA-3-13B",
                "total_params": 13_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 2,
                "priority_reason": "Next-gen model — medium priority",
            },
            {
                "model_id": "model_c",
                "name": "CodeLLaMA-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": CODE_DISTRIBUTION,
                "priority": 1,
                "priority_reason": "Internal tooling — lowest priority",
            },
        ],
        "max_steps_per_task": 5,
    },

    # ── Large Fleet: 3 models including MoE, large cluster ──
    "large_fleet": {
        "scenario_id": "large_fleet",
        "description": "3 models (including MoE) on a 16-GPU cluster — high stakes, high complexity",
        "cluster": {
            "total_gpus": 16,
            "gpu_memory_gb": 80,
            "total_memory_gb": 1280,
            "cost_per_gpu_hour_usd": 3.0,
        },
        "models": [
            {
                "model_id": "model_a",
                "name": "LLaMA-3-13B",
                "total_params": 13_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 3,
                "priority_reason": "Flagship model — release deadline in 2 weeks",
            },
            {
                "model_id": "model_b",
                "name": "Mixtral-8x7B",
                "total_params": 46_000_000_000,
                "layer_distribution": MOE_DISTRIBUTION,
                "priority": 2,
                "priority_reason": "MoE experiment — needs heavy FFN optimization",
            },
            {
                "model_id": "model_c",
                "name": "Phi-3-Mini-3.8B",
                "total_params": 3_800_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 1,
                "priority_reason": "Edge deployment research — flexible deadline",
            },
        ],
        "max_steps_per_task": 5,
    },
}


# ── Fleet Oversight Scenarios (loss trajectories per model) ─────────────────
# Used by fleet_oversight task: oversight agent monitors multiple runs at once

FLEET_OVERSIGHT_SCENARIOS = {
    "one_crash_two_healthy": {
        "scenario_id": "one_crash_two_healthy",
        "description": "Model B crashes from FP8 embedding, other two are healthy",
        "fleet_id": "medium_fleet",
        "trajectories": {
            "model_a": {
                "loss": _smooth_trajectory(100, base_loss=4.2, seed=10),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_b": {
                "loss": _crash_trajectory(100, crash_step=35, base_loss=4.8, seed=20),
                "precision_config": {"embedding": "FP8", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "crashed",
            },
            "model_c": {
                "loss": _smooth_trajectory(100, base_loss=3.9, seed=30),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
        },
        "ground_truth": {
            "crashing_model": "model_b",
            "failure_step": 35,
            "cause_keywords": ["embedding", "fp8", "underflow", "model_b"],
        },
        "step_count": 100,
        "window_size": 20,
    },

    "two_crash_one_healthy": {
        "scenario_id": "two_crash_one_healthy",
        "description": "Model A crashes early, Model C diverges slowly, Model B is healthy",
        "fleet_id": "medium_fleet",
        "trajectories": {
            "model_a": {
                "loss": _crash_trajectory(100, crash_step=18, base_loss=4.5, seed=11),
                "precision_config": {"embedding": "FP32", "attention": "FP8", "ffn": "FP8", "layernorm": "FP8", "output": "FP32"},
                "status": "crashed",
            },
            "model_b": {
                "loss": _smooth_trajectory(100, base_loss=4.0, seed=22),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_c": {
                "loss": _slow_diverge_trajectory(100, diverge_start=50, base_loss=3.8, seed=33),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP16"},
                "status": "diverging",
            },
        },
        "ground_truth": {
            "crashing_model": "model_a",       # First to crash
            "failure_step": 18,
            "cause_keywords": ["attention", "fp8", "overflow", "model_a"],
            "secondary_issue": {
                "model": "model_c",
                "type": "divergence",
                "cause_keywords": ["output", "fp16", "precision"],
            },
        },
        "step_count": 100,
        "window_size": 20,
    },

    "all_healthy": {
        "scenario_id": "all_healthy",
        "description": "All three models training healthily — no instability",
        "fleet_id": "medium_fleet",
        "trajectories": {
            "model_a": {
                "loss": _smooth_trajectory(100, base_loss=4.2, seed=41),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_b": {
                "loss": _smooth_trajectory(100, base_loss=4.6, seed=42),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_c": {
                "loss": _smooth_trajectory(100, base_loss=3.5, seed=43),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
        },
        "ground_truth": {
            "crashing_model": None,
            "failure_step": -1,
            "cause_keywords": [],
        },
        "step_count": 100,
        "window_size": 20,
    },

    "late_crash_moe": {
        "scenario_id": "late_crash_moe",
        "description": "MoE model crashes late due to FP8 layernorm, others healthy",
        "fleet_id": "large_fleet",
        "trajectories": {
            "model_a": {
                "loss": _smooth_trajectory(100, base_loss=4.0, seed=51),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_b": {
                "loss": _crash_trajectory(100, crash_step=72, base_loss=5.2, seed=52),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "FP8", "output": "FP32"},
                "status": "crashed",
            },
            "model_c": {
                "loss": _smooth_trajectory(100, base_loss=3.6, seed=53),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
        },
        "ground_truth": {
            "crashing_model": "model_b",
            "failure_step": 72,
            "cause_keywords": ["layernorm", "fp8", "model_b", "moe"],
        },
        "step_count": 100,
        "window_size": 20,
    },
}
