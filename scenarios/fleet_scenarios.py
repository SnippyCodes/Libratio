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

# ── New architecture distributions (Phase 1: scenario diversity) ─────────────

VISION_DISTRIBUTION = {
    "embedding": 0.200,   # Patch embedding is large in ViT-style models
    "attention": 0.350,   # ViT-style heavy self-attention
    "ffn": 0.300,
    "layernorm": 0.002,
    "output": 0.148,
}

MULTIMODAL_DISTRIBUTION = {
    "embedding": 0.250,   # Text + vision dual embeddings
    "attention": 0.200,   # Cross-attention is smaller per-param
    "ffn": 0.350,
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


def _intermittent_nan_trajectory(steps, nan_steps, base_loss=4.5, seed=42):
    """Generate a trajectory with intermittent NaN spikes that recover.

    Tests false-positive resistance: agent should NOT flag this as a crash
    because the model recovers each time. Real-world GPUs sometimes produce
    transient NaN from memory bit-flips or numerical edge cases.
    """
    traj = _smooth_trajectory(steps, base_loss=base_loss, seed=seed)
    for ns in nan_steps:
        if ns < steps:
            traj[ns] = float('nan')
            # Recovery: next values return to normal trajectory
            if ns + 1 < steps:
                traj[ns + 1] = traj[max(0, ns - 1)] * 1.05  # small spike on recovery
    return traj


def _cascading_failure_trajectory(steps, initial_crash, cascade_delay=8,
                                   base_loss=4.5, seed=42):
    """Generate trajectory where one crash triggers another model's OOM.

    When Model A crashes, its freed memory gets reallocated, causing
    memory pressure on Model B which then also crashes after a delay.
    """
    traj = _smooth_trajectory(steps, base_loss=base_loss, seed=seed)
    # Memory pressure spike before cascade crash
    cascade_step = initial_crash + cascade_delay
    if cascade_step < steps:
        traj[cascade_step - 1] = traj[cascade_step - 2] * 3.0  # pressure spike
        traj[cascade_step] = 1e8
        for i in range(cascade_step + 1, steps):
            traj[i] = float('nan')
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

    # ── Phase 1: New fleet configs for scenario diversity ──

    # Micro Fleet: extreme memory pressure, 2 tiny GPUs
    "micro_fleet": {
        "scenario_id": "micro_fleet",
        "description": "2 models on 2 GPUs — extreme memory pressure forces aggressive precision",
        "cluster": {
            "total_gpus": 2,
            "gpu_memory_gb": 40,           # A100 40GB variant
            "total_memory_gb": 80,
            "cost_per_gpu_hour_usd": 1.50,
        },
        "models": [
            {
                "model_id": "model_a",
                "name": "Phi-3-Mini-3.8B",
                "total_params": 3_800_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 2,
                "priority_reason": "Edge model fine-tune — needs to ship to mobile",
            },
            {
                "model_id": "model_b",
                "name": "ViT-Large-300M",
                "total_params": 300_000_000,
                "layer_distribution": VISION_DISTRIBUTION,
                "priority": 1,
                "priority_reason": "Vision backbone experiment — low priority",
            },
        ],
        "max_steps_per_task": 5,
    },

    # Mega Fleet: 4 models including 70B, scale test
    "mega_fleet": {
        "scenario_id": "mega_fleet",
        "description": "4 models (including 70B) on 32 GPUs — Meta-scale fleet management",
        "cluster": {
            "total_gpus": 32,
            "gpu_memory_gb": 80,
            "total_memory_gb": 2560,
            "cost_per_gpu_hour_usd": 3.00,
        },
        "models": [
            {
                "model_id": "model_a",
                "name": "LLaMA-3-70B",
                "total_params": 70_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 4,
                "priority_reason": "Flagship 70B — highest priority, CEO demo next week",
            },
            {
                "model_id": "model_b",
                "name": "LLaMA-3-13B",
                "total_params": 13_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 2,
                "priority_reason": "Distillation target — medium priority",
            },
            {
                "model_id": "model_c",
                "name": "LLaVA-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": MULTIMODAL_DISTRIBUTION,
                "priority": 2,
                "priority_reason": "Multimodal research — medium priority",
            },
            {
                "model_id": "model_d",
                "name": "CodeLLaMA-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": CODE_DISTRIBUTION,
                "priority": 1,
                "priority_reason": "Internal tooling experiment — lowest priority",
            },
        ],
        "max_steps_per_task": 5,
    },

    # Heterogeneous Fleet: mixed GPU types
    "heterogeneous_fleet": {
        "scenario_id": "heterogeneous_fleet",
        "description": "3 models on mixed A100+H100 cluster — different GPU capabilities",
        "cluster": {
            "total_gpus": 6,
            "gpu_memory_gb": 60,           # Average: mix of 40GB A100 + 80GB H100
            "total_memory_gb": 360,
            "cost_per_gpu_hour_usd": 2.50, # Blended rate
        },
        "models": [
            {
                "model_id": "model_a",
                "name": "LLaMA-3-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 3,
                "priority_reason": "Production model — needs H100 for FP8 support",
            },
            {
                "model_id": "model_b",
                "name": "CLIP-ViT-L",
                "total_params": 430_000_000,
                "layer_distribution": VISION_DISTRIBUTION,
                "priority": 1,
                "priority_reason": "Vision encoder — can run on A100",
            },
            {
                "model_id": "model_c",
                "name": "Mistral-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 2,
                "priority_reason": "Community fine-tune — medium priority",
            },
        ],
        "max_steps_per_task": 5,
    },

    # Oversubscribed Fleet: impossible to fit all at FP32
    "oversubscribed_fleet": {
        "scenario_id": "oversubscribed_fleet",
        "description": "3 large models on 4 GPUs — must use aggressive precision or some models don't fit",
        "cluster": {
            "total_gpus": 4,
            "gpu_memory_gb": 80,
            "total_memory_gb": 320,
            "cost_per_gpu_hour_usd": 3.00,
        },
        "models": [
            {
                "model_id": "model_a",
                "name": "LLaMA-3-13B",
                "total_params": 13_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 3,
                "priority_reason": "Revenue-critical model — must complete",
            },
            {
                "model_id": "model_b",
                "name": "Mixtral-8x7B",
                "total_params": 46_000_000_000,
                "layer_distribution": MOE_DISTRIBUTION,
                "priority": 2,
                "priority_reason": "MoE pre-training — needs maximum FP8",
            },
            {
                "model_id": "model_c",
                "name": "LLaMA-3-7B",
                "total_params": 7_000_000_000,
                "layer_distribution": DENSE_DISTRIBUTION,
                "priority": 1,
                "priority_reason": "Experimental run — can be killed if needed",
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

    # ── Phase 1: New oversight scenarios for diversity ──

    "cascading_failure": {
        "scenario_id": "cascading_failure",
        "description": "Model A crashes from FP8 embedding, Model C cascades into OOM 8 steps later",
        "fleet_id": "medium_fleet",
        "trajectories": {
            "model_a": {
                "loss": _crash_trajectory(100, crash_step=25, base_loss=4.3, seed=60),
                "precision_config": {"embedding": "FP8", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "crashed",
            },
            "model_b": {
                "loss": _smooth_trajectory(100, base_loss=4.1, seed=61),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_c": {
                "loss": _cascading_failure_trajectory(100, initial_crash=25, cascade_delay=8, base_loss=3.9, seed=62),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                "status": "cascade_crashed",
            },
        },
        "ground_truth": {
            "crashing_model": "model_a",       # Root cause is Model A
            "failure_step": 25,
            "cause_keywords": ["embedding", "fp8", "underflow", "model_a", "cascade"],
        },
        "step_count": 100,
        "window_size": 20,
    },

    "intermittent_nan_trap": {
        "scenario_id": "intermittent_nan_trap",
        "description": "Model B has transient NaN spikes but recovers — agent should NOT flag as crash",
        "fleet_id": "medium_fleet",
        "trajectories": {
            "model_a": {
                "loss": _smooth_trajectory(100, base_loss=4.0, seed=70),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_b": {
                "loss": _intermittent_nan_trajectory(100, nan_steps=[22, 45, 78], base_loss=4.5, seed=71),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",  # Recovers each time!
            },
            "model_c": {
                "loss": _smooth_trajectory(100, base_loss=3.7, seed=72),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
        },
        "ground_truth": {
            "crashing_model": None,  # No real crash! Flagging = false alarm
            "failure_step": -1,
            "cause_keywords": [],
        },
        "step_count": 100,
        "window_size": 20,
    },

    "slow_drift_no_crash": {
        "scenario_id": "slow_drift_no_crash",
        "description": "All models slowly diverge but none crash — tests patience vs trigger-happiness",
        "fleet_id": "large_fleet",
        "trajectories": {
            "model_a": {
                "loss": _slow_diverge_trajectory(100, diverge_start=60, base_loss=4.2, seed=80),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "diverging",
            },
            "model_b": {
                "loss": _slow_diverge_trajectory(100, diverge_start=70, base_loss=5.0, seed=81),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                "status": "diverging",
            },
            "model_c": {
                "loss": _slow_diverge_trajectory(100, diverge_start=55, base_loss=3.5, seed=82),
                "precision_config": {"embedding": "FP32", "attention": "FP16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                "status": "diverging",
            },
        },
        "ground_truth": {
            "crashing_model": None,  # Divergence != crash
            "failure_step": -1,
            "cause_keywords": [],
        },
        "step_count": 100,
        "window_size": 20,
    },

    "output_fp16_crash": {
        "scenario_id": "output_fp16_crash",
        "description": "Model C crashes from FP16 on output layer — different root cause than usual FP8 embedding",
        "fleet_id": "medium_fleet",
        "trajectories": {
            "model_a": {
                "loss": _smooth_trajectory(100, base_loss=4.1, seed=90),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_b": {
                "loss": _smooth_trajectory(100, base_loss=4.6, seed=91),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                "status": "healthy",
            },
            "model_c": {
                "loss": _crash_trajectory(100, crash_step=48, base_loss=3.8, seed=92),
                "precision_config": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP16"},
                "status": "crashed",
            },
        },
        "ground_truth": {
            "crashing_model": "model_c",
            "failure_step": 48,
            "cause_keywords": ["output", "fp16", "overflow", "model_c", "cross-entropy"],
        },
        "step_count": 100,
        "window_size": 20,
    },
}
