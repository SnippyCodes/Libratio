"""
Task 2: Instability Detection scenarios.
"""
import math, random

WINDOW_SIZE = 20

def _smooth(steps, seed=42):
    random.seed(seed)
    return [4.5 * math.exp(-0.025 * i) + random.gauss(0, 0.02) for i in range(steps)]

def _crash_at(steps, crash_step, seed=42):
    traj = _smooth(steps, seed)
    traj[crash_step] = 1e8
    for i in range(crash_step + 1, steps):
        traj[i] = float('nan')
    return traj

def _gradual_diverge(steps, start, seed=42):
    random.seed(seed)
    traj = _smooth(steps, seed)
    for i in range(start, steps):
        traj[i] = traj[start - 1] + 0.05 * (i - start) + random.gauss(0, 0.03)
    return traj

def _spike_recovery(steps, spike, seed=42):
    traj = _smooth(steps, seed)
    traj[spike] = traj[spike] * 15
    traj[spike + 1] = traj[spike - 1] * 2
    for i in range(spike + 2, min(spike + 10, steps)):
        traj[i] = traj[i] * 1.3
    return traj

TASK2_SCENARIOS = {
    "stable_healthy_run": {
        "scenario_id": "stable_healthy_run",
        "training_loss_trajectory": _smooth(100),
        "precision_config": {"embedding": "FP32", "attention": "BF16", "layernorm": "BF16", "ffn": "FP16", "output": "FP32"},
        "step_count": 100,
        "ground_truth": {"is_unstable": False, "failure_step": -1, "cause_keywords": []}
    },
    "early_fp8_embedding_crash": {
        "scenario_id": "early_fp8_embedding_crash",
        "training_loss_trajectory": _crash_at(100, 22),
        "precision_config": {"embedding": "FP8", "attention": "BF16", "layernorm": "FP16", "ffn": "FP8", "output": "FP32"},
        "step_count": 100,
        "ground_truth": {"is_unstable": True, "failure_step": 22, "cause_keywords": ["embedding", "fp8", "underflow"]}
    },
    "late_attention_overflow": {
        "scenario_id": "late_attention_overflow",
        "training_loss_trajectory": _crash_at(100, 78),
        "precision_config": {"embedding": "FP32", "attention": "FP8", "layernorm": "BF16", "ffn": "FP8", "output": "FP32"},
        "step_count": 100,
        "ground_truth": {"is_unstable": True, "failure_step": 78, "cause_keywords": ["attention", "fp8", "overflow"]}
    },
    "gradual_output_divergence": {
        "scenario_id": "gradual_output_divergence",
        "training_loss_trajectory": _gradual_diverge(100, 55),
        "precision_config": {"embedding": "FP32", "attention": "BF16", "layernorm": "BF16", "ffn": "FP16", "output": "FP16"},
        "step_count": 100,
        "ground_truth": {"is_unstable": True, "failure_step": 55, "cause_keywords": ["output", "fp16", "precision"]}
    },
    "spike_then_recovery": {
        "scenario_id": "spike_then_recovery",
        "training_loss_trajectory": _spike_recovery(100, 38),
        "precision_config": {"embedding": "FP32", "attention": "BF16", "layernorm": "BF16", "ffn": "FP8", "output": "FP32"},
        "step_count": 100,
        "ground_truth": {"is_unstable": False, "failure_step": -1, "cause_keywords": []}
    },
}
