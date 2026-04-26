"""
FastAPI application for Libratio Fleet OpenEnv.
Includes API endpoints and a lightweight frontend for demos.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.mixed_precision_env import MixedPrecisionEnvironment
from environment.fleet_env import FleetEnvironment
from server.models import (
    ResetRequest, ResetResponse, StepResponse, StateResponse,
    TaskDefinition, RewardPayload,
)

app = FastAPI(
    title="Libratio Fleet — Multi-Agent GPU Fleet Management",
    description="OpenEnv for multi-agent GPU cluster precision optimization and fleet oversight",
    version="3.0.0"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
COLAB_GRAPHS_DIR = ROOT_DIR / "colab_graphs"
HF_GRAPHS_DIR = ROOT_DIR / "hf_graphs"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
if COLAB_GRAPHS_DIR.exists():
    app.mount("/colab_graphs", StaticFiles(directory=str(COLAB_GRAPHS_DIR)), name="colab_graphs")
if HF_GRAPHS_DIR.exists():
    app.mount("/hf_graphs", StaticFiles(directory=str(HF_GRAPHS_DIR)), name="hf_graphs")

# Single-agent environment (backward compatible)
solo_env = MixedPrecisionEnvironment()

# Multi-agent fleet environment (new)
fleet_env = FleetEnvironment()


def _clamp_score(raw_score) -> float:
    """Defense-in-depth: clamp score to strict (0.01, 0.99) interval."""
    try:
        val = float(raw_score)
    except (ValueError, TypeError):
        val = 0.01
    return float(max(0.01, min(0.99, val)))


@app.get("/")
def root():
    if FRONTEND_DIR.exists():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    return {
        "status": "ok",
        "environment": "Libratio Fleet — Multi-Agent GPU Fleet Management",
        "version": "3.0.0",
        "modes": {
            "solo": "Original single-agent precision environment (/reset, /step, /state, /tasks)",
            "fleet": "Multi-agent fleet environment (/fleet/reset, /fleet/step, /fleet/state, /fleet/tasks)",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "Libratio Fleet — Multi-Agent GPU Fleet Management",
        "version": "3.0.0",
    }


# ═══════════════════════════════════════════════════════════
# SOLO MODE — Original single-agent endpoints (backward compatible)
# ═══════════════════════════════════════════════════════════

@app.get("/tasks", response_model=List[TaskDefinition])
def get_tasks() -> List[TaskDefinition]:
    return [TaskDefinition(**t) for t in solo_env.TASK_DEFS]


@app.post("/reset", response_model=ResetResponse)
def reset_environment(payload: dict = {}):
    task_id = payload.get("task_id", "precision_assignment")
    obs = solo_env.reset(task_id)
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step_environment(payload: dict = {}):
    action = payload.get("action", payload)
    result = solo_env.step(action)
    clamped_score = _clamp_score(result["reward"]["score"])
    return StepResponse(
        observation=result["observation"],
        reward=RewardPayload(
            score=clamped_score,
            feedback=result["reward"]["feedback"],
        ),
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/state", response_model=StateResponse)
def get_state():
    s = solo_env.state()
    return StateResponse(**s)


# ═══════════════════════════════════════════════════════════
# FLEET MODE — Multi-agent fleet endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/fleet/tasks", response_model=List[TaskDefinition])
def get_fleet_tasks() -> List[TaskDefinition]:
    return [TaskDefinition(**t) for t in fleet_env.TASK_DEFS]


@app.post("/fleet/reset", response_model=ResetResponse)
def reset_fleet(payload: dict = {}):
    task_id = payload.get("task_id", "fleet_precision")
    obs = fleet_env.reset(task_id)
    return ResetResponse(observation=obs)


@app.post("/fleet/step", response_model=StepResponse)
def step_fleet(payload: dict = {}):
    action = payload.get("action", payload)
    result = fleet_env.step(action)
    clamped_score = _clamp_score(result["reward"]["score"])
    return StepResponse(
        observation=result["observation"],
        reward=RewardPayload(
            score=clamped_score,
            feedback=result["reward"]["feedback"],
        ),
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/fleet/state", response_model=StateResponse)
def get_fleet_state():
    s = fleet_env.state()
    return StateResponse(**s)


# ═══════════════════════════════════════════════════════════
# AGENTIC KERNEL — Live Performance Benchmarking
# Proves the core Agentic Kernel thesis: pure-math physics
# evaluation runs in microseconds, not the 100ms+ of Docker/VM
# sandboxes. These endpoints let the frontend visualize this
# in real time.
# ═══════════════════════════════════════════════════════════

@app.get("/kernel/benchmark")
def kernel_benchmark():
    """
    Run a live benchmark of the Agentic Kernel physics engine.
    Evaluates N trajectories and returns precise timing data
    to prove microsecond-latency evaluation.
    """
    import time
    import random as _rnd
    from environment.physics_model import (
        compute_training_cost,
        compute_hardware_safety,
        score_precision_layer,
    )

    NUM_TRAJECTORIES = 1000
    precisions = ["FP32", "BF16", "FP16", "FP8"]
    layers = ["embedding", "attention", "ffn", "layernorm", "output"]
    layer_dist = {
        "embedding": 0.05, "attention": 0.35, "ffn": 0.45,
        "layernorm": 0.05, "output": 0.10,
    }

    # --- Phase 1: Benchmark compute_training_cost ---
    start = time.perf_counter_ns()
    for _ in range(NUM_TRAJECTORIES):
        strategy = {l: _rnd.choice(precisions) for l in layers}
        params = _rnd.choice([3_000_000_000, 7_000_000_000, 13_000_000_000, 70_000_000_000])
        compute_training_cost(
            total_params=params,
            precision_strategy=strategy,
            layer_distribution=layer_dist,
        )
    cost_ns = time.perf_counter_ns() - start

    # --- Phase 2: Benchmark hardware safety checks ---
    start = time.perf_counter_ns()
    for _ in range(NUM_TRAJECTORIES):
        strategy = {l: _rnd.choice(precisions) for l in layers}
        params = _rnd.choice([3_000_000_000, 7_000_000_000, 13_000_000_000])
        compute_hardware_safety(
            total_params=params,
            precision_strategy=strategy,
            layer_distribution=layer_dist,
            num_gpus=_rnd.choice([1, 2, 4, 8]),
            gpu_memory_gb=80.0,
        )
    safety_ns = time.perf_counter_ns() - start

    # --- Phase 3: Benchmark per-layer scoring ---
    start = time.perf_counter_ns()
    for _ in range(NUM_TRAJECTORIES):
        for l in layers:
            score_precision_layer(l, _rnd.choice(precisions))
    scoring_ns = time.perf_counter_ns() - start

    # --- Phase 4: Full environment reset+step cycle ---
    from environment.fleet_env import FleetEnvironment
    bench_env = FleetEnvironment()
    start = time.perf_counter_ns()
    cycle_count = 200
    for _ in range(cycle_count):
        bench_env.reset("fleet_precision")
        bench_env.step({
            "precision_strategy": {
                "embedding": "FP32", "attention": "BF16", "ffn": "FP8",
                "layernorm": "BF16", "output": "FP32"
            },
            "reasoning": "benchmark"
        })
    env_ns = time.perf_counter_ns() - start

    total_ns = cost_ns + safety_ns + scoring_ns
    per_trajectory_us = (total_ns / NUM_TRAJECTORIES) / 1000

    # Docker/VM comparison (published industry data)
    docker_cold_start_ms = 150  # Typical Docker container cold start
    microvm_cold_start_ms = 125  # Firecracker MicroVM

    return {
        "kernel_version": "1.0.0",
        "trajectories_evaluated": NUM_TRAJECTORIES,
        "timing": {
            "cost_model_total_us": round(cost_ns / 1000, 1),
            "cost_model_per_eval_us": round(cost_ns / NUM_TRAJECTORIES / 1000, 2),
            "safety_check_total_us": round(safety_ns / 1000, 1),
            "safety_check_per_eval_us": round(safety_ns / NUM_TRAJECTORIES / 1000, 2),
            "layer_scoring_total_us": round(scoring_ns / 1000, 1),
            "layer_scoring_per_eval_us": round(scoring_ns / NUM_TRAJECTORIES / 1000, 2),
            "full_env_cycle_per_eval_us": round(env_ns / cycle_count / 1000, 2),
            "total_per_trajectory_us": round(per_trajectory_us, 2),
        },
        "throughput": {
            "trajectories_per_second": round(1_000_000 / per_trajectory_us) if per_trajectory_us > 0 else 999999,
            "speedup_vs_docker": round(docker_cold_start_ms * 1000 / per_trajectory_us, 0) if per_trajectory_us > 0 else 999999,
            "speedup_vs_microvm": round(microvm_cold_start_ms * 1000 / per_trajectory_us, 0) if per_trajectory_us > 0 else 999999,
        },
        "comparison": {
            "agentic_kernel_us": round(per_trajectory_us, 2),
            "docker_container_ms": docker_cold_start_ms,
            "firecracker_microvm_ms": microvm_cold_start_ms,
            "sandbox_ratio_solved": per_trajectory_us < 1000,
        },
    }


@app.get("/kernel/profile")
def kernel_profile():
    """
    Profile each kernel module (physics sub-system) individually.
    Returns per-module timing and a breakdown of what the kernel evaluates.
    """
    import time
    from environment.physics_model import (
        BYTES_PER_PARAM, STABILITY_SCORE, THROUGHPUT_MULTIPLIER,
        ACCURACY_PENALTY, POWER_DRAW_MULTIPLIER,
        compute_training_cost, compute_hardware_safety, score_precision_layer,
        compute_network_topology,
    )

    layers = ["embedding", "attention", "ffn", "layernorm", "output"]
    layer_dist = {
        "embedding": 0.05, "attention": 0.35, "ffn": 0.45,
        "layernorm": 0.05, "output": 0.10,
    }
    strategy = {
        "embedding": "FP32", "attention": "BF16", "ffn": "FP8",
        "layernorm": "BF16", "output": "FP32",
    }
    params = 7_000_000_000  # 7B model

    # Profile each function
    modules = {}

    start = time.perf_counter_ns()
    result = compute_training_cost(params, strategy, layer_dist)
    modules["precision_kernel"] = {
        "latency_us": round((time.perf_counter_ns() - start) / 1000, 2),
        "description": "VRAM, throughput, cost, stability evaluation",
        "outputs": {
            "memory_gb": result["memory_gb"],
            "speedup_vs_fp32": result["speedup_vs_fp32"],
            "cost_usd": result["cost_usd"],
            "estimated_stable": result["estimated_stable"],
            "accuracy_retention": result["accuracy_retention"],
        }
    }

    start = time.perf_counter_ns()
    hw = compute_hardware_safety(params, strategy, layer_dist, num_gpus=4, gpu_memory_gb=80.0)
    modules["safety_kernel"] = {
        "latency_us": round((time.perf_counter_ns() - start) / 1000, 2),
        "description": "Thermal, power, memory utilization checks",
        "outputs": {
            "memory_per_gpu_gb": hw["memory_per_gpu_gb"],
            "memory_utilization_pct": hw["memory_utilization_pct"],
            "thermal_risk": hw["thermal_risk"],
            "overall_safe": hw["overall_safe"],
        }
    }

    start = time.perf_counter_ns()
    layer_results = {}
    for l in layers:
        score, feedback = score_precision_layer(l, strategy.get(l, "FP32"))
        layer_results[l] = {"score": score, "precision": strategy.get(l, "FP32")}
    modules["scoring_kernel"] = {
        "latency_us": round((time.perf_counter_ns() - start) / 1000, 2),
        "description": "Per-layer precision quality scoring",
        "outputs": layer_results,
    }

    start = time.perf_counter_ns()
    net = compute_network_topology("PCIe_Gen4", tensor_parallel_size=8, pipeline_parallel_size=1)
    modules["network_kernel"] = {
        "latency_us": round((time.perf_counter_ns() - start) / 1000, 2),
        "description": "Multi-GPU topology bottleneck analysis",
        "outputs": {
            "topology": net["topology"],
            "effective_throughput_pct": net["effective_throughput_pct"],
            "bottleneck_risk": net["bottleneck_risk"],
        }
    }

    total_latency = sum(m["latency_us"] for m in modules.values())

    return {
        "profile_model": "LLaMA-3-7B",
        "profile_strategy": strategy,
        "modules": modules,
        "total_kernel_latency_us": round(total_latency, 2),
        "physics_constants_loaded": {
            "precision_formats": len(BYTES_PER_PARAM),
            "layer_types": len(STABILITY_SCORE),
            "stability_rules": sum(len(v) for v in STABILITY_SCORE.values()),
            "throughput_rules": sum(len(v) for v in THROUGHPUT_MULTIPLIER.values()),
            "source": "NVIDIA Transformer Engine + Micikevicius et al. 2018/2022",
        }
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

