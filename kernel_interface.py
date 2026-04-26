"""
Agentic Kernel — Clean Interface for Physics-Based RL Evaluation.

Wraps physics grading into a composable, pluggable abstraction with
batch evaluation and built-in performance instrumentation.
"""
from __future__ import annotations
import time, statistics
from typing import Any, Dict, List, Optional

from environment.physics_model import (
    BYTES_PER_PARAM, STABILITY_SCORE, THROUGHPUT_MULTIPLIER, ACCURACY_PENALTY,
    POWER_DRAW_MULTIPLIER, H100_COST_PER_HOUR_USD, GPU_TDP_WATTS,
    MEMORY_UTILIZATION_DANGER, MEMORY_UTILIZATION_WARNING, POWER_UTILIZATION_DANGER,
    compute_training_cost, compute_hardware_safety, score_precision_layer,
    compute_network_topology,
)

def _clamp(s: float) -> float:
    try: v = float(s)
    except: v = 0.01
    return max(0.01, min(0.99, v))


class KernelModule:
    """Base class for pluggable kernel evaluation modules."""
    name: str = "base"
    def evaluate(self, state: dict, action: dict) -> dict:
        raise NotImplementedError


class PrecisionKernel(KernelModule):
    """Mixed-precision evaluation using NVIDIA TE physics constants."""
    name = "precision"

    def evaluate(self, state: dict, action: dict) -> dict:
        model = state.get("model", {})
        params = model.get("total_params", 7_000_000_000)
        layer_dist = model.get("layer_distribution", {
            "embedding": 0.15, "attention": 0.25, "ffn": 0.40,
            "layernorm": 0.002, "output": 0.198,
        })
        cluster = state.get("cluster", {})
        total_mem = cluster.get("total_memory_gb", 640)
        n_models = state.get("num_models", 1)
        strategy = action.get("precision_strategy", {})

        vals = list(strategy.values())
        if vals and all(v == "FP8" for v in vals):
            return {"score": 0.01, "feedback": "IRD: All-FP8 crash", "details": {"ird_violation": True}}
        if vals and all(v == "FP32" for v in vals):
            return {"score": 0.01, "feedback": "IRD: All-FP32 waste", "details": {"ird_violation": True}}
        if not strategy:
            return {"score": 0.01, "feedback": "IRD: Empty strategy", "details": {"ird_violation": True}}

        metrics = compute_training_cost(total_params=params, precision_strategy=strategy, layer_distribution=layer_dist)
        n_gpus = max(1, cluster.get("total_gpus", 8) // max(n_models, 1))
        hw = compute_hardware_safety(total_params=params, precision_strategy=strategy, layer_distribution=layer_dist, num_gpus=n_gpus, gpu_memory_gb=cluster.get("gpu_memory_gb", 80.0))

        layer_scores = {}
        for lt in layer_dist:
            p = strategy.get(lt, "FP32")
            ls, lf = score_precision_layer(lt, p)
            layer_scores[lt] = {"precision": p, "score": ls}

        avg = sum(v["score"] for v in layer_scores.values()) / max(len(layer_scores), 1)
        fair = total_mem / max(n_models, 1)
        mp = 0.15 if metrics["memory_gb"] > fair * 1.5 else (0.05 if metrics["memory_gb"] > fair else 0.0)
        sp = 0.0 if metrics["estimated_stable"] else 0.3
        hp = 0.0 if hw["overall_safe"] else 0.1
        score = _clamp(avg - mp - sp - hp)

        return {
            "score": score,
            "feedback": f"precision={score:.3f} mem={metrics['memory_gb']}GB speed={metrics['speedup_vs_fp32']}x",
            "details": {"layer_scores": layer_scores, "metrics": metrics, "hw_safety": hw, "ird_violation": False},
        }


class ThermalKernel(KernelModule):
    """Thermal-aware workload placement evaluation (NVIDIA H100 SXM5 specs)."""
    name = "thermal"
    AMBIENT_C = 25.0
    THERMAL_R = 0.083  # C/W
    THROTTLE_C = 83.0
    SHUTDOWN_C = 92.0
    SUSTAINED = 1.15

    def evaluate(self, state: dict, action: dict) -> dict:
        model = state.get("model", {})
        params = model.get("total_params", 7_000_000_000)
        layer_dist = model.get("layer_distribution", {"embedding": 0.15, "attention": 0.25, "ffn": 0.40, "layernorm": 0.002, "output": 0.198})
        cluster = state.get("cluster", {})
        n_models = state.get("num_models", 1)
        n_gpus = max(1, cluster.get("total_gpus", 8) // max(n_models, 1))
        strategy = action.get("precision_strategy", {})
        if not strategy:
            return {"score": 0.01, "feedback": "No strategy for thermal eval", "details": {}}

        wp, tw = 0.0, 0.0
        for lt, frac in layer_dist.items():
            p = strategy.get(lt, "FP32")
            wp += POWER_DRAW_MULTIPLIER.get(p, 1.0) * frac
            tw += frac
        avg_pr = wp / max(tw, 0.001)
        watts = avg_pr * GPU_TDP_WATTS * self.SUSTAINED
        temp = self.AMBIENT_C + watts * self.THERMAL_R

        mem_bytes = sum(params * layer_dist.get(lt, 0) * BYTES_PER_PARAM.get(strategy.get(lt, "FP32"), 4) for lt in layer_dist)
        mem_util = (mem_bytes / 1e9) / max(n_gpus, 1) / cluster.get("gpu_memory_gb", 80.0)
        temp *= 1.0 + mem_util * 0.05

        if temp >= self.SHUTDOWN_C:
            risk, score = "SHUTDOWN", 0.01
        elif temp >= self.THROTTLE_C:
            risk = "THROTTLING"
            score = max(0.10, 0.50 - (temp - self.THROTTLE_C) / (self.SHUTDOWN_C - self.THROTTLE_C) * 0.40)
        elif temp >= self.THROTTLE_C - 10:
            risk, score = "HIGH", 0.65
        elif temp >= 60:
            risk, score = "MODERATE", 0.80
        else:
            risk, score = "OPTIMAL", 0.95

        return {
            "score": _clamp(score),
            "feedback": f"thermal={temp:.1f}C {watts:.0f}W risk={risk}",
            "details": {"temp_c": round(temp, 1), "watts": round(watts, 0), "power_ratio": round(avg_pr, 3), "risk": risk, "mem_util": round(mem_util, 3)},
        }


class NetworkKernel(KernelModule):
    """Network topology and communication overhead evaluation."""
    name = "network"
    def evaluate(self, state: dict, action: dict) -> dict:
        topo = action.get("topology", "NVLink_v4")
        tp = action.get("tensor_parallel_size", 1)
        pp = action.get("pipeline_parallel_size", 1)
        hidden = state.get("model", {}).get("hidden_size", 4096)
        batch = action.get("batch_size", 4)
        r = compute_network_topology(topo, tp, pp, hidden, batch)
        score = _clamp(r["effective_throughput_pct"] / 100.0)
        return {"score": score, "feedback": f"network={r['effective_throughput_pct']:.1f}% risk={r['bottleneck_risk']}", "details": r}


class AgenticKernel:
    """The Agentic Kernel: zero-overhead, pure-math RL evaluation engine.

    Composes pluggable KernelModules (precision, thermal, network) to produce
    deterministic reward signals at 10,000+ evals/sec on a single CPU core.
    """
    DEFAULT_WEIGHTS = {"precision": 0.60, "thermal": 0.25, "network": 0.15}

    def __init__(self, modules=None, weights=None):
        if modules is None:
            modules = [PrecisionKernel(), ThermalKernel(), NetworkKernel()]
        self.modules = {m.name: m for m in modules}
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        active = {k: v for k, v in self.weights.items() if k in self.modules}
        t = sum(active.values())
        if t > 0:
            self.weights = {k: v / t for k, v in active.items()}
        self._eval_count = 0
        self._total_ns = 0
        self._latencies = []

    def evaluate(self, state: dict, action: dict) -> dict:
        start = time.perf_counter_ns()
        breakdown = {}
        ws = 0.0
        for name, mod in self.modules.items():
            r = mod.evaluate(state, action)
            breakdown[name] = r
            ws += r["score"] * self.weights.get(name, 0.0)
        score = _clamp(ws)
        ns = time.perf_counter_ns() - start
        us = ns / 1000.0
        self._eval_count += 1
        self._total_ns += ns
        self._latencies.append(us)
        parts = [f"{n}:{r['score']:.3f}" for n, r in breakdown.items()]
        return {"score": score, "feedback": f"kernel={score:.3f} [{', '.join(parts)}] ({us:.1f}us)", "breakdown": breakdown, "latency_us": round(us, 1)}

    def batch_evaluate(self, rollouts: list) -> list:
        return [self.evaluate(r["state"], r["action"]) for r in rollouts]

    def get_throughput_stats(self) -> dict:
        if not self._eval_count:
            return {"total_evaluations": 0, "throughput_evals_per_sec": 0.0, "mean_latency_us": 0.0, "median_latency_us": 0.0, "p99_latency_us": 0.0}
        s = sorted(self._latencies)
        total_s = self._total_ns / 1e9
        p99i = int(len(s) * 0.99)
        return {
            "total_evaluations": self._eval_count,
            "total_time_ms": round(self._total_ns / 1e6, 2),
            "throughput_evals_per_sec": round(self._eval_count / max(total_s, 1e-9), 0),
            "mean_latency_us": round(statistics.mean(self._latencies), 1),
            "median_latency_us": round(statistics.median(self._latencies), 1),
            "p99_latency_us": round(s[min(p99i, len(s)-1)], 1),
            "min_latency_us": round(s[0], 1),
            "max_latency_us": round(s[-1], 1),
        }

    def get_module_names(self) -> list:
        return list(self.modules.keys())

    def get_weights(self) -> dict:
        return dict(self.weights)

    def reset_stats(self):
        self._eval_count = 0
        self._total_ns = 0
        self._latencies = []
