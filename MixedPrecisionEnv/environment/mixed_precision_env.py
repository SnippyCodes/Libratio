"""
Core Environment: Multi-step state machine for Mixed Precision Training.
Each task is an episode with multiple steps and per-step rewards.
Physics model grounded in real benchmarks — see environment/physics_model.py
"""
from typing import Dict, Any, Optional
from scenarios.loader import ScenarioLoader
from scenarios.task2_scenarios import WINDOW_SIZE
from environment.physics_model import (
    BYTES_PER_PARAM,
    STABILITY_SCORE,
    ACCURACY_PENALTY,
    THROUGHPUT_MULTIPLIER,
    H100_COST_PER_HOUR_USD,
    GPU_HOURS_PER_BILLION_PARAMS,
    compute_training_cost,
    score_precision_layer,
)


class MixedPrecisionEnvironment:
    TASK_DEFS = [
        {"task_id": "precision_assignment", "description": "Assign precision formats to model layers one at a time", "difficulty": "easy", "max_steps": 5},
        {"task_id": "instability_detection", "description": "Progressively analyze training loss to detect precision-induced instability", "difficulty": "medium", "max_steps": 5},
        {"task_id": "multi_objective_optimization", "description": "Iteratively optimize precision strategy under memory, time, and accuracy constraints", "difficulty": "hard", "max_steps": 5},
    ]

    def __init__(self):
        self.current_task: Optional[str] = None
        self.scenario: Optional[Dict] = None
        self.step_number: int = 0
        self.is_done: bool = True
        self.history: list = []
        self.task_state: Dict[str, Any] = {}

    def reset(self, task_id: str) -> Dict[str, Any]:
        self.current_task = task_id
        self.step_number = 0
        self.is_done = False
        self.history = []
        self.task_state = {}

        if task_id == "precision_assignment":
            return self._reset_task1()
        elif task_id == "instability_detection":
            return self._reset_task2()
        elif task_id == "multi_objective_optimization":
            return self._reset_task3()
        else:
            raise ValueError(f"Unknown task: {task_id}")

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_done:
            return {"observation": None, "reward": {"score": 0.0, "feedback": "Episode finished. Call /reset."}, "done": True, "info": {}}
        if self.current_task == "precision_assignment":
            return self._step_task1(action)
        elif self.current_task == "instability_detection":
            return self._step_task2(action)
        elif self.current_task == "multi_objective_optimization":
            return self._step_task3(action)
        return {"observation": None, "reward": {"score": 0.0, "feedback": "No active task."}, "done": True, "info": {}}

    def state(self) -> Dict[str, Any]:
        return {
            "current_task": self.current_task,
            "current_scenario_id": self.scenario["scenario_id"] if self.scenario else None,
            "step_number": self.step_number,
            "is_done": self.is_done,
            "history": self.history,
        }

    # ── Task 1: Layer-by-layer precision assignment ──
    def _reset_task1(self) -> Dict[str, Any]:
        self.scenario = ScenarioLoader.get_task1()
        layers = self.scenario["layers"]
        self.task_state = {"layers": layers, "assigned": {}, "memory_used_gb": 0.0, "layer_idx": 0}
        return self._task1_observation()

    def _task1_observation(self) -> Dict[str, Any]:
        s = self.task_state
        layer = s["layers"][s["layer_idx"]]
        return {
            "task_id": "precision_assignment",
            "scenario_id": self.scenario["scenario_id"],
            "model_name": self.scenario["model_name"],
            "total_layers": len(s["layers"]),
            "current_layer_index": s["layer_idx"],
            "current_layer": layer,
            "assigned_so_far": s["assigned"],
            "available_precisions": ["FP32", "BF16", "FP16", "FP8"],
            "memory_budget_gb": self.scenario["memory_budget_gb"],
            "memory_used_gb": round(s["memory_used_gb"], 3),
            "speed_target_speedup": self.scenario["speed_target_speedup"],
        }

    def _step_task1(self, action: Dict[str, Any]) -> Dict[str, Any]:
        s = self.task_state
        layer = s["layers"][s["layer_idx"]]
        precision = action.get("precision", "FP32")
        layer_type = layer["layer_type"]

        # Use real empirical scoring from physics_model.py
        score, feedback = score_precision_layer(layer_type, precision)

        mem_added = (layer["num_params"] * BYTES_PER_PARAM[precision]) / 1e9
        s["memory_used_gb"] += mem_added
        s["assigned"][layer["name"]] = precision
        s["throughput_sum"] = s.get("throughput_sum", 0.0) + THROUGHPUT_MULTIPLIER.get(layer_type, {}).get(precision, 1.0)
        s["layer_idx"] += 1
        self.step_number += 1
        self.history.append({"step": self.step_number, "layer": layer["name"], "precision": precision, "score": score})

        done = s["layer_idx"] >= len(s["layers"])
        info = {"memory_used_gb": round(s["memory_used_gb"], 3)}

        if done:
            self.is_done = True
            total_params = sum(l["num_params"] for l in s["layers"])
            avg_speedup = s.get("throughput_sum", 1.0) / len(s["layers"])
            fp32_cost = (total_params / 1e9) * GPU_HOURS_PER_BILLION_PARAMS * H100_COST_PER_HOUR_USD
            actual_cost = fp32_cost / avg_speedup
            savings = fp32_cost - actual_cost
            info["cost_analysis"] = {
                "fp32_baseline_cost_usd": round(fp32_cost, 0),
                "actual_cost_usd": round(actual_cost, 0),
                "savings_usd": round(savings, 0),
                "savings_pct": round((savings / fp32_cost) * 100, 1) if fp32_cost > 0 else 0,
                "avg_speedup_vs_fp32": round(avg_speedup, 2),
            }
            if s["memory_used_gb"] > self.scenario["memory_budget_gb"]:
                score = max(0.0, score - 0.3)
                feedback += f" | WARNING: Total memory {s['memory_used_gb']:.1f}GB exceeds budget {self.scenario['memory_budget_gb']}GB!"
            feedback += f" | Est. savings vs FP32: ${savings:,.0f} ({(savings/fp32_cost*100):.0f}%)"

        next_obs = None if done else self._task1_observation()
        return {"observation": next_obs, "reward": {"score": score, "feedback": feedback}, "done": done, "info": info}

    # ── Task 2: Progressive instability detection ──
    def _reset_task2(self) -> Dict[str, Any]:
        self.scenario = ScenarioLoader.get_task2()
        traj = self.scenario["training_loss_trajectory"]
        self.task_state = {"trajectory": traj, "window_idx": 0, "total_windows": len(traj) // WINDOW_SIZE, "flagged": False}
        return self._task2_observation()

    def _task2_observation(self) -> Dict[str, Any]:
        s = self.task_state
        start = s["window_idx"] * WINDOW_SIZE
        end = start + WINDOW_SIZE
        window = self.scenario["training_loss_trajectory"][start:end]
        clean_window = [None if (v != v) else round(v, 4) for v in window]
        return {
            "task_id": "instability_detection",
            "scenario_id": self.scenario["scenario_id"],
            "precision_config": self.scenario["precision_config"],
            "total_training_steps": self.scenario["step_count"],
            "steps_revealed_so_far": end,
            "loss_trajectory_window": clean_window,
            "window_index": s["window_idx"],
            "windows_remaining": s["total_windows"] - s["window_idx"] - 1,
        }

    def _step_task2(self, action: Dict[str, Any]) -> Dict[str, Any]:
        s = self.task_state
        gt = self.scenario["ground_truth"]
        action_type = action.get("action_type", "continue_monitoring")
        self.step_number += 1

        if action_type == "flag_instability":
            s["flagged"] = True
            self.is_done = True
            score, feedback = self._grade_instability_flag(action, gt, s["window_idx"])
        elif action_type == "continue_monitoring":
            s["window_idx"] += 1
            if s["window_idx"] >= s["total_windows"]:
                self.is_done = True
                if gt["is_unstable"]:
                    score = 0.1
                    feedback = "Missed the instability entirely. The training crashed but you never flagged it."
                else:
                    score = 1.0
                    feedback = "Correctly monitored entire trajectory. No instability found — correct!"
            else:
                current_window = self.scenario["training_loss_trajectory"][s["window_idx"]*WINDOW_SIZE:(s["window_idx"]+1)*WINDOW_SIZE]
                has_nan = any(v != v for v in current_window if isinstance(v, float))
                analysis_text = action.get("analysis", "")
                if has_nan:
                    score = 0.3
                    feedback = f"Window {s['window_idx']}: NaN detected in this window. Consider flagging."
                else:
                    score = 0.6
                    feedback = f"Window {s['window_idx']}: Monitoring continues. Loss looks {'stable' if not gt['is_unstable'] else 'watch carefully'}."
                self.is_done = False

        self.history.append({"step": self.step_number, "action": action_type, "score": score})
        next_obs = None if self.is_done else self._task2_observation()
        return {"observation": next_obs, "reward": {"score": score, "feedback": feedback}, "done": self.is_done, "info": {}}

    def _grade_instability_flag(self, action, gt, window_idx):
        if not gt["is_unstable"]:
            return 0.1, "False alarm! The trajectory was actually stable. Flagging incorrectly is costly."

        score = 0.4
        parts = ["Correctly detected instability (+0.4)."]

        flagged_step = action.get("flagged_step", -1)
        actual = gt["failure_step"]
        if flagged_step is not None and actual > 0:
            diff = abs(flagged_step - actual)
            if diff <= 3:
                score += 0.3
                parts.append(f"Pinpointed failure step within 3 of actual (+0.3).")
            elif diff <= 10:
                score += 0.2
                parts.append(f"Close to failure step (off by {diff}) (+0.2).")
            else:
                score += 0.05
                parts.append(f"Failure step off by {diff} (+0.05).")

        root_cause = (action.get("root_cause") or "").lower()
        matches = sum(1 for kw in gt["cause_keywords"] if kw.lower() in root_cause)
        if gt["cause_keywords"]:
            cause_score = 0.3 * (matches / len(gt["cause_keywords"]))
            score += round(cause_score, 2)
            parts.append(f"Root cause: matched {matches}/{len(gt['cause_keywords'])} keywords (+{round(cause_score, 2)}).")

        return min(1.0, round(score, 3)), " ".join(parts)

    # ── Task 3: Iterative multi-objective optimization ──
    def _reset_task3(self) -> Dict[str, Any]:
        self.scenario = ScenarioLoader.get_task3()
        self.task_state = {"iterations_left": self.scenario["max_iterations"], "best_score": 0.0, "best_result": None, "prev_feedback": None}
        return self._task3_observation()

    def _task3_observation(self) -> Dict[str, Any]:
        s = self.task_state
        return {
            "task_id": "multi_objective_optimization",
            "scenario_id": self.scenario["scenario_id"],
            "constraints": self.scenario["constraints"],
            "model_total_params": self.scenario["total_params"],
            "layer_types": list(self.scenario["layer_distribution"].keys()),
            "iterations_remaining": s["iterations_left"],
            "best_score_so_far": round(s["best_score"], 3),
            "previous_result": s["best_result"],
            "previous_feedback": s["prev_feedback"],
        }

    def _step_task3(self, action: Dict[str, Any]) -> Dict[str, Any]:
        s = self.task_state
        strategy = action.get("precision_strategy", {})
        sc = self.scenario
        dist = sc["layer_distribution"]
        constraints = sc["constraints"]
        self.step_number += 1
        s["iterations_left"] -= 1

        # Use real empirical physics model for all computations
        metrics = compute_training_cost(
            total_params=sc["total_params"],
            precision_strategy=strategy,
            layer_distribution=dist,
        )

        result = {
            "memory_gb": metrics["memory_gb"],
            "time_days": metrics["training_days"],
            "accuracy": metrics["accuracy_retention"],
            "cost_usd": metrics["cost_usd"],
            "savings_usd": metrics["savings_usd"],
            "savings_pct": metrics["savings_pct"],
            "speedup_vs_fp32": metrics["speedup_vs_fp32"],
        }

        mem_ok = result["memory_gb"] <= constraints["memory_budget_gb"]
        time_ok = result["time_days"] <= constraints["time_budget_days"]
        acc_ok = result["accuracy"] >= constraints["accuracy_threshold"]

        if not (mem_ok and time_ok and acc_ok):
            fails = []
            if not mem_ok: fails.append(f"Memory {result['memory_gb']}GB > {constraints['memory_budget_gb']}GB")
            if not time_ok: fails.append(f"Time {result['time_days']}d > {constraints['time_budget_days']}d")
            if not acc_ok: fails.append(f"Accuracy {result['accuracy']} < {constraints['accuracy_threshold']}")
            score = 0.0
            feedback = (
                f"Constraints violated: {'; '.join(fails)}. "
                f"[Empirical model: NVIDIA-TE + Meta-LLaMA3] "
                f"{s['iterations_left']} iterations remaining."
            )
        else:
            mem_eff = 1.0 - (result["memory_gb"] / constraints["memory_budget_gb"])
            time_eff = 1.0 - (result["time_days"] / constraints["time_budget_days"])
            acc_margin = (result["accuracy"] - constraints["accuracy_threshold"]) / max(0.001, 1.0 - constraints["accuracy_threshold"])
            score = round(0.5 + 0.2 * mem_eff + 0.15 * time_eff + 0.15 * min(1.0, acc_margin), 3)
            score = min(1.0, max(0.0, score))
            feedback = (
                f"Valid! Mem={result['memory_gb']}GB, Time={result['time_days']}d, "
                f"Acc={result['accuracy']}, Speedup={result['speedup_vs_fp32']}x vs FP32. "
                f"Est. savings: ${result['savings_usd']:,} ({result['savings_pct']}%) "
                f"[Source: NVIDIA-TE + Meta-LLaMA3 + H100 cloud pricing]. "
                f"Score={score}. {s['iterations_left']} iterations left."
            )

        if score > s["best_score"]:
            s["best_score"] = score
        s["best_result"] = result
        s["prev_feedback"] = feedback

        done = s["iterations_left"] <= 0
        if done:
            self.is_done = True
        self.history.append({"step": self.step_number, "strategy": strategy, "result": result, "score": score})

        next_obs = None if done else self._task3_observation()
        return {"observation": next_obs, "reward": {"score": score, "feedback": feedback}, "done": done, "info": {"computed_metrics": result}}
