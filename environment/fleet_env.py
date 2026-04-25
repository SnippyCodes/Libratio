"""
Fleet Environment: Multi-agent GPU cluster management for mixed precision training.

This is the CORE environment for Theme 1 (Multi-Agent Interactions) + Fleet AI sub-theme.

Architecture:
  - Multiple Training Agents: each manages precision for one model
  - Oversight Agent: monitors all training runs for instability
  - Resource Allocator: distributes GPU memory across competing models

The environment runs episodes where agents take turns making decisions,
sharing a cluster-wide GPU memory pool. All physics computations are
delegated to physics_model.py (reused from the single-agent environment).

Tasks:
  1. fleet_precision    — Agents assign precision to their models under shared memory
  2. fleet_oversight    — Oversight agent monitors all runs, detects crashes
  3. fleet_resource     — Agents negotiate GPU allocation from a shared pool
  4. fleet_recovery     — Handle a mid-training crash: diagnose + reallocate
"""
from typing import Dict, Any, Optional, List
import random
import copy

from environment.physics_model import (
    BYTES_PER_PARAM,
    STABILITY_SCORE,
    THROUGHPUT_MULTIPLIER,
    ACCURACY_PENALTY,
    H100_COST_PER_HOUR_USD,
    compute_training_cost,
    score_precision_layer,
    compute_hardware_safety,
)


def clamp_score(score: float) -> float:
    """Clamp score to the open interval (0.01, 0.99) for OpenEnv compliance."""
    try:
        val = float(score)
    except (ValueError, TypeError):
        val = 0.01
    return float(max(0.01, min(0.99, val)))


class FleetEnvironment:
    """Multi-agent fleet environment managing multiple models on a shared GPU cluster."""

    TASK_DEFS = [
        {
            "id": "fleet_precision",
            "task_id": "fleet_precision",
            "description": "Multiple agents assign precision formats to their models under a shared GPU memory budget",
            "difficulty": "medium",
            "max_steps": 5,
        },
        {
            "id": "fleet_oversight",
            "task_id": "fleet_oversight",
            "description": "Oversight agent monitors loss trajectories from multiple simultaneous training runs to detect crashes",
            "difficulty": "hard",
            "max_steps": 5,
        },
        {
            "id": "fleet_resource",
            "task_id": "fleet_resource",
            "description": "Agents negotiate GPU allocation from a shared cluster pool, balancing priorities and efficiency",
            "difficulty": "hard",
            "max_steps": 5,
        },
        {
            "id": "fleet_recovery",
            "task_id": "fleet_recovery",
            "description": "Diagnose a mid-training crash, reallocate cluster resources, and recover fleet operations",
            "difficulty": "medium-hard",
            "max_steps": 3,
        },
    ]

    def __init__(self):
        self.current_task: Optional[str] = None
        self.scenario: Optional[Dict] = None
        self.step_number: int = 0
        self.is_done: bool = True
        self.history: list = []
        self.task_state: Dict[str, Any] = {}

    def reset(self, task_id: str, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment for a new fleet episode."""
        from scenarios.fleet_scenarios import FLEET_SCENARIOS, FLEET_OVERSIGHT_SCENARIOS

        self.current_task = task_id
        self.step_number = 0
        self.is_done = False
        self.history = []
        self.task_state = {}

        if task_id == "fleet_precision":
            return self._reset_fleet_precision(FLEET_SCENARIOS, scenario_id)
        elif task_id == "fleet_oversight":
            return self._reset_fleet_oversight(FLEET_OVERSIGHT_SCENARIOS, scenario_id)
        elif task_id == "fleet_resource":
            return self._reset_fleet_resource(FLEET_SCENARIOS, scenario_id)
        elif task_id == "fleet_recovery":
            return self._reset_fleet_recovery(FLEET_SCENARIOS, FLEET_OVERSIGHT_SCENARIOS, scenario_id)
        else:
            raise ValueError(f"Unknown fleet task: {task_id}")

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process an action from the current agent."""
        if self.is_done:
            return {
                "observation": None,
                "reward": {"score": 0.01, "feedback": "Episode finished. Call /reset."},
                "done": True,
                "info": {},
            }

        if self.current_task == "fleet_precision":
            return self._step_fleet_precision(action)
        elif self.current_task == "fleet_oversight":
            return self._step_fleet_oversight(action)
        elif self.current_task == "fleet_resource":
            return self._step_fleet_resource(action)
        elif self.current_task == "fleet_recovery":
            return self._step_fleet_recovery(action)

        return {
            "observation": None,
            "reward": {"score": 0.01, "feedback": "No active task."},
            "done": True,
            "info": {},
        }

    def state(self) -> Dict[str, Any]:
        """Get current environment state."""
        return {
            "task_id": self.current_task,
            "current_task": self.current_task,
            "current_scenario_id": self.scenario.get("scenario_id") if self.scenario else None,
            "step_number": self.step_number,
            "is_done": self.is_done,
            "history": self.history,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 5: Inverse Reward Design (Anti-Reward Hacking)
    #   "The reward function is a PROXY for what we actually want. Penalize
    #    behaviors that optimize the proxy but violate the intent."
    #   — Seminar Note: Inverse Reward Design tells agent the rules are
    #     guidelines, preventing exploits like shutting down a rack to save power.
    # ═══════════════════════════════════════════════════════════════════════════

    def _detect_reward_hacking(self, action: dict, task_id: str) -> tuple:
        """Detect if agent is exploiting reward function loopholes.

        Returns: (is_hacking: bool, penalty: float, reason: str)
        """
        if task_id == "fleet_precision":
            strategy = action.get("precision_strategy", {})
            values = list(strategy.values())
            # Hack: all FP8 everywhere (crashes training but minimizes memory)
            if values and all(v == "FP8" for v in values):
                return True, -0.5, "IRD: All-FP8 is degenerate — training would crash on embedding+output"
            # Hack: all FP32 everywhere (no optimization, wastes fleet resources)
            if values and all(v == "FP32" for v in values):
                return True, -0.2, "IRD: All-FP32 wastes fleet resources — zero optimization"
            # Hack: no strategy provided
            if not strategy:
                return True, -0.4, "IRD: Empty precision strategy is invalid"

        elif task_id == "fleet_resource":
            allocations = action.get("allocations", {})
            if not allocations:
                return True, -0.4, "IRD: Empty allocations are invalid"
            # Hack: starve any model completely
            for mid, alloc in allocations.items():
                if isinstance(alloc, dict) and alloc.get("gpus", 0) == 0:
                    return True, -0.3, f"IRD: Starving {mid} with 0 GPUs is not acceptable fleet management"

        elif task_id == "fleet_oversight":
            action_type = action.get("action_type", "")
            # Hack: flag on very first step with no analysis (trigger-happy)
            if action_type == "flag_instability" and self.step_number == 0:
                analysis = action.get("analysis", "") or ""
                root_cause = action.get("root_cause", "") or ""
                if len(analysis) < 15 and len(root_cause) < 10:
                    return True, -0.2, "IRD: Flagging without evidence on first window suggests gaming"

        elif task_id == "fleet_recovery":
            reasoning = action.get("reasoning", "") or ""
            # Hack: submit near-empty responses to collect base score
            if len(reasoning) < 5 and not action.get("root_cause") and not action.get("new_precision_strategy"):
                return True, -0.3, "IRD: Near-empty recovery action exploits base score"

        return False, 0.0, ""

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK 1: Fleet Precision Assignment
    #   Multiple agents assign precision to their models under shared memory.
    #   Each step, ONE agent configures ONE model's full precision strategy.
    #   Shared memory pool creates inter-agent dependency.
    # ═══════════════════════════════════════════════════════════════════════════

    def _reset_fleet_precision(self, scenarios: Dict, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Initialize fleet precision assignment episode."""
        scenario_key = scenario_id if scenario_id in scenarios else random.choice(list(scenarios.keys()))
        self.scenario = copy.deepcopy(scenarios[scenario_key])
        cluster = self.scenario["cluster"]
        models = self.scenario["models"]

        self.task_state = {
            "models": models,
            "cluster": cluster,
            "current_agent_idx": 0,       # Which agent's turn
            "agent_configs": {},           # model_id -> precision_strategy
            "agent_results": {},           # model_id -> computed metrics
            "fleet_memory_used_gb": 0.0,   # Running total across all models
            "fleet_total_cost_usd": 0.0,
        }

        return self._fleet_precision_observation()

    def _fleet_precision_observation(self) -> Dict[str, Any]:
        """Build observation for the current agent in fleet precision task."""
        s = self.task_state
        agent_idx = s["current_agent_idx"]
        model = s["models"][agent_idx]
        cluster = s["cluster"]

        # Show what other agents have done so far
        other_agents_summary = {}
        for mid, config in s["agent_configs"].items():
            result = s["agent_results"].get(mid, {})
            other_agents_summary[mid] = {
                "precision_strategy": config,
                "memory_used_gb": result.get("memory_gb", 0),
                "status": "configured",
            }

        remaining_memory = cluster["total_memory_gb"] - s["fleet_memory_used_gb"]

        return {
            "task_id": "fleet_precision",
            "scenario_id": self.scenario["scenario_id"],
            "cluster": {
                "total_gpus": cluster["total_gpus"],
                "total_memory_gb": cluster["total_memory_gb"],
                "memory_used_gb": round(s["fleet_memory_used_gb"], 2),
                "memory_remaining_gb": round(remaining_memory, 2),
            },
            "your_model": {
                "model_id": model["model_id"],
                "name": model["name"],
                "total_params": model["total_params"],
                "layer_distribution": model["layer_distribution"],
                "priority": model["priority"],
                "priority_reason": model["priority_reason"],
            },
            "other_agents": other_agents_summary,
            "agents_remaining": len(s["models"]) - agent_idx - 1,
            "available_precisions": ["FP32", "BF16", "FP16", "FP8"],
            "step_number": self.step_number + 1,
            "total_steps": len(s["models"]),
        }

    def _step_fleet_precision(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process one agent's precision strategy for their model.

        Integrates:
        - Phase 2: Breadcrumb rewards (per-layer feedback signals)
        - Phase 3: Hardware safety dashboard (thermal/power/memory checks)
        - Phase 4: Difference Rewards (counterfactual fleet contribution)
        - Phase 5: Inverse Reward Design (degenerate action detection)
        """
        s = self.task_state
        agent_idx = s["current_agent_idx"]
        model = s["models"][agent_idx]
        model_id = model["model_id"]
        strategy = action.get("precision_strategy", {})
        cluster = s["cluster"]

        self.step_number += 1

        # ── Phase 5: IRD check — catch reward hacking before scoring ──
        is_hacking, ird_penalty, ird_reason = self._detect_reward_hacking(action, "fleet_precision")
        if is_hacking:
            self.history.append({
                "step": self.step_number, "agent": model_id,
                "strategy": strategy, "score": 0.01, "ird_blocked": True,
            })
            s["current_agent_idx"] += 1
            done = s["current_agent_idx"] >= len(s["models"])
            if done:
                self.is_done = True
            return {
                "observation": None if done else self._fleet_precision_observation(),
                "reward": {"score": 0.01, "feedback": ird_reason},
                "done": done,
                "info": {"ird_violation": True},
            }

        # Compute metrics using physics model
        metrics = compute_training_cost(
            total_params=model["total_params"],
            precision_strategy=strategy,
            layer_distribution=model["layer_distribution"],
        )

        # ── Phase 3: Hardware safety dashboard ──
        hw_safety = compute_hardware_safety(
            total_params=model["total_params"],
            precision_strategy=strategy,
            layer_distribution=model["layer_distribution"],
            num_gpus=max(1, cluster["total_gpus"] // len(s["models"])),
            gpu_memory_gb=cluster.get("gpu_memory_gb", 80.0),
        )

        s["agent_configs"][model_id] = strategy
        s["agent_results"][model_id] = metrics
        s["fleet_memory_used_gb"] += metrics["memory_gb"]

        # ── Phase 2: Breadcrumb rewards — per-layer feedback signals ──
        layer_scores = []
        breadcrumbs = []
        for layer_type in model["layer_distribution"]:
            precision = strategy.get(layer_type, "FP32")
            lscore, lfeedback = score_precision_layer(layer_type, precision)
            layer_scores.append(lscore)
            # Breadcrumb: tell agent exactly which layers are good/bad
            if lscore >= 0.95:
                crumb = f"[OK] {layer_type}={precision}"
            elif lscore >= 0.55:
                crumb = f"[~~] {layer_type}={precision}"
            else:
                crumb = f"[!!] {layer_type}={precision}"
            breadcrumbs.append(crumb)

        avg_layer_score = sum(layer_scores) / max(len(layer_scores), 1)

        # Memory penalty: if this agent's model exceeds fair share
        fair_share_gb = cluster["total_memory_gb"] / len(s["models"])
        if metrics["memory_gb"] > fair_share_gb * 1.5:
            memory_penalty = 0.15
            memory_feedback = f"WARNING: Using {metrics['memory_gb']}GB — exceeds fair share ({fair_share_gb:.0f}GB) by >50%!"
        elif metrics["memory_gb"] > fair_share_gb:
            memory_penalty = 0.05
            memory_feedback = f"Note: Using {metrics['memory_gb']}GB (fair share: {fair_share_gb:.0f}GB)"
        else:
            memory_penalty = 0.0
            memory_feedback = f"Good: Using {metrics['memory_gb']}GB — within fair share ({fair_share_gb:.0f}GB)"

        # Stability check
        stability_penalty = 0.0
        if not metrics["estimated_stable"]:
            stability_penalty = 0.3
            memory_feedback += " | CRITICAL: Unstable configuration detected!"

        # ── Phase 3: Hardware safety penalty ──
        hw_penalty = 0.0
        if not hw_safety["overall_safe"]:
            hw_penalty = 0.1
            memory_feedback += f" | HW DANGER: {hw_safety['thermal_risk']} thermal risk, mem={hw_safety['memory_utilization_pct']}%"

        base_agent_score = avg_layer_score - memory_penalty - stability_penalty - hw_penalty

        # ── Phase 4: Difference Rewards (Seminar Note 4) ──
        # How much did THIS agent's choice affect the fleet?
        # D_i = R(fleet) - R(fleet without agent_i)
        fleet_memory = s["fleet_memory_used_gb"]
        fleet_utilization = fleet_memory / cluster["total_memory_gb"]
        fleet_reward_with = 1.0 - abs(fleet_utilization - 0.75)  # ideal = 75% util

        # Fleet reward WITHOUT this agent (counterfactual)
        fleet_memory_without = fleet_memory - metrics["memory_gb"]
        fleet_util_without = fleet_memory_without / cluster["total_memory_gb"]
        fleet_reward_without = 1.0 - abs(fleet_util_without - 0.75)

        # Difference reward: positive = you helped, negative = you hurt
        difference = fleet_reward_with - fleet_reward_without

        # Blend: 70% individual score + 30% difference reward
        blended_score = 0.7 * base_agent_score + 0.3 * max(0, difference + 0.5)
        agent_score = clamp_score(blended_score)

        # ── Phase 2: Breadcrumb feedback string ──
        breadcrumb_str = " | Layers: " + ", ".join(breadcrumbs)
        feedback = (
            f"Agent {model_id} ({model['name']}): score={agent_score:.3f}, "
            f"mem={metrics['memory_gb']}GB, speedup={metrics['speedup_vs_fp32']}x, "
            f"cost=${metrics['cost_usd']:,.0f}. {memory_feedback}{breadcrumb_str}"
        )

        self.history.append({
            "step": self.step_number,
            "agent": model_id,
            "strategy": strategy,
            "score": agent_score,
            "metrics": metrics,
            "hw_safety": hw_safety,
        })

        # Advance to next agent
        s["current_agent_idx"] += 1
        done = s["current_agent_idx"] >= len(s["models"])

        if done:
            self.is_done = True
            # ── Fleet-wide scoring ──
            fleet_total_cost = sum(r["cost_usd"] for r in s["agent_results"].values())
            fleet_fp32_cost = sum(r["fp32_baseline_cost_usd"] for r in s["agent_results"].values())
            fleet_savings = fleet_fp32_cost - fleet_total_cost
            fleet_all_stable = all(r["estimated_stable"] for r in s["agent_results"].values())

            # Exceeded total cluster memory?
            memory_overflow = s["fleet_memory_used_gb"] > cluster["total_memory_gb"]

            fleet_bonus = ""
            if memory_overflow:
                agent_score = clamp_score(agent_score - 0.3)
                fleet_bonus = f" | FLEET FAIL: Total memory {s['fleet_memory_used_gb']:.0f}GB exceeds cluster capacity {cluster['total_memory_gb']}GB!"
            elif fleet_all_stable:
                agent_score = clamp_score(agent_score + 0.05)
                fleet_bonus = f" | FLEET SUCCESS: All models stable. Total savings: ${fleet_savings:,.0f} ({fleet_fp32_cost:.0f} → {fleet_total_cost:.0f})"
            else:
                fleet_bonus = f" | FLEET WARNING: Some models have unstable configs. Savings: ${fleet_savings:,.0f}"

            feedback += fleet_bonus

            info = {
                "fleet_summary": {
                    "total_memory_used_gb": round(s["fleet_memory_used_gb"], 2),
                    "cluster_capacity_gb": cluster["total_memory_gb"],
                    "memory_overflow": memory_overflow,
                    "total_cost_usd": round(fleet_total_cost, 0),
                    "fp32_baseline_cost_usd": round(fleet_fp32_cost, 0),
                    "fleet_savings_usd": round(fleet_savings, 0),
                    "all_stable": fleet_all_stable,
                    "per_model": {
                        mid: {
                            "strategy": s["agent_configs"][mid],
                            "memory_gb": res["memory_gb"],
                            "cost_usd": round(res["cost_usd"], 0),
                            "stable": res["estimated_stable"],
                        }
                        for mid, res in s["agent_results"].items()
                    },
                },
            }
        else:
            info = {"memory_used_so_far_gb": round(s["fleet_memory_used_gb"], 2)}

        next_obs = None if done else self._fleet_precision_observation()
        return {
            "observation": next_obs,
            "reward": {"score": clamp_score(agent_score), "feedback": feedback},
            "done": done,
            "info": info,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK 2: Fleet Oversight Monitoring (Fleet AI Sub-theme)
    #   Oversight agent watches loss trajectories from ALL training runs.
    #   Must identify WHICH model is crashing, WHEN, and WHY.
    # ═══════════════════════════════════════════════════════════════════════════

    def _reset_fleet_oversight(self, scenarios: Dict, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Initialize fleet oversight monitoring episode."""
        scenario_key = scenario_id if scenario_id in scenarios else random.choice(list(scenarios.keys()))
        self.scenario = copy.deepcopy(scenarios[scenario_key])

        window_size = self.scenario["window_size"]
        step_count = self.scenario["step_count"]

        self.task_state = {
            "window_idx": 0,
            "total_windows": step_count // window_size,
            "flagged": False,
            "window_size": window_size,
        }

        return self._fleet_oversight_observation()

    def _fleet_oversight_observation(self) -> Dict[str, Any]:
        """Build observation showing loss windows from ALL models simultaneously."""
        s = self.task_state
        ws = s["window_size"]
        start = s["window_idx"] * ws
        end = start + ws

        # Build per-model loss windows
        model_windows = {}
        for model_id, traj_data in self.scenario["trajectories"].items():
            window = traj_data["loss"][start:end]
            # Replace NaN with None for JSON serialization
            clean_window = [None if (v != v) else round(v, 4) for v in window]
            model_windows[model_id] = {
                "loss_window": clean_window,
                "precision_config": traj_data["precision_config"],
            }

        return {
            "task_id": "fleet_oversight",
            "scenario_id": self.scenario["scenario_id"],
            "description": self.scenario["description"],
            "num_models": len(self.scenario["trajectories"]),
            "model_trajectories": model_windows,
            "total_training_steps": self.scenario["step_count"],
            "steps_revealed_so_far": end,
            "window_index": s["window_idx"],
            "windows_remaining": s["total_windows"] - s["window_idx"] - 1,
        }

    def _step_fleet_oversight(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process oversight agent's monitoring decision.

        Integrates:
        - Phase 2: Breadcrumb progress signal (confidence ramps with safe scanning)
        - Phase 5: IRD check for trigger-happy flagging
        """
        s = self.task_state
        gt = self.scenario["ground_truth"]
        action_type = action.get("action_type", "continue_monitoring")
        self.step_number += 1

        # ── Phase 5: IRD check ──
        is_hacking, ird_penalty, ird_reason = self._detect_reward_hacking(action, "fleet_oversight")
        if is_hacking:
            score = clamp_score(0.05)
            feedback = ird_reason
            self.is_done = True
            self.history.append({"step": self.step_number, "action": action_type, "score": score, "ird_blocked": True})
            return {
                "observation": None,
                "reward": {"score": score, "feedback": feedback},
                "done": True,
                "info": {"ird_violation": True},
            }

        if action_type == "flag_instability":
            s["flagged"] = True
            self.is_done = True
            score, feedback = self._grade_fleet_flag(action, gt, s["window_idx"])

        elif action_type == "continue_monitoring":
            # BUG FIX: Check current window BEFORE incrementing
            # (Composer found: was evaluating next window instead of observed one)
            current_idx = s["window_idx"]
            ws = s["window_size"]
            has_issues = False
            for model_id, traj_data in self.scenario["trajectories"].items():
                window = traj_data["loss"][current_idx*ws:(current_idx+1)*ws]
                if any(v != v for v in window if isinstance(v, float)):
                    has_issues = True
                    break

            s["window_idx"] += 1
            if s["window_idx"] >= s["total_windows"]:
                # End of all windows
                self.is_done = True
                if gt["crashing_model"] is not None:
                    score = 0.10
                    feedback = (
                        "MISSED: Fleet had a crashing model but oversight never flagged it. "
                        f"Model {gt['crashing_model']} crashed at step {gt['failure_step']}."
                    )
                else:
                    score = 0.95
                    feedback = "CORRECT: All models healthy — no instability to flag. Good monitoring."
            else:
                # ── Phase 2: Breadcrumb progress signal ──
                # Score gradually increases as more of the timeline is safely scanned
                progress = s["window_idx"] / max(s["total_windows"], 1)
                if has_issues:
                    score = 0.30
                    feedback = f"Window {s['window_idx']}: Issues visible in fleet — consider flagging. [scanned {progress*100:.0f}%]"
                else:
                    # Breadcrumb: reward increases with safe progress
                    score = 0.40 + (0.15 * progress)
                    feedback = f"Window {s['window_idx']}: Fleet looks stable. [scanned {progress*100:.0f}%, confidence={score:.2f}]"
                self.is_done = False

        score = clamp_score(score)
        self.history.append({
            "step": self.step_number,
            "action": action_type,
            "score": score,
        })

        next_obs = None if self.is_done else self._fleet_oversight_observation()
        return {
            "observation": next_obs,
            "reward": {"score": score, "feedback": feedback},
            "done": self.is_done,
            "info": {},
        }

    def _grade_fleet_flag(self, action: Dict, gt: Dict, window_idx: int) -> tuple:
        """Grade an oversight agent's instability flag."""
        # False alarm — nothing was wrong
        if gt["crashing_model"] is None:
            return 0.10, "FALSE ALARM: Fleet is healthy. Flagging incorrectly wastes resources and causes unnecessary rollbacks."

        score = 0.30
        parts = ["Correctly detected fleet instability (+0.30)."]

        # Did agent identify the RIGHT model?
        flagged_model = action.get("flagged_model", "")
        if flagged_model and flagged_model.lower() == gt["crashing_model"].lower():
            score += 0.25
            parts.append(f"Identified correct model: {flagged_model} (+0.25).")
        elif flagged_model:
            score += 0.05
            parts.append(f"Identified wrong model: {flagged_model} (actual: {gt['crashing_model']}) (+0.05).")

        # Did agent pinpoint the failure step?
        flagged_step = action.get("flagged_step", -1)
        actual_step = gt["failure_step"]
        if flagged_step is not None and actual_step > 0:
            diff = abs(flagged_step - actual_step)
            if diff <= 3:
                score += 0.20
                parts.append(f"Pinpointed failure step within 3 of actual (+0.20).")
            elif diff <= 10:
                score += 0.12
                parts.append(f"Close to failure step (off by {diff}) (+0.12).")
            else:
                score += 0.03
                parts.append(f"Failure step off by {diff} (+0.03).")

        # Root cause analysis
        root_cause = (action.get("root_cause") or "").lower()
        matches = sum(1 for kw in gt["cause_keywords"] if kw.lower() in root_cause)
        if gt["cause_keywords"]:
            cause_score = 0.25 * (matches / len(gt["cause_keywords"]))
            score += round(cause_score, 2)
            parts.append(f"Root cause: matched {matches}/{len(gt['cause_keywords'])} keywords (+{round(cause_score, 2)}).")

        return clamp_score(round(score, 3)), " ".join(parts)

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK 3: Fleet Resource Negotiation
    #   Agents bid for GPU allocation from a shared cluster.
    #   Must balance priorities, efficiency, and fairness.
    # ═══════════════════════════════════════════════════════════════════════════

    def _reset_fleet_resource(self, scenarios: Dict, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Initialize fleet resource negotiation episode."""
        scenario_key = scenario_id if scenario_id in scenarios else random.choice(list(scenarios.keys()))
        self.scenario = copy.deepcopy(scenarios[scenario_key])
        cluster = self.scenario["cluster"]
        models = self.scenario["models"]

        self.task_state = {
            "iteration": 0,
            "max_iterations": self.scenario.get("max_steps_per_task", 5),
            "models": models,
            "cluster": cluster,
            "best_score": 0.0,
            "best_allocation": None,
            "prev_feedback": None,
            "prev_result": None,
        }

        return self._fleet_resource_observation()

    def _fleet_resource_observation(self) -> Dict[str, Any]:
        """Build observation for resource negotiation."""
        s = self.task_state
        cluster = s["cluster"]

        model_specs = []
        for m in s["models"]:
            model_specs.append({
                "model_id": m["model_id"],
                "name": m["name"],
                "total_params": m["total_params"],
                "priority": m["priority"],
                "priority_reason": m["priority_reason"],
            })

        return {
            "task_id": "fleet_resource",
            "scenario_id": self.scenario["scenario_id"],
            "cluster": {
                "total_gpus": cluster["total_gpus"],
                "gpu_memory_gb": cluster["gpu_memory_gb"],
                "total_memory_gb": cluster["total_memory_gb"],
                "cost_per_gpu_hour_usd": cluster["cost_per_gpu_hour_usd"],
            },
            "models": model_specs,
            "iterations_remaining": s["max_iterations"] - s["iteration"],
            "best_score_so_far": round(s["best_score"], 3),
            "previous_result": s["prev_result"],
            "previous_feedback": s["prev_feedback"],
        }

    def _step_fleet_resource(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process a resource allocation proposal.

        Integrates:
        - Phase 2: Breadcrumb delta (shows improvement from previous iteration)
        - Phase 4: Difference Rewards (counterfactual vs naive equal-split)
        - Phase 5: IRD check for degenerate allocations
        """
        s = self.task_state
        cluster = s["cluster"]
        allocations = action.get("allocations", {})
        self.step_number += 1
        s["iteration"] += 1

        # ── Phase 5: IRD check ──
        is_hacking, ird_penalty, ird_reason = self._detect_reward_hacking(action, "fleet_resource")
        if is_hacking:
            s["prev_feedback"] = ird_reason
            done = s["iteration"] >= s["max_iterations"]
            if done:
                self.is_done = True
            self.history.append({"step": self.step_number, "allocations": allocations, "score": 0.01, "ird_blocked": True})
            return {
                "observation": None if done else self._fleet_resource_observation(),
                "reward": {"score": 0.01, "feedback": ird_reason},
                "done": done,
                "info": {"ird_violation": True},
            }

        # allocations = { model_id: { "gpus": N, "precision_strategy": {...} } }
        total_gpus_assigned = 0
        fleet_cost = 0.0
        fleet_fp32_cost = 0.0
        per_model_results = {}
        all_valid = True
        issues = []

        for model in s["models"]:
            mid = model["model_id"]
            alloc = allocations.get(mid, {})
            gpus = alloc.get("gpus", 0)
            strategy = alloc.get("precision_strategy", {
                "embedding": "FP32", "attention": "BF16",
                "ffn": "FP8", "layernorm": "BF16", "output": "FP32",
            })
            total_gpus_assigned += gpus

            if gpus <= 0:
                all_valid = False
                issues.append(f"{mid}: assigned 0 GPUs")
                per_model_results[mid] = {"gpus": 0, "valid": False}
                continue

            # Compute metrics
            metrics = compute_training_cost(
                total_params=model["total_params"],
                precision_strategy=strategy,
                layer_distribution=model["layer_distribution"],
            )

            # Memory check: model must fit in allocated GPU memory
            available_mem = gpus * cluster["gpu_memory_gb"]
            mem_ok = metrics["memory_gb"] <= available_mem

            if not mem_ok:
                all_valid = False
                issues.append(f"{mid}: needs {metrics['memory_gb']}GB but only has {available_mem}GB ({gpus} GPUs)")

            per_model_results[mid] = {
                "gpus": gpus,
                "memory_gb": metrics["memory_gb"],
                "available_memory_gb": available_mem,
                "memory_fits": mem_ok,
                "cost_usd": round(metrics["cost_usd"], 0),
                "training_days": metrics["training_days"],
                "speedup": metrics["speedup_vs_fp32"],
                "stable": metrics["estimated_stable"],
                "valid": mem_ok and metrics["estimated_stable"],
            }
            fleet_cost += metrics["cost_usd"]
            fleet_fp32_cost += metrics["fp32_baseline_cost_usd"]

        # Check total GPU constraint
        if total_gpus_assigned > cluster["total_gpus"]:
            all_valid = False
            issues.append(f"Total GPUs assigned ({total_gpus_assigned}) exceeds cluster ({cluster['total_gpus']})")
        elif total_gpus_assigned < cluster["total_gpus"]:
            issues.append(f"Underutilization: only {total_gpus_assigned}/{cluster['total_gpus']} GPUs assigned")

        # ── Scoring ──
        if not all_valid:
            score = 0.01
            feedback = f"INVALID allocation: {'; '.join(issues)}. {s['max_iterations'] - s['iteration']} iterations remaining."
        else:
            # Score components:
            # 1. Cost efficiency (how much saved vs FP32)
            savings_pct = (fleet_fp32_cost - fleet_cost) / max(fleet_fp32_cost, 1)
            cost_score = min(1.0, savings_pct / 0.5)  # Normalize: 50% savings = perfect

            # 2. GPU utilization
            util = total_gpus_assigned / cluster["total_gpus"]
            util_score = min(1.0, util)

            # 3. Priority alignment (higher priority models get more resources)
            priority_score = self._score_priority_alignment(s["models"], allocations, cluster)

            # 4. All stable
            all_stable = all(r.get("stable", False) for r in per_model_results.values())
            stability_bonus = 0.1 if all_stable else 0.0

            base_score = 0.3 * cost_score + 0.25 * util_score + 0.25 * priority_score + 0.1 + stability_bonus

            # ── Phase 4: Difference Rewards ──
            # Counterfactual: what if we used naive equal-split allocation?
            naive_gpus = cluster["total_gpus"] // max(len(s["models"]), 1)
            naive_cost = 0.0
            for model in s["models"]:
                naive_metrics = compute_training_cost(
                    total_params=model["total_params"],
                    precision_strategy={"embedding": "FP32", "attention": "BF16",
                                        "ffn": "BF16", "layernorm": "BF16", "output": "FP32"},
                    layer_distribution=model["layer_distribution"],
                )
                naive_cost += naive_metrics["cost_usd"]
            # Agent's improvement over naive baseline
            diff_reward = (naive_cost - fleet_cost) / max(naive_cost, 1)
            # Blend: 80% base + 20% difference reward
            score = clamp_score(0.8 * base_score + 0.2 * max(0, diff_reward + 0.3))

            fleet_savings = fleet_fp32_cost - fleet_cost

            # ── Phase 2: Breadcrumb delta ──
            prev_score = s.get("prev_score", 0.0)
            delta = score - prev_score
            delta_str = f" | {'↑' if delta > 0 else '↓'}{abs(delta):.3f} from last" if prev_score > 0 else ""

            feedback = (
                f"Valid allocation! Fleet cost: ${fleet_cost:,.0f} (savings: ${fleet_savings:,.0f}). "
                f"GPU utilization: {util*100:.0f}%. Priority alignment: {priority_score:.2f}. "
                f"All stable: {all_stable}. Score: {score:.3f}. "
                f"{s['max_iterations'] - s['iteration']} iterations remaining.{delta_str}"
            )
            s["prev_score"] = score

        if score > s["best_score"]:
            s["best_score"] = score
            s["best_allocation"] = allocations

        s["prev_result"] = per_model_results
        s["prev_feedback"] = feedback

        done = s["iteration"] >= s["max_iterations"]
        if done:
            self.is_done = True

        score = clamp_score(score)
        self.history.append({
            "step": self.step_number,
            "allocations": allocations,
            "score": score,
        })

        next_obs = None if done else self._fleet_resource_observation()
        return {
            "observation": next_obs,
            "reward": {"score": score, "feedback": feedback},
            "done": done,
            "info": {"per_model": per_model_results},
        }

    def _score_priority_alignment(self, models: List, allocations: Dict, cluster: Dict) -> float:
        """Score how well GPU allocation aligns with model priorities."""
        total_priority = sum(m["priority"] for m in models)
        if total_priority == 0:
            return 0.5

        score = 0.0
        for m in models:
            mid = m["model_id"]
            alloc = allocations.get(mid, {})
            gpus = alloc.get("gpus", 0)
            expected_share = m["priority"] / total_priority
            actual_share = gpus / max(cluster["total_gpus"], 1)
            # Reward for being close to priority-proportional allocation
            alignment = 1.0 - min(1.0, abs(expected_share - actual_share) / max(expected_share, 0.01))
            score += alignment * expected_share

        return min(1.0, score)

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK 4: Fleet Recovery
    #   One model crashes mid-training. Agent must diagnose and reallocate.
    # ═══════════════════════════════════════════════════════════════════════════

    def _reset_fleet_recovery(self, fleet_scenarios: Dict, oversight_scenarios: Dict, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Initialize fleet recovery episode — a crash has occurred."""
        # Pick a scenario with a crash
        crash_scenarios = {k: v for k, v in oversight_scenarios.items()
                          if v["ground_truth"]["crashing_model"] is not None}
        scenario_key = None
        if scenario_id is not None and scenario_id.startswith("recovery_"):
            key = scenario_id[9:]
            if key in crash_scenarios:
                scenario_key = key
        if scenario_key is None:
            scenario_key = random.choice(list(crash_scenarios.keys()))
        oversight = copy.deepcopy(crash_scenarios[scenario_key])

        # Get the fleet config
        fleet_key = oversight.get("fleet_id", "medium_fleet")
        fleet = copy.deepcopy(fleet_scenarios.get(fleet_key, fleet_scenarios["medium_fleet"]))

        crashed_model_id = oversight["ground_truth"]["crashing_model"]

        self.scenario = {
            "scenario_id": f"recovery_{scenario_key}",
            "fleet": fleet,
            "oversight": oversight,
            "crashed_model_id": crashed_model_id,
        }

        # Show the crash context
        crashed_traj = oversight["trajectories"][crashed_model_id]
        crash_step = oversight["ground_truth"]["failure_step"]
        crash_window_start = max(0, crash_step - 10)
        crash_window = crashed_traj["loss"][crash_window_start:crash_step + 5]
        clean_crash_window = [None if (v != v) else round(v, 4) for v in crash_window]

        self.task_state = {
            "iteration": 0,
            "max_iterations": 3,
            "phase": "diagnose",  # diagnose → reallocate → verify
            "crashed_model_id": crashed_model_id,
            "crash_step": crash_step,
            "crash_window": clean_crash_window,
            "crashed_config": crashed_traj["precision_config"],
            "fleet": fleet,
            "best_score": 0.0,
            "prev_feedback": None,
        }

        return self._fleet_recovery_observation()

    def _fleet_recovery_observation(self) -> Dict[str, Any]:
        """Build observation for fleet recovery."""
        s = self.task_state
        fleet = s["fleet"]
        cluster = fleet["cluster"]

        # Build model info
        model_summaries = []
        for m in fleet["models"]:
            status = "crashed" if m["model_id"] == s["crashed_model_id"] else "running"
            model_summaries.append({
                "model_id": m["model_id"],
                "name": m["name"],
                "total_params": m["total_params"],
                "priority": m["priority"],
                "status": status,
            })

        obs = {
            "task_id": "fleet_recovery",
            "scenario_id": self.scenario["scenario_id"],
            "phase": s["phase"],
            "cluster": {
                "total_gpus": cluster["total_gpus"],
                "total_memory_gb": cluster["total_memory_gb"],
            },
            "models": model_summaries,
            "crashed_model": {
                "model_id": s["crashed_model_id"],
                "crash_step": s["crash_step"],
                "loss_around_crash": s["crash_window"],
                "precision_config": s["crashed_config"],
            },
            "iterations_remaining": s["max_iterations"] - s["iteration"],
            "previous_feedback": s["prev_feedback"],
        }

        return obs

    def _step_fleet_recovery(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process a fleet recovery action.

        Integrates:
        - Phase 4: Difference Rewards in reallocation (vs generic safe config)
        - Phase 5: IRD check for empty/gaming recovery actions
        """
        s = self.task_state
        self.step_number += 1
        s["iteration"] += 1
        gt = self.scenario["oversight"]["ground_truth"]

        # ── Phase 5: IRD check ──
        is_hacking, ird_penalty, ird_reason = self._detect_reward_hacking(action, "fleet_recovery")
        if is_hacking:
            s["prev_feedback"] = ird_reason
            done = s["iteration"] >= s["max_iterations"]
            if done:
                self.is_done = True
            self.history.append({"step": self.step_number, "phase": s["phase"], "score": 0.01, "ird_blocked": True})
            return {
                "observation": None if done else self._fleet_recovery_observation(),
                "reward": {"score": 0.01, "feedback": ird_reason},
                "done": done,
                "info": {"ird_violation": True},
            }

        phase = s["phase"]

        if phase == "diagnose":
            # Agent must identify root cause
            score, feedback = self._grade_recovery_diagnosis(action, gt)
            s["phase"] = "reallocate"

        elif phase == "reallocate":
            # Agent proposes new precision for crashed model + resource reallocation
            score, feedback = self._grade_recovery_reallocation(action, s)
            s["phase"] = "verify"

        else:  # verify
            # Agent confirms the recovery plan
            score, feedback = self._grade_recovery_verification(action, s)
            s["phase"] = "done"

        if score > s["best_score"]:
            s["best_score"] = score
        s["prev_feedback"] = feedback

        done = s["iteration"] >= s["max_iterations"] or s["phase"] == "done"
        if done:
            self.is_done = True

        score = clamp_score(score)
        self.history.append({"step": self.step_number, "phase": phase, "score": score})

        next_obs = None if done else self._fleet_recovery_observation()
        return {
            "observation": next_obs,
            "reward": {"score": score, "feedback": feedback},
            "done": done,
            "info": {},
        }

    def _grade_recovery_diagnosis(self, action: Dict, gt: Dict) -> tuple:
        """Grade diagnosis of the crash."""
        score = 0.20
        parts = ["Diagnosis submitted (+0.20 base)."]

        # Root cause
        root_cause = (action.get("root_cause") or "").lower()
        matches = sum(1 for kw in gt["cause_keywords"] if kw.lower() in root_cause)
        if gt["cause_keywords"]:
            cause_score = 0.50 * (matches / len(gt["cause_keywords"]))
            score += round(cause_score, 2)
            parts.append(f"Root cause: matched {matches}/{len(gt['cause_keywords'])} keywords (+{round(cause_score, 2)}).")

        # Identified correct model
        diagnosed_model = (action.get("diagnosed_model") or "").lower()
        if diagnosed_model == gt["crashing_model"].lower():
            score += 0.30
            parts.append("Correctly identified crashing model (+0.30).")

        return clamp_score(round(score, 3)), " ".join(parts)

    def _grade_recovery_reallocation(self, action: Dict, state: Dict) -> tuple:
        """Grade the reallocation plan for the crashed model.

        Phase 4: Includes Difference Reward comparing agent's fix to a generic
        safe fallback (BF16 everywhere). Agent gets bonus for being more
        efficient than the naive fix while maintaining stability.
        """
        new_strategy = action.get("new_precision_strategy", {})
        fleet = state["fleet"]
        crashed_id = state["crashed_model_id"]

        # Find the crashed model
        crashed_model = None
        for m in fleet["models"]:
            if m["model_id"] == crashed_id:
                crashed_model = m
                break

        if not crashed_model:
            return 0.01, "Could not find crashed model in fleet."

        # Compute new metrics
        metrics = compute_training_cost(
            total_params=crashed_model["total_params"],
            precision_strategy=new_strategy,
            layer_distribution=crashed_model["layer_distribution"],
        )

        score = 0.20
        parts = ["Reallocation plan submitted (+0.20 base)."]

        if metrics["estimated_stable"]:
            score += 0.40
            parts.append(f"New config is STABLE (+0.40). Mem={metrics['memory_gb']}GB, Speed={metrics['speedup_vs_fp32']}x.")
        else:
            parts.append("WARNING: New config is still UNSTABLE.")

        # Did they fix the problematic layer?
        old_config = state["crashed_config"]
        changes = sum(1 for k in new_strategy if new_strategy.get(k) != old_config.get(k))
        if changes > 0:
            adapt_bonus = min(0.20, changes * 0.07)
            score += adapt_bonus
            parts.append(f"Changed {changes} layer(s) from crashed config (+{adapt_bonus:.2f}).")

        # Efficiency bonus
        if metrics["speedup_vs_fp32"] > 1.5:
            score += 0.10
            parts.append(f"Efficient recovery: {metrics['speedup_vs_fp32']}x speedup (+0.10).")

        # ── Phase 4: Difference Reward vs generic safe config ──
        generic_safe = {"embedding": "FP32", "attention": "BF16",
                        "ffn": "BF16", "layernorm": "BF16", "output": "FP32"}
        generic_metrics = compute_training_cost(
            total_params=crashed_model["total_params"],
            precision_strategy=generic_safe,
            layer_distribution=crashed_model["layer_distribution"],
        )
        if metrics["estimated_stable"] and metrics["speedup_vs_fp32"] > generic_metrics["speedup_vs_fp32"]:
            diff_bonus = min(0.10, (metrics["speedup_vs_fp32"] - generic_metrics["speedup_vs_fp32"]) * 0.15)
            score += round(diff_bonus, 3)
            parts.append(f"Better than generic fix: {metrics['speedup_vs_fp32']}x vs {generic_metrics['speedup_vs_fp32']}x (+{diff_bonus:.3f}).")

        return clamp_score(round(score, 3)), " ".join(parts)

    def _grade_recovery_verification(self, action: Dict, state: Dict) -> tuple:
        """Grade the verification step."""
        reasoning = action.get("reasoning", "")
        confidence = action.get("confidence", "")

        score = 0.50
        parts = ["Verification submitted (+0.50 base)."]

        # Reward thorough reasoning
        if len(reasoning) > 50:
            score += 0.20
            parts.append("Detailed recovery reasoning provided (+0.20).")

        if "stable" in reasoning.lower() or "safe" in reasoning.lower():
            score += 0.10
            parts.append("Stability considerations mentioned (+0.10).")

        if "monitor" in reasoning.lower() or "watch" in reasoning.lower():
            score += 0.10
            parts.append("Post-recovery monitoring plan mentioned (+0.10).")

        return clamp_score(round(score, 3)), " ".join(parts)
