"""
Fleet Training Script: Training agents for Libratio Fleet environment.

Run locally or on HuggingFace Spaces with GPU compute credits.
Can use server mode (HTTP) or direct import mode.

Usage:
    # Direct import (no server needed)
    python train_fleet.py --mode direct --task fleet_precision --episodes 100

    # Server mode (requires running server)
    python train_fleet.py --mode server --task fleet_precision --episodes 100
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Any


def _init_direct_mode():
    """Initialize direct import mode (no server needed)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from environment.fleet_env import FleetEnvironment
    _env = FleetEnvironment()

    def reset_fn(task_id: str) -> Dict:
        return _env.reset(task_id)

    def step_fn(action: Dict) -> Dict[str, Any]:
        result = _env.step(action)
        return {
            "observation": result["observation"],
            "reward": result["reward"]["score"],
            "feedback": result["reward"]["feedback"],
            "done": result["done"],
        }
    return reset_fn, step_fn


def _init_server_mode():
    """Initialize server mode (HTTP calls to running server)."""
    import httpx
    env_url = os.getenv("ENV_URL", "http://localhost:7860")
    fleet_url = f"{env_url}/fleet"

    def reset_fn(task_id: str) -> Dict:
        res = httpx.post(f"{fleet_url}/reset", json={"task_id": task_id}, timeout=30.0)
        return res.json()["observation"]

    def step_fn(action: Dict) -> Dict[str, Any]:
        res = httpx.post(f"{fleet_url}/step", json={"action": action}, timeout=30.0)
        data = res.json()
        return {
            "observation": data.get("observation"),
            "reward": data["reward"]["score"],
            "feedback": data["reward"]["feedback"],
            "done": data["done"],
        }
    return reset_fn, step_fn


# Default to direct mode (overridden by main() based on CLI args)
reset_environment, step_environment = _init_direct_mode()


def collect_episode(task_id: str, policy_fn, max_steps: int = 10) -> Dict:
    """Collect one episode using the current policy."""
    obs = reset_environment(task_id)
    observations = []
    actions = []
    rewards = []

    done = False
    steps = 0

    while not done and steps < max_steps:
        action = policy_fn(obs)
        step_result = step_environment(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(step_result["reward"])

        obs = step_result["observation"]
        done = step_result["done"]
        steps += 1

    avg_step_reward = sum(rewards) / len(rewards) if rewards else 0.0
    final_step_reward = rewards[-1] if rewards else 0.0
    
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "avg_step_reward": avg_step_reward,
        "final_step_reward": final_step_reward,
        "num_steps": steps,
    }


def random_policy(observation: Dict) -> Dict:
    """Random policy - baseline for testing."""
    import random

    task_id = observation.get("task_id", "")

    if task_id == "fleet_precision":
        precisions = ["FP32", "BF16", "FP16", "FP8"]
        return {
            "precision_strategy": {
                "embedding": "FP32",
                "attention": random.choice(precisions),
                "ffn": random.choice(precisions),
                "layernorm": "BF16",
                "output": "FP32",
            },
            "reasoning": "random baseline",
        }

    elif task_id == "fleet_oversight":
        return {
            "action_type": "continue_monitoring",
            "analysis": "random baseline",
            "flagged_model": None,
            "flagged_step": None,
            "root_cause": None,
        }

    elif task_id == "fleet_resource":
        cluster = observation.get("cluster", {})
        models = observation.get("models", [])
        allocations = {}
        for m in models:
            mid = m["model_id"]
            gpus = max(1, cluster.get("total_gpus", 8) // len(models))
            allocations[mid] = {
                "gpus": gpus,
                "precision_strategy": {
                    "embedding": "FP32",
                    "attention": "BF16",
                    "ffn": "FP8",
                    "layernorm": "BF16",
                    "output": "FP32",
                },
            }
        return {"allocations": allocations, "reasoning": "random baseline"}

    elif task_id == "fleet_recovery":
        phase = observation.get("phase", "diagnose")
        if phase == "diagnose":
            return {
                "diagnosed_model": "model_a",
                "root_cause": "unknown",
                "reasoning": "random baseline",
            }
        elif phase == "reallocate":
            return {
                "new_precision_strategy": {
                    "embedding": "FP32",
                    "attention": "BF16",
                    "ffn": "FP8",
                    "layernorm": "BF16",
                    "output": "FP32",
                },
                "reasoning": "random baseline",
            }
        else:
            return {
                "reasoning": "random baseline",
                "confidence": "medium",
            }

    return {}


def greedy_policy(observation: Dict) -> Dict:
    """Simple greedy policy - uses known good defaults."""
    import random

    task_id = observation.get("task_id", "")

    if task_id == "fleet_precision":
        return {
            "precision_strategy": {
                "embedding": "FP32",
                "attention": "BF16",
                "ffn": "FP8",
                "layernorm": "BF16",
                "output": "FP32",
            },
            "reasoning": "greedy baseline using optimal defaults",
        }

    elif task_id == "fleet_oversight":
        trajectories = observation.get("model_trajectories", {})
        for mid, traj_data in trajectories.items():
            loss_window = traj_data.get("loss_window", [])
            if None in loss_window or any(v is None for v in loss_window):
                return {
                    "action_type": "flag_instability",
                    "analysis": f"Detected NaN in {mid}",
                    "flagged_model": mid,
                    "flagged_step": 35,
                    "root_cause": "fp8 embedding underflow",
                }
        return {
            "action_type": "continue_monitoring",
            "analysis": "No obvious issues",
            "flagged_model": None,
            "flagged_step": None,
            "root_cause": None,
        }

    elif task_id == "fleet_resource":
        cluster = observation.get("cluster", {})
        models = observation.get("models", [])
        total_gpus = cluster.get("total_gpus", 8)
        num_models = len(models)

        if num_models == 0:
            return {"allocations": {}, "reasoning": "no models"}

        # Priority-weighted allocation with budget cap
        total_priority = sum(m.get("priority", 1) for m in models)
        allocations = {}
        gpus_remaining = total_gpus
        for i, m in enumerate(models):
            mid = m["model_id"]
            priority = m.get("priority", 1)
            # Proportional to priority, minimum 1
            if i < num_models - 1:
                gpus = max(1, round(total_gpus * priority / max(total_priority, 1)))
                gpus = min(gpus, gpus_remaining - (num_models - i - 1))  # leave 1 for each remaining
            else:
                gpus = max(1, gpus_remaining)  # last model gets the rest
            gpus_remaining -= gpus
            allocations[mid] = {
                "gpus": gpus,
                "precision_strategy": {
                    "embedding": "FP32",
                    "attention": "BF16",
                    "ffn": "FP8",
                    "layernorm": "BF16",
                    "output": "FP32",
                },
            }
        return {"allocations": allocations, "reasoning": "priority-weighted allocation"}

    elif task_id == "fleet_recovery":
        crashed = observation.get("crashed_model", {})
        phase = observation.get("phase", "diagnose")
        if phase == "diagnose":
            crashed_config = crashed.get("precision_config", {})
            config_str = json.dumps(crashed_config).lower()
            cause = "fp8 embedding underflow"
            if "embedding" in config_str and "fp8" in config_str:
                cause = "fp8 embedding underflow"
            elif "output" in config_str and ("fp8" in config_str or "fp16" in config_str):
                cause = "low precision output layer"
            return {
                "diagnosed_model": crashed.get("model_id", "model_a"),
                "root_cause": cause,
                "reasoning": "analyzed crash data",
            }
        elif phase == "reallocate":
            return {
                "new_precision_strategy": {
                    "embedding": "FP32",
                    "attention": "BF16",
                    "ffn": "FP8",
                    "layernorm": "BF16",
                    "output": "FP32",
                },
                "reasoning": "switched to stable precision",
            }
        else:
            return {
                "reasoning": "stable config with monitoring plan",
                "confidence": "high",
            }

    return {}


def run_training(
    task_id: str,
    num_episodes: int = 100,
    policy_type: str = "greedy",
) -> Dict[str, List]:
    """Run training episodes and collect results."""
    if policy_type == "random":
        policy_fn = random_policy
    else:
        policy_fn = greedy_policy

    results = {
        "episodes": [],
        "rewards": [],
        "avg_step_rewards": [],
        "final_step_rewards": [],
        "num_steps": [],
        "best_reward": 0.0,
        "successes": 0,
    }

    print(f"\n{'='*60}")
    print(f"[TRAINING] task={task_id} episodes={num_episodes} policy={policy_type}")
    print(f"{'='*60}")

    for episode in range(num_episodes):
        episode_result = collect_episode(task_id, policy_fn)

        results["episodes"].append(episode)
        results["rewards"].append(episode_result["total_reward"])
        results["avg_step_rewards"].append(episode_result["avg_step_reward"])
        results["final_step_rewards"].append(episode_result["final_step_reward"])
        results["num_steps"].append(episode_result["num_steps"])

        if episode_result["total_reward"] > results["best_reward"]:
            results["best_reward"] = episode_result["total_reward"]
            
        # Consider an episode a "success" if the final step reward is decently high (>0.8)
        if episode_result["final_step_reward"] > 0.8:
            results["successes"] += 1

        if episode % 10 == 0:
            avg_final = sum(results["final_step_rewards"][-10:]) / min(10, len(results["final_step_rewards"]))
            print(f"[EPISODE] {episode:>3} total_reward={episode_result['total_reward']:.3f} final_step={episode_result['final_step_reward']:.3f} steps={episode_result['num_steps']} avg_final_last10={avg_final:.3f}")

    avg_total_reward = sum(results["rewards"]) / len(results["rewards"])
    avg_final_reward = sum(results["final_step_rewards"]) / len(results["final_step_rewards"])
    success_rate = results["successes"] / num_episodes
    print(f"\n[RESULT] task={task_id} success_rate={success_rate*100:.1f}% avg_final_reward={avg_final_reward:.3f} avg_total_reward={avg_total_reward:.3f} best_total={results['best_reward']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fleet training script")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "server"], help="Run mode")
    parser.add_argument("--task", type=str, default="fleet_precision", help="Task ID to train on")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--policy", type=str, default="greedy", choices=["random", "greedy"], help="Policy type")
    parser.add_argument("--output", type=str, default="training_results.json", help="Output file")
    args = parser.parse_args()

    if args.mode == "server":
        global reset_environment, step_environment
        reset_environment, step_environment = _init_server_mode()

    results = run_training(args.task, args.episodes, args.policy)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()