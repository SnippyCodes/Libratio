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

# Try direct import, fallback to server
USE_SERVER = os.getenv("TRAIN_MODE", "").lower() == "server"

if USE_SERVER:
    import httpx
    ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
    FLEET_URL = f"{ENV_URL}/fleet"

    def reset_environment(task_id: str) -> Dict:
        res = httpx.post(f"{FLEET_URL}/reset", json={"task_id": task_id}, timeout=30.0)
        return res.json()["observation"]

    def step_environment(action: Dict) -> Dict[str, Any]:
        res = httpx.post(f"{FLEET_URL}/step", json={"action": action}, timeout=30.0)
        data = res.json()
        return {
            "observation": data.get("observation"),
            "reward": data["reward"]["score"],
            "feedback": data["reward"]["feedback"],
            "done": data["done"],
        }
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from environment.fleet_env import FleetEnvironment

    _fleet_env = FleetEnvironment()

    def reset_environment(task_id: str) -> Dict:
        result = _fleet_env.reset(task_id)
        return result

    def step_environment(action: Dict) -> Dict[str, Any]:
        result = _fleet_env.step(action)
        return {
            "observation": result.get("observation"),
            "reward": result["reward"]["score"],
            "feedback": result["reward"]["feedback"],
            "done": result["done"],
        }


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

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "total_reward": sum(rewards),
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

        base_gpus = total_gpus // num_models
        allocations = {}
        for m in models:
            mid = m["model_id"]
            priority = m.get("priority", 1)
            gpus = max(1, base_gpus + priority - 1)
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
        "num_steps": [],
        "best_reward": 0.0,
    }

    print(f"\n{'='*60}")
    print(f"[TRAINING] task={task_id} episodes={num_episodes} policy={policy_type}")
    print(f"{'='*60}")

    for episode in range(num_episodes):
        episode_result = collect_episode(task_id, policy_fn)

        results["episodes"].append(episode)
        results["rewards"].append(episode_result["total_reward"])
        results["num_steps"].append(episode_result["num_steps"])

        if episode_result["total_reward"] > results["best_reward"]:
            results["best_reward"] = episode_result["total_reward"]

        if episode % 10 == 0:
            avg_reward = sum(results["rewards"][-10:]) / min(10, len(results["rewards"]))
            print(f"[EPISODE] {episode:>3} reward={episode_result['total_reward']:.3f} steps={episode_result['num_steps']} avg_last10={avg_reward:.3f}")

    avg_reward = sum(results["rewards"]) / len(results["rewards"])
    print(f"\n[RESULT] task={task_id} avg_reward={avg_reward:.3f} best={results['best_reward']:.3f}")

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
        os.environ["TRAIN_MODE"] = "server"

    results = run_training(args.task, args.episodes, args.policy)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()