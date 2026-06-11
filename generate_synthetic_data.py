"""
Generate a rich, diverse synthetic dataset of GPU cluster trajectories
for the Libratio Predictive MLOps Agent.

This script uses the Agentic Kernel physics model to simulate 1000 episodes
with realistic, randomized mixed-precision strategies across 7 fleet scenarios.

Fixes applied vs v1:
  1. Mixed-precision strategies generated randomly per-layer (not uniform)
  2. Added realistic mixed configs alongside uniform ones
  3. Enriched output with model_name, cost, speedup, accuracy, memory_utilization
  4. Seeded RNG for reproducibility
  5. Added fleet-level metadata (total_memory_used, memory_overflow)
"""
import json
import random
from environment.fleet_env import FleetEnvironment
from environment.physics_model import compute_training_cost, compute_hardware_safety

# All available precisions per layer type, ordered from safest to most aggressive
LAYER_PRECISIONS = {
    "embedding":  ["FP32", "BF16"],           # FP8/FP16 crash on embedding (stability < 0.5)
    "attention":  ["BF16", "FP16", "FP8"],    # FP32 is overkill, FP8 is risky
    "ffn":        ["FP8", "BF16", "FP16"],    # FP8 is the ideal target here
    "layernorm":  ["BF16", "FP16"],           # FP8 is dangerous (stability = 0.2)
    "output":     ["FP32"],                    # Must always be FP32 (stability < 0.5 for everything else)
}

# Also include some deliberately bad choices to create crash data
RISKY_LAYER_PRECISIONS = {
    "embedding":  ["FP8", "FP16"],             # Will cause CRASH_NUMERICAL_INSTABILITY
    "attention":  ["FP8"],                     # Risky
    "ffn":        ["FP32"],                    # Wasteful but stable
    "layernorm":  ["FP8"],                     # Dangerous
    "output":     ["FP16", "BF16", "FP8"],     # Will cause crashes
}


def random_safe_strategy():
    """Generate a random but physically plausible mixed-precision strategy."""
    return {
        layer: random.choice(options)
        for layer, options in LAYER_PRECISIONS.items()
    }


def random_risky_strategy():
    """Generate a strategy that has at least one dangerous layer choice."""
    strategy = random_safe_strategy()
    # Pick 1-2 layers to make risky
    num_risky = random.randint(1, 2)
    risky_layers = random.sample(list(RISKY_LAYER_PRECISIONS.keys()), num_risky)
    for layer in risky_layers:
        strategy[layer] = random.choice(RISKY_LAYER_PRECISIONS[layer])
    return strategy


def random_uniform_strategy():
    """Generate a uniform precision strategy (all layers same precision)."""
    precision = random.choice(["FP32", "BF16", "FP8"])
    return {layer: precision for layer in LAYER_PRECISIONS.keys()}


def generate_dataset(num_episodes=1000, seed=42):
    random.seed(seed)
    env = FleetEnvironment()
    dataset = []

    # Strategy distribution: 50% safe mixed, 30% risky mixed, 20% uniform
    strategy_generators = [
        (random_safe_strategy,    0.50),
        (random_risky_strategy,   0.30),
        (random_uniform_strategy, 0.20),
    ]

    print(f"Generating {num_episodes} synthetic MLOps trajectories (seed={seed})...")

    for i in range(num_episodes):
        obs = env.reset("fleet_precision")
        cluster = env.task_state["cluster"]
        models = env.task_state["models"]

        episode_data = {
            "episode_id": i,
            "scenario_id": env.scenario["scenario_id"],
            "cluster_capacity_gb": cluster["total_memory_gb"],
            "total_gpus": cluster["total_gpus"],
            "gpu_memory_gb": cluster.get("gpu_memory_gb", 80.0),
            "trajectories": [],
            "fleet_total_memory_used_gb": 0.0,
        }

        for idx, model in enumerate(models):
            # Pick a strategy generator based on weighted distribution
            roll = random.random()
            cumulative = 0.0
            chosen_gen = random_safe_strategy
            for gen_fn, weight in strategy_generators:
                cumulative += weight
                if roll < cumulative:
                    chosen_gen = gen_fn
                    break

            strategy = chosen_gen()

            # Compute physics using the kernel
            metrics = compute_training_cost(
                model["total_params"], strategy, model["layer_distribution"]
            )
            num_gpus_for_model = max(1, cluster["total_gpus"] // len(models))
            hw_safety = compute_hardware_safety(
                model["total_params"], strategy, model["layer_distribution"],
                num_gpus_for_model, cluster.get("gpu_memory_gb", 80.0)
            )

            # Determine outcome from physics
            outcome = "SUCCESS"
            failure_reason = None
            if not metrics["estimated_stable"]:
                outcome = "CRASH_NUMERICAL_INSTABILITY"
                # Find which layer caused it
                from environment.physics_model import STABILITY_SCORE
                for layer, precision in strategy.items():
                    stab = STABILITY_SCORE.get(layer, {}).get(precision, 1.0)
                    if stab < 0.5:
                        failure_reason = f"{layer} at {precision} (stability={stab})"
                        break
            elif not hw_safety["overall_safe"]:
                outcome = "CRASH_THERMAL_THROTTLE"
                if not hw_safety["memory_safe"]:
                    failure_reason = f"memory_utilization={hw_safety['memory_utilization_pct']}%"
                else:
                    failure_reason = f"power_draw={hw_safety['estimated_power_pct']}%"

            traj_record = {
                "model_id": model["model_id"],
                "model_name": model["name"],
                "total_params_b": round(model["total_params"] / 1e9, 1),
                "precision_strategy": strategy,
                "memory_used_gb": metrics["memory_gb"],
                "memory_utilization_pct": hw_safety["memory_utilization_pct"],
                "thermal_risk": hw_safety["thermal_risk"],
                "estimated_power_pct": hw_safety["estimated_power_pct"],
                "speedup_vs_fp32": metrics["speedup_vs_fp32"],
                "cost_usd": metrics["cost_usd"],
                "accuracy_retention": metrics["accuracy_retention"],
                "outcome": outcome,
            }
            if failure_reason:
                traj_record["failure_reason"] = failure_reason

            episode_data["trajectories"].append(traj_record)
            episode_data["fleet_total_memory_used_gb"] += metrics["memory_gb"]

            # Step the environment to advance the agent turn
            env.step({"precision_strategy": strategy})

        # Round fleet total and check for overflow
        episode_data["fleet_total_memory_used_gb"] = round(
            episode_data["fleet_total_memory_used_gb"], 2
        )
        episode_data["memory_overflow"] = (
            episode_data["fleet_total_memory_used_gb"] > cluster["total_memory_gb"]
        )

        dataset.append(episode_data)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_episodes} episodes...")

    # Print dataset statistics
    all_outcomes = [t["outcome"] for ep in dataset for t in ep["trajectories"]]
    total = len(all_outcomes)
    success = all_outcomes.count("SUCCESS")
    crash_num = all_outcomes.count("CRASH_NUMERICAL_INSTABILITY")
    crash_therm = all_outcomes.count("CRASH_THERMAL_THROTTLE")

    print(f"\n--- Dataset Statistics ---")
    print(f"  Total trajectory records: {total}")
    print(f"  SUCCESS:                   {success} ({100*success/total:.1f}%)")
    print(f"  CRASH_NUMERICAL_INSTAB:    {crash_num} ({100*crash_num/total:.1f}%)")
    print(f"  CRASH_THERMAL_THROTTLE:    {crash_therm} ({100*crash_therm/total:.1f}%)")

    # Count unique strategies
    unique_strats = set()
    for ep in dataset:
        for t in ep["trajectories"]:
            key = tuple(sorted(t["precision_strategy"].items()))
            unique_strats.add(key)
    print(f"  Unique precision strategies: {len(unique_strats)}")

    with open("synthetic_trajectories.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDone! Saved {len(dataset)} episodes to synthetic_trajectories.json")


if __name__ == "__main__":
    generate_dataset()
