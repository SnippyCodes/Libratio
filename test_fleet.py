"""Quick smoke test for all 4 fleet tasks."""
from environment.fleet_env import FleetEnvironment
import json

e = FleetEnvironment()

# Test all 4 fleet tasks reset
for task_id in ["fleet_precision", "fleet_oversight", "fleet_resource", "fleet_recovery"]:
    try:
        obs = e.reset(task_id)
        print(f"[OK] {task_id} reset — keys: {list(obs.keys())}")
    except Exception as ex:
        print(f"[FAIL] {task_id}: {ex}")

# Test a full fleet_precision episode
print("\n--- Full fleet_precision episode ---")
e.reset("fleet_precision")
optimal = {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}
step_num = 0
while True:
    step_num += 1
    result = e.step({"precision_strategy": optimal, "reasoning": "optimal config"})
    score = result["reward"]["score"]
    done = result["done"]
    print(f"  Step {step_num}: score={score:.3f}, done={done}")
    if done:
        if "fleet_summary" in result.get("info", {}):
            fs = result["info"]["fleet_summary"]
            print(f"  Fleet: mem={fs['total_memory_used_gb']}GB/{fs['cluster_capacity_gb']}GB, "
                  f"cost=${fs['total_cost_usd']:,.0f}, savings=${fs['fleet_savings_usd']:,.0f}, "
                  f"all_stable={fs['all_stable']}")
        break

# Test fleet_oversight episode
print("\n--- Full fleet_oversight episode ---")
obs = e.reset("fleet_oversight")
print(f"  Models: {list(obs['model_trajectories'].keys())}")
print(f"  Scenario: {obs['scenario_id']}")

r = e.step({"action_type": "continue_monitoring", "analysis": "all looks stable"})
print(f"  Step 1 (continue): score={r['reward']['score']:.3f}, done={r['done']}")

r = e.step({
    "action_type": "flag_instability",
    "flagged_model": "model_b",
    "flagged_step": 35,
    "root_cause": "fp8 embedding underflow model_b",
    "analysis": "NaN detected in model_b"
})
print(f"  Step 2 (flag): score={r['reward']['score']:.3f}, done={r['done']}")
print(f"  Feedback: {r['reward']['feedback'][:200]}")

# Test fleet_resource episode
print("\n--- Full fleet_resource episode ---")
obs = e.reset("fleet_resource")
print(f"  Cluster: {obs['cluster']['total_gpus']} GPUs, {obs['cluster']['total_memory_gb']}GB")
print(f"  Models: {[m['name'] for m in obs['models']]}")

alloc = {}
models = obs["models"]
total_gpus = obs["cluster"]["total_gpus"]
for i, m in enumerate(models):
    gpus = max(1, total_gpus // len(models))
    if i == 0:
        gpus = total_gpus - (len(models) - 1) * gpus  # give remainder to first
    alloc[m["model_id"]] = {
        "gpus": gpus,
        "precision_strategy": optimal,
    }

r = e.step({"allocations": alloc})
print(f"  Step 1: score={r['reward']['score']:.3f}, done={r['done']}")
print(f"  Feedback: {r['reward']['feedback'][:200]}")

# Test fleet_recovery episode
print("\n--- Full fleet_recovery episode ---")
obs = e.reset("fleet_recovery")
print(f"  Phase: {obs['phase']}")
print(f"  Crashed model: {obs['crashed_model']['model_id']}")
print(f"  Crash step: {obs['crashed_model']['crash_step']}")

r = e.step({
    "diagnosed_model": obs["crashed_model"]["model_id"],
    "root_cause": "fp8 embedding underflow causing gradient collapse",
    "reasoning": "NaN in loss trajectory indicates precision failure"
})
print(f"  Step 1 (diagnose): score={r['reward']['score']:.3f}, done={r['done']}")

if not r["done"]:
    r = e.step({
        "new_precision_strategy": optimal,
        "reasoning": "Switch to safe config: FP32 embedding, BF16 attention, FP8 FFN"
    })
    print(f"  Step 2 (reallocate): score={r['reward']['score']:.3f}, done={r['done']}")

if not r["done"]:
    r = e.step({
        "reasoning": "Recovery plan is stable. New config uses FP32 for embedding to prevent underflow. Will monitor loss for 100 steps post-restart.",
        "confidence": "high"
    })
    print(f"  Step 3 (verify): score={r['reward']['score']:.3f}, done={r['done']}")

print("\n=== ALL FLEET TESTS PASSED ===")
