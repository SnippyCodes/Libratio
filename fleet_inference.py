"""
Fleet Inference Script: Multi-agent LLM inference for Libratio Fleet.

Each fleet task uses different agent roles with specialized system prompts.
The LLM is called once per agent per turn, with per-agent conversation history.

Usage:
    export API_BASE_URL="https://api.groq.com/openai/v1"
    export MODEL_NAME="llama-3.3-70b-versatile"
    export HF_TOKEN="your_api_key"
    python fleet_inference.py
"""
import os
import re
import json
import httpx
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Config
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
FLEET_URL = f"{ENV_URL}/fleet"

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def clamp_reward(r: float) -> float:
    """Ensure reward is strictly in (0, 1)."""
    try:
        val = float(r)
    except (ValueError, TypeError):
        val = 0.01
    return float(max(0.001, min(0.999, val)))


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — one per agent role
# ═══════════════════════════════════════════════════════════════════════════════

FLEET_PROMPTS = {
    "fleet_precision": """You are an ML infrastructure engineer managing ONE model in a shared GPU cluster.
Multiple models are training simultaneously and sharing a common GPU memory pool.

Your job: assign precision formats to YOUR model's layers. But be aware:
- Other agents are also using the same memory pool
- If you use too much memory, other models won't fit
- You need to balance YOUR model's performance with fleet-wide resource fairness

RULES:
- embedding: ALWAYS FP32 (FP8 crashes, BF16 is risky)
- output: ALWAYS FP32 (loss computation needs full precision)
- ffn: FP8 is ideal (2.5x speedup, safe for FFN layers)
- attention: BF16 is optimal (1.85x speedup)
- layernorm: BF16 is best

Consider your model's priority level — higher priority models may justify using more memory.
Consider how much memory other agents have already used.

RESPOND WITH ONLY valid JSON:
{"precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}, "reasoning": "brief explanation considering fleet context"}""",

    "fleet_oversight": """You are a Fleet AI Oversight Agent monitoring multiple neural network training runs simultaneously.
You see loss trajectories from ALL models in the cluster at once.

YOUR MISSION: Detect which model (if any) is experiencing precision-induced instability.

SCORING:
- continue_monitoring on healthy fleet = 0.55 per step (safe choice)
- CORRECT flag_instability = up to 1.0 (identify model + step + cause)
- FALSE ALARM = 0.10 (heavy penalty — don't cry wolf!)
- MISSED crash = 0.10 (heavy penalty — you failed your oversight duty)

DETECTION RULES:
1. Look for None/null values in any model's loss window → that's a NaN crash, flag it immediately
2. Look for sudden spikes (>100x previous value) that don't recover
3. Check each model's precision_config: FP8 on embedding or output = HIGH RISK
4. A spike that recovers within a few steps is NORMAL — do NOT flag
5. If all models look healthy, continue_monitoring

When flagging:
- flagged_model: which model_id is crashing (e.g., "model_b")
- flagged_step: the training step where the first anomaly appeared
- root_cause: the precision+layer combo causing the issue (e.g., "fp8 embedding underflow in model_b")

RESPOND WITH ONLY valid JSON:
{"action_type": "continue_monitoring|flag_instability", "analysis": "what you observe across all models", "flagged_model": null|"model_id", "flagged_step": null|integer, "root_cause": null|"description"}""",

    "fleet_resource": """You are a GPU Cluster Resource Manager allocating GPUs across multiple training jobs.

You must divide a shared GPU cluster among competing models, considering:
1. Each model's parameter count (bigger models need more GPUs for memory)
2. Priority levels (higher priority = should get proportionally more resources)
3. Precision strategies (aggressive precision like FP8 reduces memory → fewer GPUs needed)
4. Total cluster utilization (unused GPUs = wasted money)

RULES:
- Every model must get at least 1 GPU
- Total GPUs assigned cannot exceed cluster total
- Each model's memory usage must fit in its allocated GPU memory
- Higher priority models should generally get more resources
- Use efficient precision (FP8 for FFN, BF16 for attention) to minimize memory per model

STRATEGY:
- Iteration 1: Allocate proportionally to param count × priority. Use optimal precision.
- Iteration 2+: Adjust based on feedback. If a model doesn't fit, give it more GPUs or use less aggressive precision.

RESPOND WITH ONLY valid JSON:
{"allocations": {"model_a": {"gpus": N, "precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}}, "model_b": {...}}, "reasoning": "explanation of allocation logic"}""",

    "fleet_recovery": """You are a Fleet Recovery Specialist. A model in the GPU cluster has crashed mid-training.

You have 3 steps:
1. DIAGNOSE: Identify the root cause from the crash data and precision config
2. REALLOCATE: Propose a new, safe precision strategy for the crashed model
3. VERIFY: Confirm the recovery plan is sound and describe post-recovery monitoring

PHASE 1 - DIAGNOSE: Look at the loss trajectory around the crash point. Check the precision config for dangerous combinations (FP8 on embedding/output, FP16 on output).
RESPOND WITH: {"diagnosed_model": "model_id", "root_cause": "description of what caused the crash", "reasoning": "your analysis"}

PHASE 2 - REALLOCATE: Propose a new precision strategy that fixes the crash cause. Key: switch any dangerous precision to a safe alternative (embedding→FP32, output→FP32).
RESPOND WITH: {"new_precision_strategy": {"embedding": "FP32", ...}, "reasoning": "what you changed and why"}

PHASE 3 - VERIFY: Explain why the new config is safe and describe your monitoring plan.
RESPOND WITH: {"reasoning": "detailed explanation of why recovery will be stable and what to monitor post-restart", "confidence": "high|medium|low"}""",
}


def call_llm(system_prompt: str, observation: dict, conversation_history: list = None, task_id: str = None) -> dict:
    """Call the LLM with the observation and return parsed JSON action."""
    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    user_msg = f"Current environment state:\n{json.dumps(observation, indent=2, default=str)}\n\nProvide your action as JSON."
    messages.append({"role": "user", "content": user_msg})

    # Dynamic routing: use custom model for precision/recovery, Groq for oversight/resource
    current_model = MODEL_NAME
    current_base_url = API_BASE_URL
    current_api_key = os.getenv("HF_TOKEN")

    if task_id in ["fleet_oversight", "fleet_resource"] or "groq.com" in current_base_url:
        current_model = "llama-3.3-70b-versatile"
        current_base_url = "https://api.groq.com/openai/v1"
        current_api_key = os.getenv("GROQ_API_KEY")

    if not current_api_key:
        raise ValueError(f"Required API key not found in environment for {current_base_url}")

    local_client = OpenAI(api_key=current_api_key, base_url=current_base_url)

    try:
        response = local_client.chat.completions.create(
            model=current_model,
            messages=messages,
            temperature=0.2,
            max_tokens=600,
        )
        content = response.choices[0].message.content.strip()

        # Extract the first matching JSON object by brace counting
        first_brace = content.find('{')
        if first_brace != -1:
            count = 0
            in_string = False
            escape = False
            for i in range(first_brace, len(content)):
                char = content[i]
                if char == '"' and not escape:
                    in_string = not in_string
                elif char == '\\' and in_string:
                    escape = not escape
                    continue
                elif not in_string:
                    if char == '{':
                        count += 1
                    elif char == '}':
                        count -= 1
                        if count == 0:
                            content = content[first_brace:i+1]
                            break
                escape = False

        # Strip trailing commas inside objects or arrays (common LLM generation issue)
        content = re.sub(r",\s*([\]}])", r"\1", content)

        return json.loads(content)
    except Exception as e:
        print(f"  LLM Error: {e}")
        return None


def run_fleet_task(task_id: str):
    """Run a complete fleet task episode."""
    system_prompt = FLEET_PROMPTS[task_id]
    conversation_history = []

    # Reset fleet environment
    try:
        res = httpx.post(f"{FLEET_URL}/reset", json={"task_id": task_id})
        obs = res.json()["observation"]
    except Exception as e:
        print(f"Error connecting to fleet env: {e}", flush=True)
        return 0.01, 0

    print(f"\n{'='*60}", flush=True)
    print(f"[FLEET START] task={task_id} model={MODEL_NAME}", flush=True)
    print(f"{'='*60}", flush=True)

    rewards_list = []
    steps = 0

    while True:
        steps += 1

        # Call LLM
        action = call_llm(system_prompt, obs, conversation_history, task_id)

        error = None
        done = False
        reward = 0.01

        if action is None:
            # Fallback action
            if task_id == "fleet_precision":
                action = {"precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}, "reasoning": "fallback"}
            elif task_id == "fleet_oversight":
                action = {"action_type": "continue_monitoring", "analysis": "fallback"}
            elif task_id == "fleet_resource":
                action = {"allocations": {}, "reasoning": "fallback"}
            elif task_id == "fleet_recovery":
                action = {"reasoning": "fallback", "root_cause": "unknown", "diagnosed_model": "unknown"}
            error = "LLM returned invalid response, using fallback"

        # Add to conversation history
        conversation_history.append({"role": "user", "content": json.dumps(obs, default=str)})
        conversation_history.append({"role": "assistant", "content": json.dumps(action, default=str)})

        # Send action to fleet environment
        try:
            step_res = httpx.post(f"{FLEET_URL}/step", json={"action": action})
            data = step_res.json()

            reward = clamp_reward(data["reward"]["score"])
            feedback = data["reward"]["feedback"]
            done = data["done"]
            obs = data.get("observation", None)

            # Log step-by-step metrics to MongoDB Time Series collection
            try:
                from mongodb_metrics import log_step_metrics
                cluster_info = obs.get("cluster", {}) if obs else {}
                model_info = obs.get("your_model", {}) if obs else {}
                mem_used = cluster_info.get("memory_used_gb")
                if mem_used is None:
                    mem_used = model_info.get("memory_used_gb", 0.0)
                therm = cluster_info.get("thermal_risk")
                if therm is None:
                    therm = model_info.get("thermal_risk", "UNKNOWN")
                pwr = cluster_info.get("power_util", cluster_info.get("estimated_power_pct", 0.0))
                m_id = model_info.get("model_id", MODEL_NAME)
                
                log_step_metrics(
                    task_id=task_id,
                    model_id=m_id,
                    step=steps,
                    reward=reward,
                    memory_used_gb=mem_used,
                    thermal_risk=therm,
                    power_util=pwr
                )
            except Exception:
                pass

            # Add feedback to conversation history
            conversation_history.append({
                "role": "user",
                "content": f"Environment feedback: reward={reward:.3f}, feedback={feedback}"
            })
        except Exception as e:
            done = True
            error = f"env error: {str(e)}"

        rewards_list.append(clamp_reward(reward))
        action_summary = json.dumps(action, default=str)[:150]
        error_str = error if error else "null"

        print(f"[STEP] step={steps} reward={reward:.3f} done={done} error={error_str}", flush=True)
        print(f"       action={action_summary}", flush=True)

        if done:
            break

    score = clamp_reward(sum(rewards_list) / max(steps, 1))
    success = score >= 0.60
    rewards_str = ",".join(f"{r:.3f}" for r in rewards_list)

    print(f"[FLEET END] task={task_id} score={score:.3f} steps={steps} success={success}", flush=True)
    print(f"            rewards=[{rewards_str}]", flush=True)

    # Log run telemetry to MongoDB runs collection
    mongo_uri = os.getenv("MONGO_URI")
    if mongo_uri:
        try:
            from pymongo import MongoClient
            from datetime import datetime
            client_mongo = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            db_mongo = client_mongo["libratio"]
            runs_col = db_mongo["runs"]
            runs_col.insert_one({
                "task_id": task_id,
                "model_name": MODEL_NAME,
                "score": score,
                "steps": steps,
                "success": success,
                "rewards": rewards_list,
                "timestamp": datetime.now(timezone.utc)
            })
            print("[OK] Logged run telemetry to MongoDB Atlas ('runs' collection)")
        except Exception as e:
            print(f"[WARN] Could not log telemetry to MongoDB: {e}")

    return score, steps


def main():
    """Run all fleet tasks or a specific one."""
    # Initialize MongoDB Time Series metrics collection
    try:
        from mongodb_metrics import create_metrics_timeseries_collection
        create_metrics_timeseries_collection()
    except Exception as e:
        print(f"[WARN] Failed to initialize Time Series metrics: {e}")

    task_id = os.environ.get("TASK_ID")

    if task_id:
        tasks = [task_id]
    else:
        tasks = ["fleet_precision", "fleet_oversight", "fleet_resource", "fleet_recovery"]

    results = {}
    for t in tasks:
        score, steps = run_fleet_task(t)
        results[t] = {"score": score, "steps": steps}

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print("FLEET BENCHMARK SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Task':<25} {'Score':>8} {'Steps':>6}", flush=True)
    print("-" * 42, flush=True)
    for t, r in results.items():
        print(f"{t:<25} {r['score']:>8.3f} {r['steps']:>6}", flush=True)
    avg = sum(r["score"] for r in results.values()) / max(len(results), 1)
    print("-" * 42, flush=True)
    print(f"{'AVERAGE':<25} {avg:>8.3f}", flush=True)


if __name__ == "__main__":
    main()
