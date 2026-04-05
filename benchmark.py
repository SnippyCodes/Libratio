"""
Benchmark script — runs inference across multiple models and generates a comparison chart.
"""
import os
import json
import httpx
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from openai import OpenAI

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""

MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]

TASKS = ["precision_assignment", "instability_detection", "multi_objective_optimization"]
TASK_SHORT = ["Task 1\nPrecision", "Task 2\nInstability", "Task 3\nOptimization"]

TASK_SYSTEM_PROMPTS = {
    "precision_assignment": """You are an expert ML infrastructure engineer configuring mixed-precision training.
You will be shown ONE layer at a time from a neural network. Assign a precision format to it.

Key rules:
- FP32: Safest. Required for embedding layers (gradient underflow risk) and output/loss layers.
- BF16: Good balance. Works well for attention and layernorm layers.
- FP16: Similar to BF16 but narrower dynamic range.
- FP8: Maximum speed/memory savings. Only safe for FFN (feed-forward) layers.

RESPOND WITH ONLY valid JSON:
{"precision": "FP32|BF16|FP16|FP8", "reasoning": "brief explanation"}""",

    "instability_detection": """You are diagnosing a neural network training run for precision-induced instability.
You will see windows of the loss trajectory progressively revealed. Each window = 20 steps.

YOUR SCORE: Each 'continue_monitoring' = 0.6 points. A CORRECT 'flag_instability' = up to 1.0. A WRONG 'flag_instability' = 0.1 (false alarm penalty). If you never flag and training was unstable, you score 0.1 for that final step.

STRICT RULES:
1. REAL instability = loss values contain None/NaN, OR a value is >100x the previous value AND does NOT recover
2. A spike that recovers within 3-5 steps is NORMAL FLUCTUATION. Do NOT flag it.
3. Gradual INCREASING trend for 5+ consecutive steps MAY indicate instability if it persists.
4. Look at the precision_config: FP8 on embedding or output = high risk. FP8 on attention = moderate risk.
5. If you see NaN/None in the window, that is definitive — flag it immediately.
6. If you see a big spike (>5x) but the next values are back to normal range, that is a SPIKE — do NOT flag.

DECISION PROTOCOL:
- Window 1: ALWAYS continue_monitoring unless you see NaN/None or a value >1000.
- Window 2+: Only flag if you see: (a) NaN/None values, (b) loss > 10x its value from 5 steps earlier AND not recovering, (c) sustained upward trend for all 10+ remaining steps.
- When in doubt, continue_monitoring. False alarms cost you heavily (0.1 vs 0.6).

If you flag instability:
- flagged_step: the exact index where the first anomaly appeared
- root_cause: describe the suspicious layer+precision combo

RESPOND WITH ONLY valid JSON:
{"action_type": "continue_monitoring|flag_instability", "analysis": "what you observe", "flagged_step": null|integer, "root_cause": null|"description"}""",

    "multi_objective_optimization": """You are optimizing a precision strategy for neural network training under hard constraints.
You have multiple iterations to refine your approach.

Layer types: embedding, attention, ffn, layernorm, output
Precision options: FP32, BF16, FP16, FP8

HARD CONSTRAINTS (NEVER violate):
- embedding: ALWAYS FP32
- output: ALWAYS FP32
- Accuracy must stay >= threshold shown in constraints.

EMPIRICAL RULES:
- ffn=FP8: ALWAYS use this. FFN is 40% of params, FP8 gives 2.5x speedup.
- layernorm=BF16: Use this. Optimal with 0% accuracy penalty.
- attention: BF16 is safe (0% penalty, 1.85x speedup). FP16 gives slightly less.

STRATEGY: Start with {embedding:FP32, attention:BF16, ffn:FP8, layernorm:BF16, output:FP32} then iterate.
Every iteration MUST be a valid strategy (score > 0). Never submit something that violates constraints.

RESPOND WITH ONLY valid JSON:
{"precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}, "reasoning": "what you changed and why"}"""
}


def call_llm(client, model, system_prompt, observation, conversation_history=None):
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    user_msg = f"Current environment state:\n{json.dumps(observation, indent=2, default=str)}\n\nProvide your action as JSON."
    messages.append({"role": "user", "content": user_msg})
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        print(f"    LLM Error ({model}): {e}")
        return None


def run_task(client, model, task_id):
    system_prompt = TASK_SYSTEM_PROMPTS[task_id]
    conversation_history = []
    try:
        res = httpx.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        obs = res.json()["observation"]
    except Exception as e:
        print(f"  ENV error: {e}")
        return 0.0

    rewards = []
    steps = 0
    while True:
        steps += 1
        action = call_llm(client, model, system_prompt, obs, conversation_history)
        if action is None:
            rewards.append(0.0)
            break

        conversation_history.append({"role": "user", "content": json.dumps(obs, default=str)})
        conversation_history.append({"role": "assistant", "content": json.dumps(action, default=str)})

        try:
            step_res = httpx.post(f"{ENV_URL}/step", json={"action": action})
            data = step_res.json()
            reward = data["reward"]["score"]
            feedback = data["reward"]["feedback"]
            done = data["done"]
            obs = data.get("observation")
            conversation_history.append({"role": "user", "content": f"Feedback: reward={reward}, {feedback}"})
            rewards.append(reward)
            if done:
                break
        except Exception as e:
            rewards.append(0.0)
            break

    avg = sum(rewards) / max(len(rewards), 1)
    return round(avg, 3)


def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    results = {}  # model -> {task_id: score}

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model}")
        print(f"{'='*60}")
        results[model] = {}
        for task in TASKS:
            print(f"  Running {task}...", end=" ", flush=True)
            start = time.time()
            score = run_task(client, model, task)
            elapsed = time.time() - start
            results[model][task] = score
            print(f"score={score:.3f}  ({elapsed:.1f}s)")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'MODEL':<30} {'Task1':>8} {'Task2':>8} {'Task3':>8} {'AVG':>8}")
    print(f"{'-'*70}")
    for model in MODELS:
        scores = results[model]
        avg = sum(scores.values()) / len(scores)
        print(f"{model:<30} {scores[TASKS[0]]:>8.3f} {scores[TASKS[1]]:>8.3f} {scores[TASKS[2]]:>8.3f} {avg:>8.3f}")
    print(f"{'='*70}")

    # Generate chart
    fig, ax = plt.subplots(figsize=(12, 7))

    x = range(len(TASKS))
    width = 0.25
    colors = ['#6366f1', '#f59e0b', '#10b981']  # indigo, amber, emerald

    for i, model in enumerate(MODELS):
        scores = [results[model][t] for t in TASKS]
        bars = ax.bar([xi + i * width for xi in x], scores, width,
                      label=model, color=colors[i], edgecolor='white', linewidth=0.5)
        # Add score labels on top of each bar
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Average Score', fontsize=13, fontweight='bold')
    ax.set_title('Libratio — Model Benchmark Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(TASK_SHORT, fontsize=11, ha='center')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.4, label='Target threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Chart saved to: {chart_path}")

    # Save raw data
    data_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(data_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Raw data saved to: {data_path}")


if __name__ == "__main__":
    main()
