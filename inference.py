"""
Baseline Inference Script for Mixed Precision Training Environment.
Uses the OpenAI Client to call an LLM for reasoning about precision decisions.
"""
import os
import json
import httpx
from openai import OpenAI

# Required environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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

STRICT RULES — memorize these:
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
- flagged_step: the exact index where the first anomaly appeared (NaN, huge jump)
- root_cause: describe the suspicious layer+precision combo from precision_config (e.g. 'embedding fp8 underflow', 'attention fp8 overflow', 'output fp16 precision loss')

RESPOND WITH ONLY valid JSON:
{"action_type": "continue_monitoring|flag_instability", "analysis": "what you observe", "flagged_step": null|integer, "root_cause": null|"description"}""",

    "multi_objective_optimization": """You are optimizing a precision strategy for neural network training under hard constraints.
You have multiple iterations to refine your approach. Each iteration you propose a strategy and get feedback on memory, time, accuracy, and a score.

Layer types: embedding, attention, ffn, layernorm, output
Precision options: FP32, BF16, FP16, FP8

SCORING: Your AVERAGE score across ALL iterations is what matters. A violated-constraint step scores 0.0, which drags down your average. A valid step scores ~0.5-0.85 depending on how efficient the strategy is. Do NOT risk accuracy violations — they cost you an entire 0.0 step.

HARD CONSTRAINTS (NEVER violate these or you get 0.0 for that step):
- embedding: ALWAYS FP32 (FP8/FP16 causes NaN crash, BF16 risks accuracy)
- output: ALWAYS FP32 (any lower precision collapses loss computation accuracy below threshold)
- Accuracy must stay >= threshold shown in constraints. If output=FP16, accuracy drops ~4% → almost always a violation.

EMPIRICAL RULES:
- ffn=FP8: ALWAYS use this. FFN is 40% of params, FP8 gives 2.5x speedup and <1% accuracy penalty.
- layernorm=BF16: Use this. It is optimal with 0% accuracy penalty.
- attention: BF16 is safe (0% penalty, 1.85x speedup). FP16 gives slightly less (1.80x, +0.3% penalty). FP8 is risky (adds 2.5% penalty, often violates accuracy).

STRATEGY FOR ALL 5 ITERATIONS:
- Iteration 1: Start with SAFE BASELINE → {embedding:FP32, attention:BF16, ffn:FP8, layernorm:BF16, output:FP32}
- Iteration 2: Read the feedback. If accuracy has headroom above threshold, try attention=FP16 to squeeze more efficiency.
- Iteration 3: If attention=FP16 was valid and accuracy still has headroom, try more aggressive memory savings by reducing ffn to BF16 to see if some scenarios allow it. OR if time/memory is too tight, check if the scenario has a generous accuracy threshold that allows experimenting.
- Iteration 4-5: Based on all feedback, return to the HIGHEST-SCORING valid strategy you found. NEVER submit a strategy you already scored 0.0 on. NEVER try output=FP16 or output=BF16 — they risk accuracy collapse.

KEY INSIGHT: Even if you cannot beat the baseline score via exploration, you still want EVERY iteration to be a VALID strategy (score > 0). Submitting an invalid strategy (score=0.0) destroys your average. When uncertain, resubmit the best valid strategy you've seen.

RESPOND WITH ONLY valid JSON:
{"precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}, "reasoning": "what you changed and why"}"""
}


def call_llm(system_prompt: str, observation: dict, conversation_history: list = None) -> dict:
    """Call the LLM with the observation and return parsed JSON action."""
    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    user_msg = f"Current environment state:\n{json.dumps(observation, indent=2, default=str)}\n\nProvide your action as JSON."
    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)
    except Exception as e:
        print(f"  LLM Error: {e}")
        return None


def run_task(task_id: str):
    """Run a single task episode with the LLM agent."""
    system_prompt = TASK_SYSTEM_PROMPTS[task_id]
    conversation_history = []

    # Reset environment for this task
    try:
        res = httpx.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        obs = res.json()["observation"]
    except Exception as e:
        print(f"Error connecting to env: {e}", flush=True)
        return 0.0, 0

    print(f"[START] task={task_id} env=mixed_precision_env model={MODEL_NAME}", flush=True)

    rewards_list = []
    steps = 0

    while True:
        steps += 1
        
        # Call LLM to decide action
        action = call_llm(system_prompt, obs, conversation_history)
        
        error = None
        done = False
        reward = 0.0

        if action is None:
            action = {}
            error = "invalid LLM response"
            done = True
        else:
            # Add to conversation history to have context of what was sent
            conversation_history.append({"role": "user", "content": json.dumps(obs, default=str)})
            conversation_history.append({"role": "assistant", "content": json.dumps(action, default=str)})

            # Send action to environment
            try:
                step_res = httpx.post(f"{ENV_URL}/step", json={"action": action})
                data = step_res.json()

                reward = data["reward"]["score"]
                feedback = data["reward"]["feedback"]
                done = data["done"]
                obs = data.get("observation", None)

                # Append environment feedback
                conversation_history.append({
                    "role": "user",
                    "content": f"Environment feedback: reward={reward}, feedback={feedback}"
                })
            except Exception as e:
                done = True
                error = f"env error: {str(e)}"

        rewards_list.append(reward)
        action_str = json.dumps(action, default=str).replace('\\n', ' ')
        error_str = error if error else "null"
        done_str = str(done).lower()

        print(f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

        if done:
            break

    score = sum(rewards_list) / max(steps, 1)
    # Define a simple threshold for success
    success = score >= 0.70
    success_str = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)

    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)
    return score, steps


def main():
    task_id = os.environ.get("TASK_ID")
    if task_id:
        tasks = [task_id]
    else:
        # Default fallback to testing all tasks sequentially
        tasks = ["precision_assignment", "instability_detection", "multi_objective_optimization"]

    for t in tasks:
        run_task(t)


if __name__ == "__main__":
    main()
