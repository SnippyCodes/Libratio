# ============================================================
# LIBRATIO FLEET - QLoRA + GRPO Training (Single Colab Cell)
# ============================================================
# 1. Open Google Colab (colab.research.google.com)
# 2. Change Runtime: Runtime > Change runtime type > T4 GPU
# 3. Paste this ENTIRE script into one cell
# 4. Hit Shift+Enter and wait ~2-3 hours
# ============================================================

# ── Step 1: Install dependencies ────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("Installing dependencies...")
install("unsloth")
install("trl")
install("datasets")
install("peft")
install("bitsandbytes")
install("accelerate")
install("transformers")
print("Dependencies installed!")

# ── Step 2: Clone your repo ────────────────────────────────
import os

REPO_URL = "https://github.com/SnippyCodes/Libratio.git"  # <-- UPDATE THIS to your actual GitHub repo URL
REPO_DIR = "./Libratio"

if not os.path.exists(REPO_DIR):
    print(f"Cloning repo from {REPO_URL}...")
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
else:
    print(f"Repo already exists at {REPO_DIR}")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"Working directory: {os.getcwd()}")

# ── Step 3: Verify environment works ───────────────────────
from environment.fleet_env import FleetEnvironment
import json
from typing import List

env = FleetEnvironment()
obs = env.reset("fleet_precision")
print(f"Environment OK! Scenario: {obs.get('scenario_id', 'unknown')}")
print(f"Fleet scenarios loaded, cluster: {obs['cluster']['total_gpus']} GPUs")

# ── Step 4: Define reward function ─────────────────────────

def clamp(x):
    return max(0.01, min(0.99, float(x)))

SYSTEM_PROMPT = """You are an ML infra engineer managing precision for one model in a shared GPU fleet.

RULES:
- embedding: ALWAYS FP32 (FP8 crashes training)
- output: ALWAYS FP32 (loss needs full precision)
- ffn: FP8 is ideal (2.5x speedup, stable)
- attention: BF16 is optimal (1.85x speedup)
- layernorm: BF16 is best

Return ONLY valid JSON:
{"precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}, "reasoning": "..."}"""


def build_prompts(task_id="fleet_precision", n=128):
    """Generate diverse training prompts from fleet scenarios."""
    env = FleetEnvironment()
    out = []
    for _ in range(n):
        obs = env.reset(task_id)
        out.append({
            "prompt": (
                f"{SYSTEM_PROMPT}\n\n"
                f"Observation:\n{json.dumps(obs, indent=2, default=str)}\n\n"
                f"Return JSON action only."
            ),
        })
    print(f"Generated {len(out)} training prompts")
    return out


def parse_json_action(text):
    """Extract JSON from model output (handles markdown fences)."""
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(t)


def fleet_reward_function(completions, prompts, **kwargs):
    """GRPO reward: score each completion against fleet environment."""
    rewards = []
    for completion, prompt in zip(completions, prompts):
        try:
            # Handle different TRL versions
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = str(completion)
                
            prompt_text = str(prompt)
            
            # Extract scenario_id from the JSON observation in the prompt
            scenario_id = None
            if "Observation:\n{" in prompt_text:
                try:
                    obs_str = prompt_text.split("Observation:\n")[1].split("\n\nReturn")[0]
                    obs = json.loads(obs_str)
                    scenario_id = obs.get("scenario_id")
                except:
                    pass

            action = parse_json_action(text)
            env = FleetEnvironment()
            env.reset("fleet_precision", scenario_id=scenario_id)
            result = env.step(action)
            r = clamp(result["reward"]["score"]) + 0.03  # bonus for valid JSON
            rewards.append(clamp(r))
        except Exception:
            rewards.append(0.05)
    return rewards


# Quick sanity check
test_rewards = fleet_reward_function(
    ['{"precision_strategy":{"embedding":"FP32","attention":"BF16","ffn":"FP8","layernorm":"BF16","output":"FP32"}}', 'bad json'],
    ["p1", "p2"]
)
print(f"Reward function test: good={test_rewards[0]:.3f}, bad={test_rewards[1]:.3f}")

# ── Step 5: Load model in QLoRA ────────────────────────────
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

print(f"\nLoading {MODEL_NAME} in 4-bit QLoRA...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("Model loaded with LoRA adapters!")

# ── Step 6: Build training dataset ─────────────────────────
from datasets import Dataset

NUM_PROMPTS = 128  # <-- Increase for better training (200+)
samples = build_prompts(n=NUM_PROMPTS)
ds = Dataset.from_list([{"prompt": s["prompt"]} for s in samples])
print(f"Dataset: {len(ds)} samples")

# ── Step 7: Train with GRPO ────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

OUT_DIR = "./results/qlora_grpo_fleet"

cfg = GRPOConfig(
    output_dir=OUT_DIR,
    num_generations=4,             # 4 completions per prompt for relative ranking
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=2,
    max_completion_length=512,
    logging_steps=10,
    save_steps=50,
    warmup_ratio=0.1,
    fp16=True,      # T4 GPU requires fp16 (bfloat16 is Ampere+ only)
    bf16=False,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=cfg,
    train_dataset=ds,
    reward_funcs=[fleet_reward_function],
)

print("\n" + "=" * 60)
print("STARTING GRPO TRAINING")
print(f"Prompts: {NUM_PROMPTS} | Epochs: 2 | Generations: 4")
print("=" * 60 + "\n")

trainer.train()

# ── Step 7.5: Plot Training Curves ─────────────────────────
import matplotlib.pyplot as plt

history = trainer.state.log_history
steps = []
rewards = []
losses = []

for log in history:
    if "step" in log:
        if "reward" in log:
            steps.append(log["step"])
            rewards.append(log["reward"])
        if "loss" in log:
            losses.append(log["loss"])

if steps and rewards:
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards, marker='o', color='g', label='Mean Reward')
    plt.title('GRPO Training: Mean Fleet Reward')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig('reward_curve.png')
    plt.show()

if losses:
    # Match the steps length to the losses length for plotting
    loss_steps = [log["step"] for log in history if "loss" in log]
    plt.figure(figsize=(10, 5))
    plt.plot(loss_steps, losses, marker='o', color='r', label='Training Loss')
    plt.title('GRPO Training: Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

# ── Step 8: Save results ───────────────────────────────────
trainer.save_model(OUT_DIR)
print(f"\nLoRA adapter saved to {OUT_DIR}")

# Merge to 16-bit
MERGED_DIR = OUT_DIR + "_merged_16bit"
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
print(f"Merged 16-bit model saved to {MERGED_DIR}")

# ── Step 9: Download (optional) ────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"LoRA adapter: {OUT_DIR}")
print(f"Merged model: {MERGED_DIR}")
print("\nTo download, run in a new cell:")
print("  from google.colab import files")
print("  !zip -r fleet_model.zip ./results")
print("  files.download('fleet_model.zip')")
print("  files.download('reward_curve.png')")
print("  files.download('loss_curve.png')")