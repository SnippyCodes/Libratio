"""
QLoRA + GRPO Training Pipeline for Libratio Fleet

This implements the training flow described in the hackathon seminar:
1. Load base model (LLaMA-3.1-8B) in 4-bit QLoRA via Unsloth
2. Attach LoRA adapters to attention + FFN layers
3. Run GRPO episodes against Libratio Fleet environment
4. Train LoRA weights on reward signal
5. After training: merge LoRA → 16-bit full model
6. Export merged model for vLLM inference

Key insights from Daniel's session:
- "Weight sharing: launch vLLM instance and suck the weights out into
   Unsloth land to cut memory usage"
- "If you do QLoRA during training, you should also use QLoRA for inference"
- "After finishing RL process we can take weights and merge it back to 16-bit"

Usage (Colab with T4/A100):
    !pip install unsloth trl peft bitsandbytes accelerate
    %run qlora_grpo_training.py
"""

import os
import sys
import json
import random
from typing import Dict, List, Any

# ── Step 0: Add project root to path ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.fleet_env import FleetEnvironment


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load Model in QLoRA via Unsloth
# ══════════════════════════════════════════════════════════════════════════════

def load_model_qlora(model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"):
    """Load model in 4-bit QLoRA using Unsloth for maximum memory efficiency.

    Unsloth automatically patches the model for 2x faster training
    and uses ~60% less VRAM than standard QLoRA implementations.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Installing unsloth...")
        os.system("pip install -q unsloth")
        from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,        # QLoRA: 4-bit base model
        dtype=None,               # Auto-detect (float16 for T4, bfloat16 for A100)
    )

    # Attach LoRA adapters — target attention + FFN (the layers GRPO will tune)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                     # LoRA rank (16 is good balance of capacity vs memory)
        lora_alpha=32,            # Scaling factor (alpha/r = 2 is standard)
        lora_dropout=0.05,
        target_modules=[          # Which layers get LoRA adapters
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",       # FFN (SwiGLU)
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 60% less VRAM
        random_state=42,
    )

    print(f"[QLoRA] Model loaded: {model_name}")
    print(f"[QLoRA] Trainable params: {model.print_trainable_parameters()}")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Define Reward Functions Against Fleet Environment
# ══════════════════════════════════════════════════════════════════════════════

# System prompt for the model during training
FLEET_SYSTEM_PROMPT = """You are an ML infrastructure engineer managing GPU fleet precision.
Given a model's specs and cluster state, assign optimal precision to each layer.

RULES:
- embedding: ALWAYS FP32 (FP8 crashes)
- output: ALWAYS FP32 (loss needs full precision)
- ffn: FP8 is ideal (2.5x speedup)
- attention: BF16 is optimal (1.85x speedup)
- layernorm: BF16 is best

RESPOND WITH ONLY valid JSON:
{"precision_strategy": {"embedding": "FP32", "attention": "BF16", "ffn": "FP8", "layernorm": "BF16", "output": "FP32"}, "reasoning": "explanation"}"""


def _build_meta_header(task_id: str, scenario_id: str) -> str:
    """Attach deterministic metadata for reward reconstruction."""
    return f"[META] task_id={task_id}; scenario_id={scenario_id}"


def _extract_meta_from_prompt(prompt: Any, fallback_task_id: str) -> Dict[str, str]:
    """Extract task/scenario metadata from a prompt string."""
    text = str(prompt)
    marker = "[META]"
    task_id = fallback_task_id
    scenario_id = None

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(marker):
            continue
        payload = line[len(marker):].strip()
        for chunk in payload.split(";"):
            if "=" not in chunk:
                continue
            key, val = chunk.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key == "task_id" and val:
                task_id = val
            elif key == "scenario_id" and val:
                scenario_id = val
        break

    return {"task_id": task_id, "scenario_id": scenario_id}


def _extract_completion_text(completion: Any) -> str:
    """Handle different TRL completion payload formats safely."""
    if isinstance(completion, list):
        if not completion:
            return ""
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)


def build_training_prompts(task_id: str = "fleet_precision", num_prompts: int = 100):
    """Generate training prompts by resetting the fleet environment multiple times.

    Each prompt is a unique scenario observation that the model must respond to.
    The diversity of scenarios (7 fleet configs × random selection) ensures
    the model generalizes rather than memorizing.
    """
    env = FleetEnvironment()
    prompts = []

    for _ in range(num_prompts):
        obs = env.reset(task_id)
        scenario_id = obs.get("scenario_id", "")
        meta_header = _build_meta_header(task_id=task_id, scenario_id=scenario_id)
        prompt = f"Current environment state:\n{json.dumps(obs, indent=2, default=str)}\n\nProvide your action as JSON."

        prompts.append({
            "prompt": f"{meta_header}\n{prompt}",
            "system": FLEET_SYSTEM_PROMPT,
        })

    print(f"[DATA] Generated {len(prompts)} training prompts for {task_id}")
    return prompts


def fleet_reward_function(completions: List[str], prompts: List[str],
                          task_id: str = "fleet_precision") -> List[float]:
    """Score model completions against the Libratio Fleet environment.

    This is the reward function for GRPO. For each completion:
    1. Parse the JSON action from the model's output
    2. Reset a fresh environment episode
    3. Step with the parsed action
    4. Return the environment's reward score

    The environment already implements:
    - Physics-based scoring (per-layer precision quality)
    - Difference Rewards (counterfactual fleet contribution)
    - Inverse Reward Design (degenerate action penalty)
    - Hardware safety checks (thermal/memory/power)
    """
    rewards = []

    for completion, prompt in zip(completions, prompts):
        reward = 0.05
        try:
            # Parse JSON from completion
            content = _extract_completion_text(completion).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            action = json.loads(content)

            # Reconstruct the exact scenario from prompt metadata for deterministic credit assignment
            meta = _extract_meta_from_prompt(prompt, fallback_task_id=task_id)
            env = FleetEnvironment()
            env.reset(meta["task_id"], scenario_id=meta["scenario_id"])
            result = env.step(action)
            reward = float(result["reward"]["score"])

            # Bonus for valid JSON (format compliance)
            reward += 0.05

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Invalid JSON or missing fields = low reward
            reward = 0.05  # Not 0 — GRPO needs non-zero for gradient signal
        except Exception as e:
            print(f"[REWARD WARNING] Unexpected reward error: {type(e).__name__}: {e}")
            reward = 0.05

        rewards.append(max(0.01, min(0.99, reward)))

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: GRPO Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def run_grpo_training(
    model,
    tokenizer,
    task_id: str = "fleet_precision",
    num_prompts: int = 200,
    num_epochs: int = 3,
    batch_size: int = 1,
    num_generations: int = 4,    # G in GRPO — number of completions per prompt
    learning_rate: float = 2e-5,
    output_dir: str = "./results/qlora_grpo",
):
    """Run GRPO training using TRL's GRPOTrainer.

    GRPO (Group Relative Policy Optimization) works by:
    1. For each prompt, generate G completions
    2. Score all G with the reward function
    3. Use the relative ranking within the group as the advantage
    4. Update policy towards better-ranked completions

    This is more stable than PPO because it doesn't need a value function,
    and the group-relative scoring is self-normalizing.
    """
    try:
        from trl import GRPOTrainer, GRPOConfig
        from datasets import Dataset
    except ImportError:
        print("Installing trl and datasets...")
        os.system("pip install -q trl datasets")
        from trl import GRPOTrainer, GRPOConfig
        from datasets import Dataset

    # Generate training data
    training_prompts = build_training_prompts(task_id, num_prompts)

    # Convert to dataset format
    dataset = Dataset.from_list([
        {"prompt": p["system"] + "\n\n" + p["prompt"]}
        for p in training_prompts
    ])

    # GRPO config
    config = GRPOConfig(
        output_dir=output_dir,
        num_generations=num_generations,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        max_steps=500,               # Cap at 500 steps
        max_completion_length=512,
        logging_steps=10,
        save_steps=50,
        warmup_ratio=0.1,
        bf16=True,               # Use BF16 for training (A100/H100)
        report_to="none",        # Disable wandb for hackathon
    )

    # Define reward function wrapper for GRPO
    def reward_fn(completions, prompts):
        return fleet_reward_function(completions, prompts, task_id)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    print(f"\n{'='*60}")
    print(f"[GRPO] Starting training: {num_prompts} prompts × {num_epochs} epochs")
    print(f"[GRPO] Generations per prompt: {num_generations}")
    print(f"[GRPO] Learning rate: {learning_rate}")
    print(f"{'='*60}\n")

    # Train!
    trainer.train()

    # Save LoRA adapter
    trainer.save_model(output_dir)
    print(f"\n[GRPO] LoRA adapter saved to {output_dir}")

    return trainer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Merge LoRA Weights → 16-bit Full Model
# ══════════════════════════════════════════════════════════════════════════════

def merge_to_16bit(model, tokenizer, output_path: str = "./results/libratio-fleet-merged-16bit"):
    """Merge QLoRA adapter weights into the base model at 16-bit precision.

    This is the key step Daniel described:
    "After finishing RL process we do not necessarily need QLoRA —
     we can take weights and merge it back to 16-bit model"

    The merged model can then be served with vLLM for fast inference
    without needing the LoRA adapter at runtime.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("Unsloth not available, using manual merge...")
        model = model.merge_and_unload()
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"[MERGE] Saved merged 16-bit model to {output_path}")
        return

    # Unsloth-optimized merge (handles dtype conversion properly)
    model.save_pretrained_merged(
        output_path,
        tokenizer,
        save_method="merged_16bit",  # Merge LoRA → 16-bit
    )

    print(f"[MERGE] Saved merged 16-bit model to {output_path}")
    print(f"[MERGE] Model is now ready for vLLM inference")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Verify with Inference
# ══════════════════════════════════════════════════════════════════════════════

def verify_merged_model(model_path: str = "./results/libratio-fleet-merged-16bit"):
    """Quick verification that the merged model produces valid fleet actions.

    Optional: if vLLM is available, test with vLLM for production-speed inference.
    """
    env = FleetEnvironment()
    obs = env.reset("fleet_precision")

    prompt = f"{FLEET_SYSTEM_PROMPT}\n\nCurrent environment state:\n{json.dumps(obs, indent=2, default=str)}\n\nProvide your action as JSON."

    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model_path)
        params = SamplingParams(temperature=0.2, max_tokens=512)
        outputs = llm.generate([prompt], params)
        response = outputs[0].outputs[0].text
        print(f"[VERIFY] vLLM response: {response[:200]}...")
    except ImportError:
        print("[VERIFY] vLLM not available — using transformers for verification")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[VERIFY] Transformers response: {response[:200]}...")

    # Try to parse and score
    try:
        if "```json" in response:
            content = response.split("```json")[1].split("```")[0].strip()
        else:
            content = response.strip()
        action = json.loads(content)
        result = env.step(action)
        print(f"[VERIFY] Score: {result['reward']['score']:.3f}")
        print(f"[VERIFY] Feedback: {result['reward']['feedback']}")
    except Exception as e:
        print(f"[VERIFY] Could not parse response: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the complete QLoRA → GRPO → Merge pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="QLoRA + GRPO Training for Libratio Fleet")
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--task", default="fleet_precision")
    parser.add_argument("--prompts", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output", default="./results/qlora_grpo")
    parser.add_argument("--merge-output", default="./results/libratio-fleet-merged-16bit")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("LIBRATIO FLEET — QLoRA + GRPO Training Pipeline")
    print("=" * 60)
    print(f"Base model:    {args.model}")
    print(f"Task:          {args.task}")
    print(f"Prompts:       {args.prompts}")
    print(f"Epochs:        {args.epochs}")
    print(f"Generations:   {args.generations}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Step 1: Load in QLoRA
    print("\n[STEP 1/4] Loading model in QLoRA...")
    model, tokenizer = load_model_qlora(args.model)

    # Step 2-3: GRPO Training
    print("\n[STEP 2/4] Running GRPO training...")
    trainer = run_grpo_training(
        model, tokenizer,
        task_id=args.task,
        num_prompts=args.prompts,
        num_epochs=args.epochs,
        num_generations=args.generations,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    # Step 4: Merge to 16-bit
    if not args.skip_merge:
        print("\n[STEP 3/4] Merging LoRA → 16-bit...")
        merge_to_16bit(model, tokenizer, args.merge_output)
    else:
        print("\n[STEP 3/4] Skipping merge (--skip-merge)")

    # Step 5: Verify
    if not args.skip_verify and not args.skip_merge:
        print("\n[STEP 4/4] Verifying merged model...")
        verify_merged_model(args.merge_output)
    else:
        print("\n[STEP 4/4] Skipping verification")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
