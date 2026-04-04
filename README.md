---
title: Libratio — Mixed Precision Training Env
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - mixed-precision
  - reinforcement-learning
  - ml-engineering
---

# ⚖️ Libratio — Mixed Precision Training Environment

> *Libratio (Latin): "the act of balancing"* — the art of balancing precision, speed, and memory in modern AI training.

A reinforcement learning environment simulating one of the highest-stakes decisions in large-scale AI infrastructure: **configuring floating-point precision formats (FP32, BF16, FP16, FP8) for neural network layers** during training.

This is the exact problem Meta engineers face when training LLaMA at scale — assign the wrong precision to the wrong layer and you waste millions of dollars in GPU compute, or worse, crash an entire training run with NaN losses.

---

## 🎯 Motivation

Training a 70B parameter model on H100 GPUs costs ~$1M+. Precision misconfiguration is one of the leading causes of:
- **Training crashes** (NaN loss from FP8 on embedding layers)
- **Wasted compute** (using FP32 everywhere when BF16 is safe)
- **Memory overflows** (underestimating the memory footprint of your precision strategy)

Libratio provides a structured RL environment where agents can learn, be evaluated, and improve at these decisions — grounded in real empirical benchmarks from NVIDIA Transformer Engine and Meta's LLaMA-3 technical reports.

---

## 🏗️ Environment Architecture

```
POST /reset  →  { task_id }          →  Initial Observation
POST /step   →  { action }           →  { observation, reward, done, info }
POST /state  →  {}                   →  Current State
GET  /tasks  →  {}                   →  Task Definitions
```

The environment follows the [OpenEnv](https://openenv.dev) specification with typed Pydantic models for all actions and observations.

---

## 📊 Observation & Action Spaces

### Task 1: `precision_assignment` *(Easy)*

**Observation:**
```json
{
  "task_id": "precision_assignment",
  "scenario_id": "llama3_70b",
  "model_name": "LLaMA-3-70B",
  "total_layers": 5,
  "current_layer_index": 2,
  "current_layer": {
    "name": "ffn_layer",
    "layer_type": "ffn",
    "num_params": 28000000000,
    "gradient_sensitivity": "low",
    "activation_range": "bounded"
  },
  "assigned_so_far": { "embedding": "FP32", "attention": "BF16" },
  "available_precisions": ["FP32", "BF16", "FP16", "FP8"],
  "memory_budget_gb": 80.0,
  "memory_used_gb": 12.4,
  "speed_target_speedup": 2.0
}
```

**Action:**
```json
{ "precision": "FP8", "reasoning": "FFN layers have bounded activations safe for FP8" }
```

---

### Task 2: `instability_detection` *(Medium)*

**Observation:**
```json
{
  "task_id": "instability_detection",
  "scenario_id": "fp8_embedding_crash",
  "precision_config": { "embedding": "FP8", "attention": "BF16", "ffn": "FP8" },
  "total_training_steps": 100,
  "steps_revealed_so_far": 40,
  "loss_trajectory_window": [2.34, 2.18, 2.05, null, null, null],
  "window_index": 1,
  "windows_remaining": 3
}
```
> `null` values represent `NaN` — a training crash.

**Action:**
```json
{
  "action_type": "flag_instability",
  "analysis": "NaN values detected starting at step 23",
  "flagged_step": 23,
  "root_cause": "fp8 embedding underflow causing gradient collapse"
}
```

---

### Task 3: `multi_objective_optimization` *(Hard)*

**Observation:**
```json
{
  "task_id": "multi_objective_optimization",
  "scenario_id": "datacenter_gpu_cluster",
  "constraints": {
    "memory_budget_gb": 35.0,
    "time_budget_days": 5.5,
    "accuracy_threshold": 0.95
  },
  "model_total_params": 12000000000,
  "iterations_remaining": 4,
  "best_score_so_far": 0.718,
  "previous_feedback": "Valid! Mem=27.5GB, Time=3.8d, Acc=0.991, Speedup=1.81x"
}
```

**Action:**
```json
{
  "precision_strategy": {
    "embedding": "FP32",
    "attention": "BF16",
    "ffn": "FP8",
    "layernorm": "BF16",
    "output": "FP32"
  },
  "reasoning": "Baseline strategy satisfies all constraints with maximum FFN throughput"
}
```

---

## 🎮 Tasks & Difficulty

| Task | Difficulty | Max Steps | Description |
|---|---|---|---|
| `precision_assignment` | **Easy** | 5 | Assign FP32/BF16/FP16/FP8 to each layer one at a time based on its gradient sensitivity and activation range |
| `instability_detection` | **Medium** | 5 | Progressively observe training loss windows, detect precision-induced crashes, identify root cause |
| `multi_objective_optimization` | **Hard** | 5 | Design a precision strategy satisfying memory, time, and accuracy constraints simultaneously; iteratively refine |

### Reward Functions

**Task 1 — Per-step stability + efficiency score:**
```
FP32 on embedding  →  1.0  (IEEE 754 baseline, zero underflow risk)
BF16 on attention  →  1.0  (Kalamkar2019: 1.85x throughput, stable)
FP8 on ffn         →  1.0  (NVIDIA-TE 2023: 2.5x throughput, bounded activation)
FP8 on embedding   →  0.0  (catastrophic: gradient underflow)
```

**Task 2 — Timing + accuracy bonus:**
- `continue_monitoring` on stable data → `0.6` per step
- Correct `flag_instability` → `0.4` base + up to `0.3` for step accuracy + `0.3` for root cause match
- False alarm → `0.1` (heavy penalty)
- Missed crash → `0.1` (heavy penalty)

**Task 3 — Pareto frontier proximity:**
```python
score = 0.5 + 0.2 * mem_efficiency + 0.15 * time_efficiency + 0.15 * accuracy_margin
# 0.0 if any hard constraint is violated
```
Grounded in H100 cloud pricing + LLaMA-3 FLOPs data.

---

## ⚙️ Setup & Usage

### Option A: Docker (Recommended)

```bash
docker build -t libratio .
docker run -p 7860:7860 libratio
# API available at http://localhost:7860
```

### Option B: Local Python

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Running the Baseline Inference Agent

```bash
# Required environment variables
export API_BASE_URL="https://api.groq.com/openai/v1"   # or any OpenAI-compatible endpoint
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_api_key_here"

python inference.py
```

To run a single task:
```bash
export TASK_ID="precision_assignment"  # or instability_detection / multi_objective_optimization
python inference.py
```

---

## 📈 Baseline Scores

Evaluated using `llama-3.1-8b-instant` via Groq API (OpenAI-compatible):

| Task | Baseline Score | Expected Range | Notes |
|---|---|---|---|
| `precision_assignment` | **1.000** | 0.91 – 1.00 | Perfect — model knows FP32/BF16/FP8 rules |
| `instability_detection` | **0.680** | 0.60 – 0.85 | Scenario-dependent; improves with larger models |
| `multi_objective_optimization` | **0.718** | 0.65 – 0.85 | Scenario-dependent; constrained by physics ceiling |

> Scores vary slightly between runs due to random scenario selection. The physics model ensures graders are deterministic given the same scenario.

---

## 🔬 Physics Model Sources

All rewards are grounded in published empirical benchmarks:

| Claim | Source |
|---|---|
| BF16 attention → 1.85x speedup | Kalamkar et al. (2019), *BFloat16 for Deep Learning* |
| FP8 FFN → 2.5x speedup | NVIDIA Transformer Engine (2023) |
| FP8 embedding → NaN crash | IEEE 754 analysis, confirmed in LLaMA-3 training logs |
| H100 cost model | AWS/GCP H100 cloud pricing (2024) |
| LLaMA-3 FLOPs estimate | Meta AI (2024), *LLaMA 3 Technical Report* |

---

## 🗂️ Project Structure

```
Libratio/
├── app/
│   ├── main.py              # FastAPI app — /reset, /step, /state, /tasks
│   └── models.py            # Pydantic typed request/response models
├── environment/
│   ├── mixed_precision_env.py   # Core state machine & graders
│   └── physics_model.py         # Empirical benchmark constants & computations
├── scenarios/
│   ├── task1_scenarios.py   # Layer architectures (LLaMA-3 7B/13B/70B)
│   ├── task2_scenarios.py   # Loss trajectories (stable/unstable/spike)
│   └── task3_scenarios.py   # Optimization scenarios (tight/balanced/generous)
├── inference.py             # Baseline LLM agent (OpenAI-compatible)
├── openenv.yaml             # OpenEnv spec metadata & task definitions
├── Dockerfile               # Container for HF Spaces deployment
├── requirements.txt
└── .env.example             # Environment variable template
```

---

## 🧠 Why This Problem Matters

Every frontier AI lab (Meta, Google DeepMind, Anthropic) runs into this problem at scale:

- **GPT-4** training cost: estimated $100M+
- **Precision misconfiguration**: can waste 10-40% of compute budget
- **Manual tuning**: takes expert ML engineers days per model

An agent that can reliably configure mixed precision could save **millions per training run** and is directly applicable to production ML infrastructure.

---

*Built for the Meta × Scalar Hackathon 2026 | OpenEnv Specification*
