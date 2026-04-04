# Mixed Precision Training Configuration — OpenEnv Environment
**Hackathon:** Scalar × Meta PyTorch OpenEnv Hackathon 2026  
**Submission Deadline:** April 8, 2026 (submitting April 7)  
**Participant:** Solo  
**Success Probability:** 75%  

---

## Table of Contents
1. [Terminology & Concepts](#1-terminology--concepts)
2. [The Problem Domain](#2-the-problem-domain)
3. [Environment Design: 3 Tasks Per Episode](#3-environment-design-3-tasks-per-episode)
4. [Grader Design](#4-grader-design)
5. [Scenario Design (25 Pre-Built)](#5-scenario-design-25-pre-built)
6. [OpenEnv Specification](#6-openenv-specification)
7. [Baseline Agent](#7-baseline-agent)
8. [Project Structure](#8-project-structure)
9. [Pre-Submission Checklist](#9-pre-submission-checklist)
10. [Scoring Rubric](#10-scoring-rubric)
11. [5-Day Build Timeline](#11-5-day-build-timeline)

---

# 1. Terminology & Concepts

Before diving into the project, here's every term you need to understand. This is your reference glossary.

---

## 1.1 Floating Point Precision Formats

Computers represent decimal numbers using **floating-point** formats. Each format uses a fixed number of **bits** (binary digits) to store a number, split into three parts:

```
┌──────┬──────────────┬──────────────────┐
│ Sign │   Exponent   │     Mantissa     │
│ 1bit │   (range)    │   (precision)    │
└──────┴──────────────┴──────────────────┘
```

- **Sign bit:** Is the number positive or negative? (1 bit)
- **Exponent:** How large/small is the number? (determines **dynamic range**)
- **Mantissa (Significand):** How precise is the number? (determines **accuracy**)

### FP32 — Single Precision (32 bits)
```
Format:  1 sign + 8 exponent + 23 mantissa = 32 bits
Memory:  4 bytes per number
Range:   ±1.18 × 10⁻³⁸ to ±3.40 × 10³⁸
Precision: ~7 decimal digits
```
- **The gold standard** for neural network training
- Every parameter, gradient, and activation is stored in FP32 by default
- **Pros:** Very accurate, huge range, no stability issues
- **Cons:** Slow computation, uses the most memory (4 bytes per value)
- **Used for:** Master weight copies, loss computation, gradient accumulation

### FP16 — Half Precision (16 bits)
```
Format:  1 sign + 5 exponent + 10 mantissa = 16 bits
Memory:  2 bytes per number (50% savings vs FP32)
Range:   ±6.10 × 10⁻⁵ to ±65,504
Precision: ~3.3 decimal digits
```
- **The workhorse** of mixed precision training
- 2× less memory, 2-8× faster computation on modern GPUs (Tensor Cores)
- **Pros:** Fast, good enough for most forward/backward pass operations
- **Cons:** Limited range — very small gradients (< 6.1×10⁻⁵) become zero (**underflow**)
- **Requires:** Loss scaling to prevent gradient underflow
- **Used for:** Attention layers, layer normalization, most computation

### BF16 — Brain Floating Point (16 bits)
```
Format:  1 sign + 8 exponent + 7 mantissa = 16 bits
Memory:  2 bytes per number (50% savings vs FP32)
Range:   Same as FP32 (±1.18 × 10⁻³⁸ to ±3.40 × 10³⁸)
Precision: ~2.4 decimal digits (less precise than FP16!)
```
- Invented by Google Brain — same range as FP32, but lower precision
- **Pros:** No loss scaling needed (same range as FP32), drop-in replacement
- **Cons:** Less precise than FP16 (only 7 mantissa bits vs 10)
- **Used for:** When FP16 underflows too much; popular in LLaMA, GPT training

### FP8 — Quarter Precision (8 bits)
```
Two variants:
  E4M3:  1 sign + 4 exponent + 3 mantissa = 8 bits  (more precision)
  E5M2:  1 sign + 5 exponent + 2 mantissa = 8 bits  (more range)

Memory:  1 byte per number (75% savings vs FP32)
Range:   E4M3: ±448, E5M2: ±57,344
Precision: ~1-2 decimal digits
```
- **Cutting edge** — only works on H100/H200 GPUs and newer
- 4× less memory than FP32, potential 2× speedup over FP16
- **Pros:** Maximum speed and memory savings
- **Cons:** Very risky — can cause NaN/divergence if used on sensitive layers
- **Used for:** FFN (feed-forward) layers with normalized inputs, where values are bounded

### Comparison Table

| Format | Bits | Bytes | Speed vs FP32 | Memory vs FP32 | Stability Risk | Best For |
|--------|------|-------|---------------|----------------|----------------|----------|
| FP32   | 32   | 4     | 1.0×          | 100%           | None           | Embeddings, loss, master weights |
| BF16   | 16   | 2     | 2-4×          | 50%            | Low            | General training (modern GPUs) |
| FP16   | 16   | 2     | 2-8×          | 50%            | Medium         | Attention, normalized layers |
| FP8    | 8    | 1     | 4-16×         | 25%            | High           | FFN layers with bounded values |

---

## 1.2 Neural Network Layer Types

### Embedding Layer
```
What it does:  Converts token IDs (integers) into dense vectors
Example:       Token "hello" (ID: 5923) → [0.012, -0.034, 0.089, ...]
Values range:  Tiny, unbounded (0.001 to 0.1 typically)
Gradients:     Very small (1e-5), high variance
```
- **Why FP32:** Gradients are so small they underflow to 0 in FP16/FP8, killing learning
- **This is the first layer** — errors here propagate through the entire model

### Attention Layer (Self-Attention / Multi-Head Attention)
```
What it does:  Lets the model "look at" all other tokens to understand context
Computation:   Q × K^T / √d_k → Softmax → × V
Values range:  Normalized by LayerNorm, typically [-2, +2]
Gradients:     Medium magnitude, stable
```
- **Why FP16/BF16:** Values are normalized, midrange — FP16 handles this well
- Contains the famous **Q, K, V** (Query, Key, Value) matrices

### Feed-Forward Network (FFN) Layer
```
What it does:  Applies non-linear transformation: FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
Values range:  Normalized inputs, bounded outputs [-1, +1]
Gradients:     Stable, well-behaved
```
- **Why FP8 is possible:** After LayerNorm, values are bounded — FP8 can represent them
- **Largest compute cost** in a transformer — using FP8 here gives maximum speedup
- **Risk:** If values occasionally spike, FP8 clips them → NaN propagation

### LayerNorm (Layer Normalization)
```
What it does:  Normalizes activations to mean=0, variance=1
Formula:       LayerNorm(x) = (x - μ) / √(σ² + ε) × γ + β
Values range:  Output always centered around 0, spread ~[-3, +3]
```
- **Why FP16:** Normalization keeps values in a safe range for FP16
- **Critical role:** Makes downstream layers (attention, FFN) safe for lower precision

### Output / LM Head Layer
```
What it does:  Final projection to vocabulary size, produces logits for next token
Values range:  Can be large (logits range: -100 to +100)
Gradients:     Must be precise for correct loss computation
```
- **Why FP32:** Loss computation (cross-entropy) requires high precision
- Wrong loss → wrong gradients → model learns the wrong thing

---

## 1.3 Training Concepts

### Loss
The **loss** is a single number that measures how wrong the model's predictions are.
- **Lower loss = better model**
- Training minimizes loss over many iterations (steps)
- Example: Cross-entropy loss for language models

### Loss Curve / Training Trajectory
```
A sequence of loss values over training steps:

Step 0:    loss = 4.50  ← Random model, high loss
Step 100:  loss = 3.20  ← Learning patterns
Step 500:  loss = 1.50  ← Getting better
Step 1000: loss = 0.87  ← Converging
Step 1500: loss = 0.85  ← Almost converged
```
- **Healthy curve:** Monotonically decreasing, eventually plateaus
- **Unstable curve:** Spikes, oscillations, or sudden NaN

### NaN (Not a Number)
When a computation produces an invalid result (0/0, ∞-∞, overflow):
```
Step 95:   loss = 0.87
Step 96:   loss = 0.86
Step 97:   loss = 1e+15   ← Explosion starts
Step 98:   loss = inf     ← Overflow
Step 99:   loss = NaN     ← Training is dead 💀
```
- **FP8 cause:** Small gradient underflows to 0 → division by zero → NaN
- **FP16 cause:** Large activation overflows past 65,504 → inf → NaN
- **Once NaN appears, training cannot recover** — it propagates everywhere

### Gradient
The gradient tells each parameter how to change to reduce the loss:
```
gradient = ∂Loss / ∂parameter

If gradient is:
  +0.01  → decrease this parameter slightly
  -0.05  → increase this parameter more
  0.0    → don't change (could be underflow!)
```
- **Gradient underflow:** When using FP16/FP8, very small gradients (< 6.1×10⁻⁵) round to 0
- **Gradient explosion:** When gradients become very large (> 65,504 in FP16) → inf → NaN

### Loss Scaling
A technique to prevent gradient underflow in FP16:
```
1. Multiply loss by a large number (e.g., 1024) before backward pass
2. This scales all gradients up by 1024×
3. After computing gradients, divide by 1024 to get real values
4. Now small gradients that would have been 0 are representable
```
- **Dynamic loss scaling:** Automatically adjusts the scale factor
- **Not needed for BF16** (same exponent range as FP32)

### Mixed Precision Training (The Core Concept)
Instead of using one precision for everything, **strategically mix** precisions:
```
┌─────────────────────────────────────────────────┐
│              Mixed Precision Pipeline            │
│                                                  │
│  Master Weights (FP32)                           │
│       ↓ cast to FP16                             │
│  Forward Pass (FP16) ──→ Loss (FP32)             │
│       ↓                      ↓                   │
│  Backward Pass (FP16) ←── Scaled Loss            │
│       ↓                                          │
│  Gradients (FP16) ──→ Unscale ──→ FP32 Gradients│
│       ↓                                          │
│  Update Master Weights (FP32)                    │
└─────────────────────────────────────────────────┘
```
- **Per-layer precision:** Our environment goes further — each layer gets its own precision
- This is what the RL agent must learn to configure

---

## 1.4 RL & OpenEnv Concepts

### Reinforcement Learning (RL)
An agent learns by trial and error:
```
Agent ──(action)──→ Environment ──(observation + reward)──→ Agent
                         ↑                                    │
                         └────────────────────────────────────┘
```
- **Agent:** The AI that makes decisions (our baseline = rule-based agent)
- **Environment:** The simulation (our mixed precision training simulator)
- **Action:** What the agent does (assign FP32/FP16/FP8 to each layer)
- **Observation:** What the agent sees (model architecture, loss curves, constraints)
- **Reward:** Score for how good the action was (our grader computes this)

### Episode
One complete run through all tasks:
```
Episode = Reset → Task 1 → Task 2 → Task 3 → Done
```
Our environment has 3 tasks per episode, 25 scenarios total.

### OpenEnv Framework
Meta's standard for packaging RL environments:
- **Gymnasium-compatible API:** `reset()`, `step()`, `state()`
- **Containerized:** Runs inside Docker
- **HTTP endpoints:** `/tasks`, `/baseline`, `/grader` — accessible via REST API
- **Deployed on:** HuggingFace Spaces
- **Graders:** Must be **deterministic** — same input always produces same score

### Grader
The function that scores an agent's action:
```python
def grader(agent_action, ground_truth) -> float:
    # Returns a score (0-100)
    # MUST be deterministic (no randomness, no LLM calls)
```
- Our graders use **lookup tables** — pre-computed correct answers
- This makes them fast, deterministic, and easy to validate

### Baseline Agent
A simple, non-AI agent that demonstrates the environment works:
```python
class BaselineAgent:
    def act(self, observation):
        # Simple rules, no ML needed
        if layer_type == "embedding":
            return "FP32"  # Always safe
        elif layer_type == "attention":
            return "FP16"  # Usually works
        ...
```
- Must score > 0 on at least one task (proves environment is solvable)
- Judges use it to understand the environment quickly

---

## 1.5 Infrastructure Concepts

### Docker
A container system that packages code + dependencies into a single image:
```
Dockerfile → docker build → Docker Image → docker run → Running Container
```
- **Why:** Ensures judges can run your environment on any machine
- **Requirement:** Your Dockerfile must build without errors

### HuggingFace Spaces
A platform to deploy and host ML applications:
- **Your environment runs here** as a web service
- Judges access your `/tasks`, `/baseline`, `/grader` endpoints
- Must deploy successfully to be eligible for scoring

### API Endpoints (Required)
```
GET  /tasks     → Returns list of task IDs and descriptions
POST /baseline  → Runs baseline agent, returns actions + scores
POST /grader    → Accepts agent actions, returns scores
POST /reset     → Resets environment to new episode
POST /step      → Executes one step in the environment
```

---

# 2. The Problem Domain

## What Problem Are We Solving?

Training large neural networks is expensive. The #1 lever for reducing cost is **mixed precision training** — strategically using lower-precision number formats (FP16, FP8) instead of FP32 for different layers.

**Today:** ML engineers manually choose precision per layer through trial and error (takes days of experimentation per model).

**Our environment:** An RL agent learns the optimal precision assignment in seconds.

### Real-World Impact
```
Training a 7B parameter model (like LLaMA-7B):

❌ Naive approach (FP32 everywhere):
   Memory: 120 GB VRAM
   Time:   21 days on 8×A100
   Cost:   $50,000
   Status: Doesn't fit on most GPU clusters

✅ Smart mixed precision (what our agent learns):
   Memory: 45 GB VRAM  (62% reduction)
   Time:   7 days       (3× faster)
   Cost:   $15,000      (70% savings)
   Status: Fits on a single A100
```

### Why Meta Cares
- Meta trains models like **LLaMA** (7B to 405B parameters)
- Every training run costs **$1M-$100M** in compute
- A 30% memory reduction saves **millions of dollars** per run
- Mixed precision configuration is done manually today — **this is a real pain point**

---

# 3. Environment Design: 3 Tasks Per Episode

Each episode presents a model training scenario. The agent must complete 3 sequential tasks:

```
Episode Flow:
┌──────────┐    ┌──────────────┐    ┌────────────────┐
│  Task 1  │───→│    Task 2    │───→│     Task 3     │
│ Precision │    │  Instability │    │ Multi-Objective │
│Assignment │    │  Detection   │    │  Optimization   │
└──────────┘    └──────────────┘    └────────────────┘
```

---

### Task 1: Static Precision Assignment

**Goal:** Assign FP32/FP16/FP8 to each layer of a given model architecture.

**What the agent sees (observation):**
```json
{
  "scenario_id": "gpt2_1.5b",
  "model_architecture": [
    {
      "layer_type": "embedding",
      "name": "token_embedding",
      "num_params": 300000000,
      "gradient_sensitivity": "high_variance",
      "activation_range": "unbounded"
    },
    {
      "layer_type": "attention_block",
      "name": "attention_0",
      "num_params": 200000000,
      "gradient_sensitivity": "medium",
      "activation_range": "normalized"
    },
    {
      "layer_type": "ffn",
      "name": "ffn_0",
      "num_params": 300000000,
      "gradient_sensitivity": "stable",
      "activation_range": "normalized"
    },
    {
      "layer_type": "output",
      "name": "lm_head",
      "num_params": 200000000,
      "gradient_sensitivity": "high_precision_critical",
      "activation_range": "loss_computation"
    }
  ],
  "memory_budget_gb": 40,
  "speed_target_speedup": 2.5,
  "accuracy_threshold": 0.995
}
```

**What the agent outputs (action):**
```json
{
  "precision_config": {
    "token_embedding": "FP32",
    "attention_0": "FP16",
    "ffn_0": "FP8",
    "lm_head": "FP32"
  },
  "reasoning": "High-variance embedding needs FP32. Normalized attention safe for FP16. Stable FFN can handle FP8. Output must be FP32 for loss computation."
}
```

**Scoring:**
- Exact match with optimal config → **100 points** + efficiency bonus
- Valid alternative (stable but suboptimal) → **50-90 points**
- Unstable config (would cause NaN) → **0 points**

---

### Task 2: Detecting Precision-Induced Instability

**Goal:** Given a training loss trajectory, detect if/when instability occurs and identify the cause.

**What the agent sees (observation):**
```
Training loss trajectory (100 steps):
Step 0:    loss = 4.50
Step 10:   loss = 3.20
Step 20:   loss = 2.15
Step 50:   loss = 1.20
Step 75:   loss = 0.98
Step 95:   loss = 0.87
Step 100:  loss = NaN 💀

Current precision config from Task 1
```

**What the agent outputs (action):**
```json
{
  "instability_detected": true,
  "step_detected": 95,
  "likely_cause": "FFN layer FP8 precision loss accumulating",
  "recommendation": "Change FFN layers to FP16",
  "confidence": 0.78
}
```

**Scoring:**
- Correct detection (unstable/stable) → **30 points**
- Early detection (before failure) → **+20 points**
- Correct root cause identification → **+40 points**
- Total possible: **90 points**

---

### Task 3: Multi-Objective Optimization

**Goal:** Given hard constraints (memory, time, accuracy), find the best precision strategy.

**What the agent sees (observation):**
```json
{
  "constraints": {
    "memory_budget_gb": 24,
    "time_budget_days": 7,
    "accuracy_threshold": 0.98,
    "dataset_size": "2T tokens",
    "model": "GPT-3.5-class (175B params)"
  },
  "current_baseline": {
    "all_fp32": {
      "memory_gb": 140,
      "time_days": 21,
      "accuracy": 1.0
    }
  }
}
```

**What the agent outputs (action):**
```json
{
  "precision_strategy": {
    "embedding": "FP32",
    "transformer_blocks": ["FP16", "FP16", "FP8", "FP8"],
    "output": "FP32"
  },
  "predicted_metrics": {
    "memory_gb": 22,
    "time_days": 6.5,
    "accuracy": 0.989,
    "efficiency_gain": "2.8x faster, 6x less memory"
  },
  "tradeoff_reasoning": "Fits 24GB budget. FP8 on FFN layers provides maximum speedup. Accuracy stays above 0.98 threshold."
}
```

**Scoring:**
- Memory within budget → **20 points**
- Time within budget → **20 points**
- Accuracy above threshold → **20 points**
- Pareto optimality (closeness to ideal) → **up to 40 points**
- Total possible: **100 points**

---

# 4. Grader Design

All graders are **deterministic** — no randomness, no LLM calls, same input → same output.

### Task 1 Grader: Precision Config Grader
```python
def grade_precision_assignment(agent_config, scenario_id):
    """Lookup-table based grading"""
    ground_truth = PRECISION_CONFIGS_LOOKUP[scenario_id]
    
    # First: is the config stable? (won't cause NaN)
    if not is_stable_config(agent_config):
        return 0  # Unstable config = instant fail
    
    if agent_config == ground_truth["optimal"]:
        return 100 + efficiency_bonus(agent_config, scenario_id)
    else:
        # Partial credit for valid alternatives
        efficiency = compute_efficiency(agent_config)
        optimal_efficiency = compute_efficiency(ground_truth["optimal"])
        return 50 + int((efficiency / optimal_efficiency) * 50)

def is_stable_config(config):
    """Rule-based stability validation"""
    STABILITY_RULES = {
        "embedding": ["FP32"],           # MUST be FP32
        "attention": ["FP16", "BF16", "FP32"],  # FP16+ 
        "ffn":      ["FP8", "FP16", "BF16", "FP32"],  # Any
        "layernorm": ["FP16", "BF16", "FP32"],  # FP16+
        "output":   ["FP32"],            # MUST be FP32
    }
    for layer_type, allowed in STABILITY_RULES.items():
        if config.get(layer_type) not in allowed:
            return False
    return True
```

### Task 2 Grader: Instability Detection Grader
```python
def grade_instability_detection(agent_output, ground_truth):
    score = 0
    
    # Correct detection (+30)
    if agent_output["instability_detected"] == ground_truth["is_unstable"]:
        score += 30
    
    # Early detection bonus (+20 or +10)
    if ground_truth["is_unstable"]:
        failure_step = ground_truth["failure_step"]
        detected_step = agent_output["step_detected"]
        if detected_step < failure_step - 5:
            score += 20  # Caught it early
        elif detected_step <= failure_step:
            score += 10  # Caught it on time
    
    # Root cause identification (+40)
    cause_keywords = set(ground_truth["cause_keywords"])
    agent_keywords = set(agent_output["likely_cause"].lower().split())
    overlap = len(cause_keywords & agent_keywords)
    if overlap >= len(cause_keywords) * 0.5:
        score += 40
    elif overlap > 0:
        score += 20
    
    return score
```

### Task 3 Grader: Multi-Objective Grader
```python
def grade_multi_objective(agent_strategy, constraints, ground_truth):
    score = 0
    metrics = agent_strategy["predicted_metrics"]
    
    # Constraint satisfaction (20 each)
    if metrics["memory_gb"] <= constraints["memory_budget_gb"]:
        score += 20
    if metrics["time_days"] <= constraints["time_budget_days"]:
        score += 20
    if metrics["accuracy"] >= constraints["accuracy_threshold"]:
        score += 20
    
    # Pareto distance scoring (0-40)
    pareto_distance = compute_pareto_distance(
        metrics, ground_truth["pareto_frontier"]
    )
    score += max(0, int(40 * (1 - pareto_distance)))
    
    return score
```

---

# 5. Scenario Design (25 Pre-Built)

### Category 1: Small Models (5 scenarios)
| ID | Model | Params | Key Challenge |
|----|-------|--------|---------------|
| `small_1` | GPT-2 Small | 125M | Basic precision assignment |
| `small_2` | BERT Base | 110M | Bidirectional attention |
| `small_3` | DistilGPT-2 | 82M | Already compressed model |
| `small_4` | GPT-2 Medium | 355M | More layers, more decisions |
| `small_5` | T5 Small | 60M | Encoder-decoder architecture |

### Category 2: Medium Models (5 scenarios)
| ID | Model | Params | Key Challenge |
|----|-------|--------|---------------|
| `medium_1` | GPT-2 XL | 1.5B | Large attention layers |
| `medium_2` | LLaMA-1.3B | 1.3B | Modern architecture |
| `medium_3` | Mistral-1.5B | 1.5B | Grouped-query attention |
| `medium_4` | Phi-2 | 2.7B | Efficient architecture |
| `medium_5` | CodeLLaMA-3B | 3B | Code-specific model |

### Category 3: Large Models (5 scenarios)
| ID | Model | Params | Key Challenge |
|----|-------|--------|---------------|
| `large_1` | LLaMA-7B | 7B | Standard large model |
| `large_2` | LLaMA-13B | 13B | Memory-critical |
| `large_3` | Falcon-40B | 40B | Extreme memory pressure |
| `large_4` | LLaMA-70B | 70B | Multi-GPU required |
| `large_5` | GPT-175B | 175B | Maximum scale challenge |

### Category 4: Instability Scenarios (5 scenarios)
| ID | Failure Type | Steps to NaN | Root Cause |
|----|-------------|-------------|------------|
| `unstable_1` | Gradient underflow | Step 847 | FP8 on embedding |
| `unstable_2` | Activation overflow | Step 234 | FP16 on unbounded FFN |
| `unstable_3` | Loss explosion | Step 1500 | Accumulated rounding error |
| `unstable_4` | Slow divergence | Step 5000 | FP8 attention with long context |
| `unstable_5` | Stable (no failure) | N/A | Correctly configured |

### Category 5: Multi-Objective Constraint Scenarios (5 scenarios)
| ID | Memory Budget | Time Budget | Accuracy Floor | Difficulty |
|----|--------------|-------------|----------------|------------|
| `constraint_1` | 24 GB | 7 days | 0.98 | Tight memory |
| `constraint_2` | 80 GB | 3 days | 0.99 | Tight time |
| `constraint_3` | 16 GB | 14 days | 0.995 | Extreme memory |
| `constraint_4` | 40 GB | 5 days | 0.97 | Balanced |
| `constraint_5` | 8 GB | 30 days | 0.99 | Edge device |

---

# 6. OpenEnv Specification

```yaml
# openenv.yaml
name: MixedPrecisionTrainingEnvironment
version: 1.0.0
description: >
  RL environment for optimizing neural network training with mixed precision
  (FP32/FP16/BF16/FP8). Agent learns to assign precision formats to each
  layer of a transformer model to maximize training speed and minimize memory
  while maintaining accuracy.

tasks:
  - id: precision_assignment
    description: "Assign precision (FP32/FP16/FP8) to each layer of a model"
    observation_space:
      type: dict
      properties:
        model_architecture:
          type: array
          items:
            type: dict
        memory_budget_gb:
          type: number
        speed_target:
          type: number
    action_space:
      type: dict
      properties:
        precision_config:
          type: dict
          values: ["FP32", "FP16", "BF16", "FP8"]
    reward_function: "precision_config_grader"

  - id: instability_detection
    description: "Detect precision-induced training instability from loss trajectories"
    observation_space:
      type: dict
      properties:
        training_loss_trajectory:
          type: array
          items:
            type: number
        current_config:
          type: dict
        step_count:
          type: integer
    action_space:
      type: dict
      properties:
        instability_detected:
          type: boolean
        step_detected:
          type: integer
        recommended_fix:
          type: string
    reward_function: "stability_detector_grader"

  - id: multi_objective_optimization
    description: "Optimize memory/time/accuracy trade-offs under constraints"
    observation_space:
      type: dict
      properties:
        constraints:
          type: dict
        model_specs:
          type: dict
    action_space:
      type: dict
      properties:
        precision_strategy:
          type: dict
        predicted_metrics:
          type: dict
    reward_function: "pareto_optimizer_grader"

graders:
  - id: precision_config_grader
    type: lookup_table + stability_check
    implementation: graders/precision_grader.py

  - id: stability_detector_grader
    type: trajectory_analysis
    implementation: graders/stability_grader.py

  - id: pareto_optimizer_grader
    type: multi_objective_scoring
    implementation: graders/pareto_grader.py

episodes:
  - id: training_episode
    num_steps: 3
    scenarios: 25
    scenario_sampling: stratified_by_model_size

baseline_agent: agents/baseline_agent.py
```

---

# 7. Baseline Agent

```python
# agents/baseline_agent.py
import numpy as np

class MixedPrecisionBaseline:
    """Rule-based baseline agent for Mixed Precision environment.
    
    Uses simple heuristics based on layer type to assign precision.
    No ML required — demonstrates the environment is solvable.
    """
    
    def __init__(self):
        self.layer_rules = {
            "embedding":  "FP32",   # Always needs full precision (tiny gradients)
            "attention":  "FP16",   # Normalized activations, safe for FP16
            "ffn":        "FP8",    # Robust to quantization after LayerNorm
            "output":     "FP32",   # Critical for loss computation
            "layernorm":  "FP16",   # Stabilizes values, safe for FP16
        }
    
    def task_1_precision_assignment(self, model_architecture):
        config = {}
        for layer in model_architecture:
            layer_type = layer["type"]
            config[layer["name"]] = self.layer_rules.get(layer_type, "FP16")
        return {"precision_config": config}
    
    def task_2_instability_detection(self, loss_trajectory, current_config):
        losses = np.array(loss_trajectory)
        
        # Check for NaN
        if np.any(np.isnan(losses)):
            nan_step = int(np.argmax(np.isnan(losses)))
            return {
                "instability_detected": True,
                "step_detected": nan_step,
                "likely_cause": "Precision loss accumulation in low-precision layers",
                "recommendation": "Increase precision on FFN layers from FP8 to FP16",
                "confidence": 0.8
            }
        
        # Check for loss explosion
        if len(losses) > 10 and losses[-1] > losses[0] * 100:
            return {
                "instability_detected": True,
                "step_detected": len(losses) - 1,
                "likely_cause": "Gradient explosion from precision overflow",
                "recommendation": "Use FP32 for gradient accumulation",
                "confidence": 0.6
            }
        
        return {
            "instability_detected": False,
            "step_detected": len(losses),
            "likely_cause": "None",
            "recommendation": "No action needed",
            "confidence": 0.9
        }
    
    def task_3_multi_objective(self, constraints, model_specs):
        mem_budget = constraints["memory_budget_gb"]
        baseline_mem = model_specs.get("baseline_memory_gb", 100)
        
        if mem_budget < baseline_mem * 0.3:
            # Aggressive: need maximum compression
            strategy = {
                "embedding": "FP32", "attention": "FP16",
                "ffn": "FP8", "output": "FP32"
            }
            speedup, mem_saved, accuracy = 2.8, 60, 0.995
        else:
            # Conservative: prioritize stability
            strategy = {
                "embedding": "FP32", "attention": "FP16",
                "ffn": "FP16", "output": "FP32"
            }
            speedup, mem_saved, accuracy = 1.8, 35, 0.999
        
        return {
            "precision_strategy": strategy,
            "predicted_metrics": {
                "speedup": speedup,
                "memory_saved_percent": mem_saved,
                "accuracy": accuracy
            }
        }
```

---

# 8. Project Structure

```
MixedPrecisionEnv/
│
├── README.md                          # Project overview, setup, usage
├── openenv.yaml                       # OpenEnv manifest (required)
├── Dockerfile                         # Container definition (required)
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
│
├── app/                               # Main application (FastAPI server)
│   ├── __init__.py
│   ├── main.py                        # FastAPI app, all endpoints
│   ├── config.py                      # App configuration
│   └── routes/
│       ├── __init__.py
│       ├── tasks.py                   # GET /tasks endpoint
│       ├── baseline.py                # POST /baseline endpoint
│       ├── grader.py                  # POST /grader endpoint
│       └── environment.py            # POST /reset, /step endpoints
│
├── environment/                       # Core RL environment
│   ├── __init__.py
│   ├── mixed_precision_env.py         # Main Gymnasium-compatible env class
│   ├── state.py                       # Environment state management
│   └── utils.py                       # Helper functions
│
├── scenarios/                         # Pre-built training scenarios
│   ├── __init__.py
│   ├── scenario_loader.py            # Loads & validates scenarios
│   ├── small_models.json              # 5 small model scenarios
│   ├── medium_models.json             # 5 medium model scenarios
│   ├── large_models.json              # 5 large model scenarios
│   ├── instability_scenarios.json     # 5 instability detection scenarios
│   └── constraint_scenarios.json      # 5 multi-objective scenarios
│
├── graders/                           # Deterministic grading functions
│   ├── __init__.py
│   ├── precision_grader.py            # Task 1: precision config scoring
│   ├── stability_grader.py            # Task 2: instability detection scoring
│   ├── pareto_grader.py               # Task 3: multi-objective scoring
│   └── lookup_tables.py              # Pre-computed optimal configs
│
├── agents/                            # Agent implementations
│   ├── __init__.py
│   └── baseline_agent.py             # Rule-based baseline agent
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_environment.py            # Environment reset/step tests
│   ├── test_graders.py                # Grader determinism tests
│   ├── test_baseline.py               # Baseline agent tests
│   ├── test_scenarios.py              # Scenario validation tests
│   └── test_endpoints.py             # API endpoint tests
│
├── scripts/                           # Utility scripts
│   ├── validate_submission.py         # Pre-submission validation
│   ├── run_baseline.py                # Run baseline agent end-to-end
│   └── generate_scenarios.py          # Scenario generation helper
│
└── docs/                              # Additional documentation
    ├── ARCHITECTURE.md                # Technical architecture overview
    ├── SCENARIOS.md                   # Detailed scenario documentation
    └── API.md                         # API endpoint documentation
```

### Key Files Explained

| File | Purpose | Priority |
|------|---------|----------|
| `openenv.yaml` | OpenEnv manifest — defines tasks, graders, episodes | 🔴 Critical |
| `Dockerfile` | Container build — judges must be able to `docker build` | 🔴 Critical |
| `app/main.py` | FastAPI server with all required endpoints | 🔴 Critical |
| `environment/mixed_precision_env.py` | Core env with `reset()`, `step()` | 🔴 Critical |
| `graders/*.py` | All 3 graders — must be deterministic | 🔴 Critical |
| `scenarios/*.json` | 25 pre-built scenarios | 🔴 Critical |
| `agents/baseline_agent.py` | Baseline that scores > 0 | 🟡 High |
| `tests/*.py` | Validation tests | 🟡 High |
| `README.md` | Documentation for judges | 🟡 High |
| `scripts/validate_submission.py` | Pre-submit validation | 🟢 Medium |

---

# 9. Pre-Submission Checklist

### Must Pass (Disqualification if not)
- [ ] HuggingFace Space deploys without error
- [ ] `openenv validate` passes all checks
- [ ] `docker build .` succeeds
- [ ] `GET /tasks` returns 200 with all 3 task IDs
- [ ] `POST /baseline` returns 200 with valid JSON (scores > 0)
- [ ] `POST /grader` scores all 3 tasks with deterministic outputs
- [ ] Same grader input → same output (run 3× to verify)

### Should Pass (Scoring impact)
- [ ] 25 scenarios load without errors
- [ ] Baseline scores > 30 on at least 1 task
- [ ] All edge cases handled (empty input, malformed config)
- [ ] README documents setup, usage, and architecture
- [ ] Code is clean, commented, and follows PEP 8

### Nice to Have (Creativity bonus)
- [ ] Comprehensive test suite passes
- [ ] Architecture documentation
- [ ] Example agent interactions documented
- [ ] Loss curve visualizations in README

---

# 10. Scoring Rubric

| Category | Weight | What Judges Look For |
|----------|--------|---------------------|
| **Scenario Realism** | 25% | Are the 25 scenarios based on real model architectures? Do layer properties match real-world values? |
| **Grader Determinism** | 25% | Same input → same output, always. No randomness. No LLM calls. |
| **Baseline Quality** | 20% | Does baseline demonstrate the environment works? Score > 50 on at least 1 task? |
| **Technical Depth** | 15% | Is the FP32/FP16/FP8 logic technically sound? Would a Meta engineer trust this? |
| **Deployment Readiness** | 15% | Clean code, documentation, Docker builds, HF Space runs |

---

# 11. 5-Day Build Timeline

| Day | Date | Focus | Hours | Deliverable |
|-----|------|-------|-------|-------------|
| **1** | Apr 2 (Wed) | Project scaffold + research + 25 scenarios | 5h | `openenv.yaml`, all scenario JSONs, project structure created |
| **2** | Apr 3 (Thu) | All 3 graders + lookup tables | 4h | `graders/*.py` complete, all deterministic tests pass |
| **3** | Apr 4 (Fri) | Environment class + baseline agent | 4h | `reset()`, `step()` working, baseline scores > 0 |
| **4** | Apr 5 (Sat) | FastAPI endpoints + Docker + HF deployment | 4h | All endpoints return 200, Docker builds, HF Space live |
| **5** | Apr 6 (Sun) | Testing, edge cases, README polish | 3h | Full test suite passes, README complete |
| **Submit** | Apr 7 (Mon) | Final validation + submit | 1h | `openenv validate` passes, submission done |

**Total estimated effort: ~21 hours across 6 days**

---

**Document Version:** v2.0  
**Created:** April 2, 2026  
**For:** Scalar × Meta PyTorch OpenEnv Hackathon — Solo Submission  
**Deadline:** April 8, 2026 (submitting April 7)
