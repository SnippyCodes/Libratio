# Mixed Precision Training Environment

## Description and Motivation
The Mixed Precision Training Environment is a specialized reinforcement learning simulation designed for the Meta x Scalar Hackathon using the OpenEnv spec. 

**Motivation:** Training large language models (like Meta's LLaMA 3) at scale is incredibly resource-intensive. Machine Learning Engineers manually configure precision formats (FP32, FP16, BF16, FP8) for different neural network layers to speed up training and reduce memory overhead. However, assigning low precision to the wrong layer (e.g., embedding or normalization) causes catastrophic underflow, resulting in NaN loss and millions of dollars in wasted compute. 

This environment simulates this exact, high-stakes infrastructure optimization task, providing an agentic testing ground for AI to learn safe and Pareto-optimal mixed-precision configurations.

---

## Action and Observation Spaces

The environment utilizes heavily-typed Pydantic schemas (viewable in `app/models.py`) to parse agent outputs directly from text.

### Observation Space
Observations include structured contextual data based on the task:
- **Task 1:** `model_architecture` (list of layer dicts detailing `gradient_sensitivity` and `activation_range`).
- **Task 2:** `training_loss_trajectory` (a time-series array of loss values, including possible `NaN` entries) and the `current_config`.
- **Task 3:** `constraints` (hard limits on `memory_budget_gb` and `time_budget_days`) and base `model_specs`.

### Action Space
Actions represent the engineering decisions made by the agent:
- **Task 1:** `precision_config` (Key-value pairs mapping layer names to their assigned precision format: `FP32`, `FP16`, `BF16`, or `FP8`).
- **Task 2:** `instability_detected` (boolean), `step_detected` (integer), and `likely_cause` (string classification).
- **Task 3:** `precision_strategy` (proposed tradeoff) and `predicted_metrics` (numeric efficiency predictions).

---

## Tasks & Expected Difficulty

### Task 1: Static Precision Assignment
- **Difficulty:** **Easy**
- **Description:** The agent is given a model architecture definition. It must assign a precision format to each layer type. The agent must realize that `Embedding` and `Output` layers require FP32 for stability, `LayerNorm/Attention` can use BF16, while `FFN` can handle FP8.
- **Grader:** Assigns normalized scores (0.0 - 1.0) based on stability. Any catastrophic failure assignment results in `0.0`. Valid assignments are scored by efficiency.

### Task 2: Instability Detection (RCA)
- **Difficulty:** **Medium**
- **Description:** The agent is presented with a loss trajectory arrays containing normal descents or sudden divergences/NaNs. The agent must diagnose if the run crashed, exactly which step it crashed on, and infer the root precision cause.
- **Grader:** Continuous scoring (0.0 - 1.0). Heavy penalties for false positives/negatives. Generous bonuses for detecting the failure early.

### Task 3: Multi-Objective Optimization
- **Difficulty:** **Hard**
- **Description:** The agent is given hard limits on Time, Memory, and Accuracy thresholds. It must design a precision strategy that satisfies all three limits simultaneously.
- **Grader:** Continuous scoring (0.0 - 1.0). Fails hard constraints = 0.0. Successes are graded on distance to the mathematical Pareto frontier.

---

## Setup and Usage Instructions

### Docker Execution (Hugging Face Spaces Ready)
1. Build the image:
```bash
docker build -t mixed-precision-env .
```
2. Run the container:
```bash
docker run -p 8000:8000 mixed-precision-env
```
3. The OpenEnv API is now live at `http://127.0.0.1:8000`.

### Local Execution
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the server:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## Baseline Scores
The environment includes a built-in OpenEnv baseline script (`inference.py`) evaluating standard agent capabilities dynamically.

**Baseline Metrics (gpt-4-turbo proxy):**
- **Task 1 Score:** `1.0` (Perfect stable assignment)
- **Task 2 Score:** `1.0` (Correctly parsed and flagged NaN trajectories)
- **Task 3 Score:** `1.0` (Achieved >95% proximity to Pareto optimal frontier)

*Verified reproducible via automated validation tests.*
