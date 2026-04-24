# Libratio Fleet - Agent Instructions

## Quick Start

```bash
# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run fleet inference (requires HF_TOKEN)
python fleet_inference.py

# Run training (direct mode, no server needed)
python train_fleet.py --task fleet_precision --episodes 100
```

## Project Structure

```
libratio-fleet/
├── environment/           # Core RL environment
│   ├── fleet_env.py       # Multi-agent fleet (main)
│   ├── physics_model.py  # Reward constants
│   └── mixed_precision_env.py  # Legacy single-agent
├── scenarios/             # Task scenarios
│   ├── fleet_scenarios.py
│   └── task*_scenarios.py
├── server/                # FastAPI API
│   ├── app.py
│   └── models.py
├── scripts/               # Helper scripts
│   ├── inference.py       # Legacy
│   └── manual_play.py
├── notebooks/            # Colab notebooks
├── results/               # Training outputs
├── fleet_inference.py    # Multi-agent LLM inference
├── train_fleet.py         # Training script
├── test_fleet.py          # Smoke tests
├── openenv.yaml           # OpenEnv spec
└── AGENTS.md
```

## Running Tasks

```bash
# Run specific task
TASK_ID=fleet_precision python fleet_inference.py
TASK_ID=fleet_oversight python fleet_inference.py
TASK_ID=fleet_resource python fleet_inference.py
TASK_ID=fleet_recovery python fleet_inference.py
```

## Environment Variables

- `HF_TOKEN` - Required for LLM inference (Groq or OpenAI)
- `API_BASE_URL` - LLM API endpoint (default: Groq)
- `MODEL_NAME` - Model to use (default: llama-3.3-70b-versatile)
- `ENV_URL` - Fleet env URL (default: http://localhost:7860)

## Fleet Tasks

| Task | Description | Key Action Fields |
|------|-------------|-------------------|
| `fleet_precision` | Assign precision to models under shared memory | `precision_strategy` |
| `fleet_oversight` | Monitor all runs, detect crashes | `action_type`, `flagged_model` |
| `fleet_resource` | Allocate GPUs across competing models | `allocations` |
| `fleet_recovery` | Diagnose crash, reallocate, verify | Phase-dependent |

## Known Quirks

- Scores clamped to (0.01, 0.99) - never exactly 0 or 1
- Oversight task: false alarm = 0.10, missed crash = 0.10
- Resource task: invalid allocation = 0.01 immediately
- Fleet precision: memory overflow triggers -0.3 penalty
