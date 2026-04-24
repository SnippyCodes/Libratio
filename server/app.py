"""
FastAPI application for the Mixed Precision Training OpenEnv.
Implements the required OpenEnv endpoints: /reset, /step, /state, /tasks
Supports both single-agent (original) and multi-agent fleet environments.
"""
from fastapi import FastAPI
from typing import List

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.mixed_precision_env import MixedPrecisionEnvironment
from environment.fleet_env import FleetEnvironment
from server.models import (
    ResetRequest, ResetResponse, StepResponse, StateResponse,
    TaskDefinition, RewardPayload,
)

app = FastAPI(
    title="Libratio Fleet — Multi-Agent GPU Fleet Management",
    description="OpenEnv for multi-agent GPU cluster precision optimization and fleet oversight",
    version="3.0.0"
)

# Single-agent environment (backward compatible)
solo_env = MixedPrecisionEnvironment()

# Multi-agent fleet environment (new)
fleet_env = FleetEnvironment()


def _clamp_score(raw_score) -> float:
    """Defense-in-depth: clamp score to strict (0, 1) open interval."""
    try:
        val = float(raw_score)
    except (ValueError, TypeError):
        val = 0.01
    return float(max(0.001, min(0.999, val)))


@app.get("/")
def root():
    return {
        "status": "ok",
        "environment": "Libratio Fleet — Multi-Agent GPU Fleet Management",
        "version": "3.0.0",
        "modes": {
            "solo": "Original single-agent precision environment (/reset, /step, /state, /tasks)",
            "fleet": "Multi-agent fleet environment (/fleet/reset, /fleet/step, /fleet/state, /fleet/tasks)",
        },
    }


# ═══════════════════════════════════════════════════════════
# SOLO MODE — Original single-agent endpoints (backward compatible)
# ═══════════════════════════════════════════════════════════

@app.get("/tasks", response_model=List[TaskDefinition])
def get_tasks() -> List[TaskDefinition]:
    return [TaskDefinition(**t) for t in solo_env.TASK_DEFS]


@app.post("/reset", response_model=ResetResponse)
def reset_environment(payload: dict = {}):
    task_id = payload.get("task_id", "precision_assignment")
    obs = solo_env.reset(task_id)
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step_environment(payload: dict = {}):
    action = payload.get("action", payload)
    result = solo_env.step(action)
    clamped_score = _clamp_score(result["reward"]["score"])
    return StepResponse(
        observation=result["observation"],
        reward=RewardPayload(
            score=clamped_score,
            feedback=result["reward"]["feedback"],
        ),
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/state", response_model=StateResponse)
def get_state():
    s = solo_env.state()
    return StateResponse(**s)


# ═══════════════════════════════════════════════════════════
# FLEET MODE — Multi-agent fleet endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/fleet/tasks", response_model=List[TaskDefinition])
def get_fleet_tasks() -> List[TaskDefinition]:
    return [TaskDefinition(**t) for t in fleet_env.TASK_DEFS]


@app.post("/fleet/reset", response_model=ResetResponse)
def reset_fleet(payload: dict = {}):
    task_id = payload.get("task_id", "fleet_precision")
    obs = fleet_env.reset(task_id)
    return ResetResponse(observation=obs)


@app.post("/fleet/step", response_model=StepResponse)
def step_fleet(payload: dict = {}):
    action = payload.get("action", payload)
    result = fleet_env.step(action)
    clamped_score = _clamp_score(result["reward"]["score"])
    return StepResponse(
        observation=result["observation"],
        reward=RewardPayload(
            score=clamped_score,
            feedback=result["reward"]["feedback"],
        ),
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/fleet/state", response_model=StateResponse)
def get_fleet_state():
    s = fleet_env.state()
    return StateResponse(**s)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
