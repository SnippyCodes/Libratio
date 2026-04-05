"""
FastAPI application for the Mixed Precision Training OpenEnv.
Implements the required OpenEnv endpoints: /reset, /step, /state, /tasks
Uses typed Pydantic models for OpenEnv spec compliance.
"""
from fastapi import FastAPI
from typing import List

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.mixed_precision_env import MixedPrecisionEnvironment
from server.models import (
    ResetRequest, ResetResponse, StepResponse, StateResponse,
    TaskDefinition, RewardPayload,
)

app = FastAPI(
    title="Mixed Precision Training Environment",
    description="OpenEnv for optimizing neural network training precision configurations",
    version="2.0.0"
)

env = MixedPrecisionEnvironment()


@app.get("/")
def root():
    return {"status": "ok", "environment": "Mixed Precision Training"}


@app.get("/tasks", response_model=List[TaskDefinition])
def get_tasks() -> List[TaskDefinition]:
    return [TaskDefinition(**t) for t in env.TASK_DEFS]


@app.post("/reset", response_model=ResetResponse)
def reset_environment(payload: dict = {}):
    task_id = payload.get("task_id", "precision_assignment")
    obs = env.reset(task_id)
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step_environment(payload: dict = {}):
    action = payload.get("action", payload)
    result = env.step(action)
    return StepResponse(
        observation=result["observation"],
        reward=RewardPayload(
            score=result["reward"]["score"],
            feedback=result["reward"]["feedback"],
        ),
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/state", response_model=StateResponse)
def get_state():
    s = env.state()
    return StateResponse(**s)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
