"""
FastAPI application for the Mixed Precision Training OpenEnv.
Implements the required OpenEnv endpoints: /reset, /step, /state, /tasks
"""
from fastapi import FastAPI
from typing import Dict, List, Any

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.mixed_precision_env import MixedPrecisionEnvironment

app = FastAPI(
    title="Mixed Precision Training Environment",
    description="OpenEnv for optimizing neural network training precision configurations",
    version="2.0.0"
)

env = MixedPrecisionEnvironment()


@app.get("/")
def root():
    return {"status": "ok", "environment": "Mixed Precision Training"}


@app.get("/tasks")
def get_tasks() -> List[Dict[str, Any]]:
    return env.TASK_DEFS


@app.post("/reset")
def reset_environment(payload: dict = {"task_id": "precision_assignment"}):
    task_id = payload.get("task_id", "precision_assignment")
    obs = env.reset(task_id)
    return {"observation": obs}


@app.post("/step")
def step_environment(payload: dict = {}):
    action = payload.get("action", payload)
    result = env.step(action)
    return result


@app.post("/state")
def get_state():
    return env.state()
