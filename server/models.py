"""
Pydantic models for the Mixed Precision Training OpenEnv.
Typed Observation, Action, and Reward schemas for OpenEnv spec compliance.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any

PrecisionLevel = Literal["FP32", "BF16", "FP16", "FP8"]


# ──────────────────────────────────────────────
# Layer specification (shared across tasks)
# ──────────────────────────────────────────────
class LayerInfo(BaseModel):
    name: str
    layer_type: Literal["embedding", "attention", "ffn", "layernorm", "output"]
    num_params: int
    gradient_sensitivity: str
    activation_range: str


# ──────────────────────────────────────────────
# Task 1: Layer-by-Layer Precision Assignment
# Agent assigns precision to ONE layer per step
# ──────────────────────────────────────────────
class Task1Observation(BaseModel):
    task_id: Literal["precision_assignment"] = "precision_assignment"
    scenario_id: str
    model_name: str
    total_layers: int
    current_layer_index: int
    current_layer: LayerInfo
    assigned_so_far: Dict[str, str]
    available_precisions: List[str] = ["FP32", "BF16", "FP16", "FP8"]
    memory_budget_gb: float
    memory_used_gb: float
    speed_target_speedup: float


class Task1Action(BaseModel):
    precision: PrecisionLevel
    reasoning: str = ""


# ──────────────────────────────────────────────
# Task 2: Progressive Instability Detection
# Agent sees trajectory windows, decides continue or flag
# ──────────────────────────────────────────────
class Task2Observation(BaseModel):
    task_id: Literal["instability_detection"] = "instability_detection"
    scenario_id: str
    precision_config: Dict[str, str]
    total_training_steps: int
    steps_revealed_so_far: int
    loss_trajectory_window: List[Optional[float]]
    window_index: int
    windows_remaining: int


class Task2Action(BaseModel):
    action_type: Literal["continue_monitoring", "flag_instability"]
    analysis: str = ""
    flagged_step: Optional[int] = None
    root_cause: Optional[str] = None


# ──────────────────────────────────────────────
# Task 3: Iterative Multi-Objective Optimization
# Agent proposes strategy, gets feedback, iterates
# ──────────────────────────────────────────────
class Task3Observation(BaseModel):
    task_id: Literal["multi_objective_optimization"] = "multi_objective_optimization"
    scenario_id: str
    constraints: Dict[str, float]
    model_total_params: int
    layer_types: List[str]
    iterations_remaining: int
    best_score_so_far: float
    previous_result: Optional[Dict[str, Any]] = None
    previous_feedback: Optional[str] = None


class Task3Action(BaseModel):
    precision_strategy: Dict[str, PrecisionLevel]
    reasoning: str = ""


# ──────────────────────────────────────────────
# Task 4: Precision Transfer
# Agent adapts a working config to a new architecture
# ──────────────────────────────────────────────
class SourceModelInfo(BaseModel):
    name: str
    total_params: int
    layer_distribution: Dict[str, float]
    working_config: Dict[str, str]
    metrics: Dict[str, float]


class TargetModelInfo(BaseModel):
    name: str
    total_params: int
    layer_distribution: Dict[str, float]
    constraints: Dict[str, float]


class Task4Observation(BaseModel):
    task_id: Literal["precision_transfer"] = "precision_transfer"
    scenario_id: str
    source_model: SourceModelInfo
    target_model: TargetModelInfo
    iterations_remaining: int
    best_score_so_far: float = 0.0
    previous_result: Optional[Dict[str, Any]] = None
    previous_feedback: Optional[str] = None


class Task4Action(BaseModel):
    precision_strategy: Dict[str, PrecisionLevel]
    reasoning: str = ""


# ──────────────────────────────────────────────
# Universal Response Models (OpenEnv Spec)
# ──────────────────────────────────────────────
class RewardPayload(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0, description="Normalized score strictly in (0, 1)")
    feedback: str = Field(..., description="Human-readable grading explanation")


class StepResponse(BaseModel):
    observation: Optional[Dict[str, Any]] = None
    reward: RewardPayload
    done: bool
    info: Dict[str, Any] = {}


class ResetRequest(BaseModel):
    task_id: str = "precision_assignment"


class ResetResponse(BaseModel):
    observation: Dict[str, Any]


class StateResponse(BaseModel):
    task_id: Optional[str] = None
    current_task: Optional[str] = None
    current_scenario_id: Optional[str] = None
    step_number: int = 0
    is_done: bool = True
    history: List[Dict[str, Any]] = []


class TaskDefinition(BaseModel):
    id: str
    task_id: str
    description: str
    difficulty: str
    max_steps: int

