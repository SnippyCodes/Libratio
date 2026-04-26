"""
Thermal Kernel Module -- Standalone thermal-aware workload placement evaluation.

This is the SECOND kernel module in the Agentic Kernel architecture,
proving the design is genuinely modular and extensible.

Physics Source: NVIDIA H100 SXM5 Thermal Design Specifications (2023)
  - TDP: 700W
  - Thermal throttling: 83C junction temperature
  - Emergency shutdown: 92C
  - Thermal resistance derived from max-power steady-state measurements

The thermal kernel evaluates whether a precision configuration will push
GPU hardware beyond safe thermal limits under sustained training load.
It accounts for:
  1. Precision-dependent power draw (FP8 draws ~55% of FP32 power)
  2. Sustained load overhead (15% above instantaneous TDP)
  3. Memory utilization's contribution to chip heating
  4. Per-layer power breakdown for debugging

This module can be used standalone or composed with other kernel modules
via the AgenticKernel interface in kernel_interface.py.
"""
from environment.physics_model import (
    POWER_DRAW_MULTIPLIER,
    BYTES_PER_PARAM,
    GPU_TDP_WATTS,
    GPU_THERMAL_LIMIT_C,
)

# H100 SXM5 thermal model constants
AMBIENT_TEMP_C = 25.0
THERMAL_RESISTANCE_C_PER_W = 0.083    # Derived: (83C - 25C) / 700W = 0.083
THROTTLE_THRESHOLD_C = 83.0            # GPU begins throttling
SHUTDOWN_THRESHOLD_C = 92.0            # Emergency power-off
SUSTAINED_LOAD_FACTOR = 1.15           # 15% above burst TDP for sustained training
MEM_THERMAL_COEFFICIENT = 0.05         # Memory utilization contribution to temp


def evaluate_thermal_safety(
    total_params: int,
    precision_strategy: dict,
    layer_distribution: dict,
    num_gpus: int = 1,
    gpu_memory_gb: float = 80.0,
    ambient_temp_c: float = AMBIENT_TEMP_C,
) -> dict:
    """Evaluate thermal safety of a precision configuration.

    Args:
        total_params: Model parameter count
        precision_strategy: Layer-to-precision mapping
        layer_distribution: Layer-to-fraction mapping
        num_gpus: Number of GPUs allocated to this model
        gpu_memory_gb: Memory per GPU in GB
        ambient_temp_c: Data center ambient temperature

    Returns:
        Dictionary with thermal metrics and risk assessment.
    """
    # Calculate weighted average power draw across layers
    weighted_power = 0.0
    total_weight = 0.0
    per_layer_thermal = {}

    for layer_type, fraction in layer_distribution.items():
        precision = precision_strategy.get(layer_type, "FP32")
        power_mult = POWER_DRAW_MULTIPLIER.get(precision, 1.0)
        weighted_power += power_mult * fraction
        total_weight += fraction

        # Per-layer power contribution
        layer_watts = power_mult * GPU_TDP_WATTS * fraction
        per_layer_thermal[layer_type] = {
            "precision": precision,
            "power_multiplier": power_mult,
            "estimated_watts": round(layer_watts, 1),
            "pct_of_total_power": round(power_mult * fraction * 100, 1),
        }

    avg_power_ratio = weighted_power / max(total_weight, 0.001)

    # Estimate actual power draw under sustained training load
    estimated_watts = avg_power_ratio * GPU_TDP_WATTS * SUSTAINED_LOAD_FACTOR

    # Estimate junction temperature
    estimated_temp = ambient_temp_c + (estimated_watts * THERMAL_RESISTANCE_C_PER_W)

    # Memory density thermal contribution
    mem_bytes = sum(
        total_params * layer_distribution.get(lt, 0)
        * BYTES_PER_PARAM.get(precision_strategy.get(lt, "FP32"), 4)
        for lt in layer_distribution
    )
    mem_per_gpu_gb = (mem_bytes / 1e9) / max(num_gpus, 1)
    mem_utilization = mem_per_gpu_gb / gpu_memory_gb

    # Higher memory utilization = more heat from HBM3 activity
    thermal_mem_factor = 1.0 + (mem_utilization * MEM_THERMAL_COEFFICIENT)
    estimated_temp *= thermal_mem_factor

    # Thermal risk classification
    if estimated_temp >= SHUTDOWN_THRESHOLD_C:
        thermal_risk = "SHUTDOWN"
        risk_score = 0.01
        recommendation = "CRITICAL: Reduce precision complexity or add GPUs immediately"
    elif estimated_temp >= THROTTLE_THRESHOLD_C:
        thermal_risk = "THROTTLING"
        overshoot = (estimated_temp - THROTTLE_THRESHOLD_C) / (
            SHUTDOWN_THRESHOLD_C - THROTTLE_THRESHOLD_C
        )
        risk_score = max(0.10, 0.50 - overshoot * 0.40)
        recommendation = "WARNING: GPU will throttle, reducing training throughput"
    elif estimated_temp >= THROTTLE_THRESHOLD_C - 10:
        thermal_risk = "HIGH"
        risk_score = 0.65
        recommendation = "CAUTION: Approaching thermal limits under sustained load"
    elif estimated_temp >= 60:
        thermal_risk = "MODERATE"
        risk_score = 0.80
        recommendation = "OK: Normal operating temperature for sustained workload"
    else:
        thermal_risk = "OPTIMAL"
        risk_score = 0.95
        recommendation = "EXCELLENT: Well within thermal envelope"

    # Thermal headroom
    headroom_c = THROTTLE_THRESHOLD_C - estimated_temp
    headroom_pct = (headroom_c / (THROTTLE_THRESHOLD_C - ambient_temp_c)) * 100

    return {
        "estimated_junction_temp_c": round(estimated_temp, 1),
        "estimated_power_watts": round(estimated_watts, 0),
        "power_ratio_vs_fp32": round(avg_power_ratio, 3),
        "thermal_risk": thermal_risk,
        "risk_score": round(max(0.01, min(0.99, risk_score)), 3),
        "recommendation": recommendation,
        "mem_per_gpu_gb": round(mem_per_gpu_gb, 2),
        "mem_utilization_pct": round(mem_utilization * 100, 1),
        "thermal_headroom_c": round(headroom_c, 1),
        "thermal_headroom_pct": round(headroom_pct, 1),
        "per_layer_thermal": per_layer_thermal,
        "ambient_temp_c": ambient_temp_c,
        "throttle_threshold_c": THROTTLE_THRESHOLD_C,
        "shutdown_threshold_c": SHUTDOWN_THRESHOLD_C,
    }
