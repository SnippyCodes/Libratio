"""
Utility helpers to persist training artifacts after every run.

Designed for TRL/Hugging Face trainer logs (`trainer.state.log_history`).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def _as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_series(
    log_history: Sequence[Dict],
    candidate_keys: Sequence[str],
) -> Tuple[List[float], List[float], Optional[str]]:
    steps: List[float] = []
    values: List[float] = []
    key_used: Optional[str] = None

    for entry in log_history:
        if "step" not in entry:
            continue

        step = _as_float(entry.get("step"))
        if step is None:
            continue

        for key in candidate_keys:
            val = _as_float(entry.get(key))
            if val is None:
                continue
            steps.append(step)
            values.append(val)
            key_used = key
            break

    return steps, values, key_used


def _save_plot(
    x: Sequence[float],
    y: Sequence[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    color: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o", linewidth=1.8, markersize=3, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_trl_training_artifacts(
    *,
    log_history: Sequence[Dict],
    output_dir: str = "results/training_logs",
    run_name: str = "run",
) -> Dict[str, str]:
    """
    Persist TRL training artifacts for reproducibility and README evidence.

    Saves:
    - <run_name>_log_history.json
    - <run_name>_log_history.csv
    - <run_name>_loss_curve.png (if loss exists)
    - <run_name>_reward_curve.png (if reward exists)
    - <run_name>_metrics_summary.json
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    base = run_name.strip().replace(" ", "_") or "run"

    log_json_path = out_dir / f"{base}_log_history.json"
    log_csv_path = out_dir / f"{base}_log_history.csv"
    loss_plot_path = out_dir / f"{base}_loss_curve.png"
    reward_plot_path = out_dir / f"{base}_reward_curve.png"
    summary_path = out_dir / f"{base}_metrics_summary.json"

    # 1) Raw JSON history
    with log_json_path.open("w", encoding="utf-8") as f:
        json.dump(list(log_history), f, indent=2, ensure_ascii=True)

    # 2) Flat CSV history
    all_keys = sorted({k for row in log_history for k in row.keys()})
    with log_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in log_history:
            writer.writerow(row)

    # 3) Extract common loss/reward keys from TRL logs
    loss_steps, losses, loss_key = _extract_series(log_history, ["loss", "train/loss"])
    reward_steps, rewards, reward_key = _extract_series(
        log_history,
        [
            "reward",
            "rewards/mean",
            "objective/rlhf_reward",
            "train/reward",
        ],
    )

    if losses:
        _save_plot(
            loss_steps,
            losses,
            title=f"Training Loss ({base})",
            xlabel="Training Step",
            ylabel="Loss",
            output_path=loss_plot_path,
            color="#d62728",
        )

    if rewards:
        _save_plot(
            reward_steps,
            rewards,
            title=f"Training Reward ({base})",
            xlabel="Training Step",
            ylabel="Reward",
            output_path=reward_plot_path,
            color="#2ca02c",
        )

    # 4) Summary metrics
    summary = {
        "run_name": base,
        "timestamp_utc": timestamp,
        "num_log_entries": len(log_history),
        "loss_key_used": loss_key,
        "reward_key_used": reward_key,
        "final_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "final_reward": rewards[-1] if rewards else None,
        "best_reward": max(rewards) if rewards else None,
        "outputs": {
            "log_history_json": str(log_json_path),
            "log_history_csv": str(log_csv_path),
            "loss_curve_png": str(loss_plot_path) if losses else None,
            "reward_curve_png": str(reward_plot_path) if rewards else None,
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    return {
        "log_history_json": str(log_json_path),
        "log_history_csv": str(log_csv_path),
        "loss_curve_png": str(loss_plot_path) if losses else "",
        "reward_curve_png": str(reward_plot_path) if rewards else "",
        "metrics_summary_json": str(summary_path),
    }
