#!/usr/bin/env python
"""Pre-launch reward-scale and step-0 reproducibility audits for the canary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/reece_dev/msc-audio-pianist")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ala_pianist.evaluation import binary_key_vector, pressed_key_metrics, timestep_key_metrics  # noqa: E402
from ala_pianist.music import assign_right_hand_fingering, sequence_timing_from_profile, write_sequence_midi  # noqa: E402
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv  # noqa: E402
from train_general_one_hand_policy import reward_config_from_profile  # noqa: E402


FROZEN = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1"
CANARY = FROZEN / "refinement_canary"
CHECKPOINT = FROZEN / "checkpoint_800000_steps.pt"
ARMS = {
    "control_continue_sensitive_v1": "transition_cleanup_sensitive_v1",
    "release_completion_v2": "transition_cleanup_release_completion_v2",
    "release_completion_motion_v2": "transition_cleanup_release_completion_motion_v2",
}
SEQUENCES = {
    "single_73_success_anchor": [73],
    "trained_73_72_release_failure": [73, 72],
    "trained_74_75_second_missed": [74, 75],
    "trained_75_76_wrong_neighbour": [75, 76],
    "trained_72_73_saturated": [72, 73],
}
ALL_EVAL_SEQUENCES = {
    "single_72": [72],
    "single_73": [73],
    "single_74": [74],
    "single_75": [75],
    "single_76": [76],
    "trained_72_73": [72, 73],
    "trained_73_72": [73, 72],
    "trained_73_74": [73, 74],
    "trained_74_73": [74, 73],
    "trained_74_75": [74, 75],
    "trained_75_74": [75, 74],
    "trained_75_76": [75, 76],
    "trained_76_75": [76, 75],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", type=Path, default=CANARY)
    args = parser.parse_args()
    args.canary_root.mkdir(parents=True, exist_ok=True)
    validation = args.canary_root / "validation"
    validation.mkdir(exist_ok=True)
    reward_rows = reward_scale_audit(validation)
    step0_rows = step0_reproducibility(validation)
    print(
        json.dumps(
            {
                "reward_scale_rows": len(reward_rows),
                "step0_rows": len(step0_rows),
                "reward_scale_audit": str(validation / "reward_scale_audit.md"),
                "step0_reproducibility": str(validation / "step0_reproducibility.md"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def reward_scale_audit(validation: Path) -> list[dict]:
    rows = []
    for arm, reward_profile in ARMS.items():
        if arm == "control_continue_sensitive_v1":
            continue
        for sequence_id, pitches in SEQUENCES.items():
            rollout = run_rollout(reward_profile, sequence_id, pitches, validation / "reward_audit_midi")
            components = rollout["component_series"]
            totals = component_contributions(reward_profile, components)
            transition_steps = np.asarray(components["release_completion_transition_gate"]) > 0.0
            transition_count = int(np.sum(transition_steps))
            transition_denominator = max(1, transition_count)
            transition_count_expected = max(0, len(pitches) - 1)
            completion_count = float(np.sum(components["release_completion_second_target_event"]))
            row = {
                "arm": arm,
                "reward_profile": reward_profile,
                "sequence_id": sequence_id,
                "pitches": json.dumps(pitches),
                "transition_count_expected": transition_count_expected,
                "transition_steps": transition_count,
                "total_shaped_return": rollout["total_reward"],
                "release_penalty_mean": mean(totals["release_penalty"]),
                "release_penalty_min": min_or_zero(totals["release_penalty"]),
                "release_penalty_max": max_or_zero(totals["release_penalty"]),
                "release_penalty_total": float(np.sum(totals["release_penalty"])),
                "release_nonzero_transition_pct": percent_nonzero(totals["release_penalty"], transition_steps, transition_denominator),
                "completion_bonus_count": completion_count,
                "completion_bonus_total": float(np.sum(totals["completion_bonus"])),
                "completion_nonzero_transition_pct": percent_nonzero(totals["completion_bonus"], transition_steps, transition_denominator),
                "action_rate_penalty_mean": mean(totals["action_rate_penalty"]),
                "action_rate_penalty_max": max_or_zero(totals["action_rate_penalty"]),
                "action_rate_penalty_total": float(np.sum(totals["action_rate_penalty"])),
                "action_rate_nonzero_transition_pct": percent_nonzero(totals["action_rate_penalty"], transition_steps, transition_denominator),
                "saturation_penalty_mean": mean(totals["saturation_penalty"]),
                "saturation_penalty_max": max_or_zero(totals["saturation_penalty"]),
                "saturation_penalty_total": float(np.sum(totals["saturation_penalty"])),
                "saturation_nonzero_transition_pct": percent_nonzero(totals["saturation_penalty"], transition_steps, transition_denominator),
                "completion_repeated_exploit": completion_count > transition_count_expected,
                "completion_without_release_observed": completion_count > 0 and float(np.sum(components["release_completion_release_achieved_event"])) == 0.0,
                "release_penalty_after_release_observed": release_penalty_after_release(components, totals),
                "reward_persisted_after_transition_observed": reward_persisted_after_transition(components, totals),
                "max_completion_events_per_transition_ok": completion_count <= transition_count_expected,
                "main_target_total": float(np.sum(np.asarray(components["target_key_state"], dtype=float) * 4.0)),
                "main_unintended_total": float(np.sum(np.asarray(components["unintended_continuous_travel"], dtype=float) * -0.75)),
            }
            for key in ("release_penalty", "completion_bonus", "action_rate_penalty", "saturation_penalty"):
                row[f"{key}_proportion_of_abs_return"] = proportion(row[f"{key}_total"], rollout["total_reward"])
            rows.append(row)
    write_csv(validation / "reward_scale_audit.csv", rows)
    write_reward_audit_md(validation / "reward_scale_audit.md", rows)
    return rows


def step0_reproducibility(validation: Path) -> list[dict]:
    rows = []
    baseline_by_sequence = {}
    for arm, reward_profile in ARMS.items():
        for order, (sequence_id, pitches) in enumerate(ALL_EVAL_SEQUENCES.items()):
            rollout = run_rollout(reward_profile, sequence_id, pitches, validation / "step0_midi")
            actions_hash = array_hash(np.asarray(rollout["actions"], dtype=np.float32))
            fingertips_hash = array_hash(np.asarray(rollout["fingertips"], dtype=np.float32))
            behavior = {
                "pressed_keys": rollout["pressed_keys"],
                "pressed_key_f1": rollout["pressed_key_f1"],
                "timestep_f1": rollout["timestep_f1"],
                "max_target": rollout["max_target"],
                "max_unintended": rollout["max_unintended"],
                "actions_hash": actions_hash,
                "fingertips_hash": fingertips_hash,
            }
            if arm == "control_continue_sensitive_v1":
                baseline_by_sequence[sequence_id] = behavior
            baseline = baseline_by_sequence.get(sequence_id, behavior)
            rows.append(
                {
                    "arm": arm,
                    "reward_profile": reward_profile,
                    "sequence_order": order,
                    "sequence_id": sequence_id,
                    "pitches": json.dumps(pitches),
                    "actions_hash": actions_hash,
                    "fingertips_hash": fingertips_hash,
                    "actions_identical_to_control": actions_hash == baseline["actions_hash"],
                    "trajectory_identical_to_control": fingertips_hash == baseline["fingertips_hash"],
                    "pressed_keys": json.dumps(rollout["pressed_keys"]),
                    "pressed_keys_identical_to_control": rollout["pressed_keys"] == baseline["pressed_keys"],
                    "pressed_key_f1": rollout["pressed_key_f1"],
                    "pressed_key_f1_delta_vs_control": rollout["pressed_key_f1"] - baseline["pressed_key_f1"],
                    "timestep_f1": rollout["timestep_f1"],
                    "timestep_f1_delta_vs_control": rollout["timestep_f1"] - baseline["timestep_f1"],
                    "max_target": rollout["max_target"],
                    "max_unintended": rollout["max_unintended"],
                    "shaped_return": rollout["total_reward"],
                }
            )
    write_csv(validation / "step0_reproducibility.csv", rows)
    write_step0_md(validation / "step0_reproducibility.md", rows)
    return rows


def run_rollout(reward_profile: str, sequence_id: str, pitches: list[int], midi_dir: Path) -> dict:
    policy = DroQPolicy.load(CHECKPOINT, device="cpu")
    midi_path = write_sequence_midi(
        pitches,
        midi_dir / f"{reward_profile}_{sequence_id}.mid",
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=sequence_timing_from_profile("aligned"),
        fingering_fn=assign_right_hand_fingering,
        title=f"prelaunch {reward_profile} {sequence_id}",
    )
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(pitches),
        midi_max=max(pitches),
        seed=23,
        lookahead=1,
        horizon_steps=96,
        action_mode="direct",
        action_repeat=1,
        reward_config=reward_config_from_profile(reward_profile),
    )
    obs, _ = env.reset(seed=23)
    target_vectors = []
    pressed_vectors = []
    pressed_keys = set()
    total = 0.0
    max_target = 0.0
    max_unintended = 0.0
    actions = []
    fingertips = []
    component_series = defaultdict(list)
    for _ in range(env.horizon_steps):
        action, _ = policy.predict(obs, deterministic=True)
        actions.append(np.asarray(action, dtype=np.float32))
        fingertips.append(np.stack(list(env.fingertip_positions().values()), axis=0).astype(np.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        max_target = max(max_target, float(info["target_key_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        pressed_keys.update(info["pressed_keys"])
        target_vectors.append(binary_key_vector(info["target_keys"]))
        pressed_vectors.append(binary_key_vector(info["pressed_keys"]))
        for key, value in info["reward_components"].items():
            component_series[key].append(float(value))
        if terminated or truncated:
            break
    target_keys = {pitch - 21 for pitch in pitches}
    pressed = pressed_key_metrics(target_keys, pressed_keys)
    timestep = timestep_key_metrics(target_vectors, pressed_vectors)
    return {
        "total_reward": total,
        "pressed_keys": sorted(int(key) for key in pressed_keys),
        "pressed_key_f1": pressed.f1,
        "timestep_f1": timestep.f1,
        "max_target": max_target,
        "max_unintended": max_unintended,
        "actions": actions,
        "fingertips": fingertips,
        "component_series": {key: np.asarray(value, dtype=float) for key, value in component_series.items()},
    }


def component_contributions(reward_profile: str, components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cfg = reward_config_from_profile(reward_profile)
    zeros = np.zeros_like(next(iter(components.values())))
    return {
        "release_penalty": -cfg.release_completion_release_weight
        * components.get("release_completion_transition_gate", zeros)
        * components.get("release_completion_release_penalty_state", zeros),
        "completion_bonus": cfg.release_completion_bonus
        * components.get("release_completion_second_target_event", zeros),
        "action_rate_penalty": -cfg.transition_action_rate_weight
        * components.get("transition_action_rate", zeros),
        "saturation_penalty": -cfg.transition_saturation_weight
        * components.get("transition_saturation", zeros),
    }


def release_penalty_after_release(components: dict[str, np.ndarray], totals: dict[str, np.ndarray]) -> bool:
    released = np.cumsum(components.get("release_completion_release_achieved_event", np.asarray([]))) > 0
    if released.size == 0:
        return False
    return bool(np.any(np.abs(totals["release_penalty"][released]) > 1e-9))


def reward_persisted_after_transition(components: dict[str, np.ndarray], totals: dict[str, np.ndarray]) -> bool:
    gate = components.get("release_completion_transition_gate", np.asarray([])) > 0
    if gate.size == 0:
        return False
    combined = np.abs(totals["release_penalty"]) + np.abs(totals["completion_bonus"]) + np.abs(totals["action_rate_penalty"]) + np.abs(totals["saturation_penalty"])
    return bool(np.any((~gate) & (combined > 1e-9)))


def write_reward_audit_md(path: Path, rows: list[dict]) -> None:
    concerns = [
        row
        for row in rows
        if row["completion_repeated_exploit"]
        or row["release_penalty_after_release_observed"]
        or row["reward_persisted_after_transition_observed"]
    ]
    lines = [
        "# Reward Scale Audit",
        "",
        "No training was run. This audit replays deterministic frozen-actor rollouts under the treatment reward profiles.",
        "",
        f"- rows: `{len(rows)}`",
        f"- exploit concerns found: `{len(concerns)}`",
        "",
        "Completion while the previous note is not yet released is allowed by design; release is rewarded/penalized separately.",
        "",
        markdown_table(rows),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_step0_md(path: Path, rows: list[dict]) -> None:
    treatments = [row for row in rows if row["arm"] != "control_continue_sensitive_v1"]
    actions_ok = all(row["actions_identical_to_control"] for row in treatments)
    trajectory_ok = all(row["trajectory_identical_to_control"] for row in treatments)
    behavior_ok = all(row["pressed_keys_identical_to_control"] for row in treatments)
    lines = [
        "# Step-0 Reproducibility",
        "",
        "Step 0 references the immutable frozen 800k actor; no duplicate step-0 checkpoint is created.",
        "",
        f"- actions identical across arms: `{actions_ok}`",
        f"- fingertip trajectories identical across arms: `{trajectory_ok}`",
        f"- pressed-key behaviour identical across arms: `{behavior_ok}`",
        "- shaped returns differ as expected because each arm uses a different reward profile.",
        "",
        markdown_table(rows),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def min_or_zero(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def max_or_zero(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def percent_nonzero(values: np.ndarray, transition_steps: np.ndarray, denominator: int) -> float:
    if values.size == 0 or transition_steps.size == 0:
        return 0.0
    return float(100.0 * np.sum((np.abs(values) > 1e-9) & transition_steps) / denominator)


def proportion(component_total: float, total_reward: float) -> float:
    denom = max(1e-9, abs(float(total_reward)))
    return float(abs(component_total) / denom)


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


if __name__ == "__main__":
    main()

