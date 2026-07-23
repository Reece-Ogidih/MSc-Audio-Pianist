#!/usr/bin/env python
"""Local reward-scale audit for transition_cleanup_sensitive_v1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.evaluation.unintended import classify_unintended_key, unintended_penalty_components
from ala_pianist.music import assign_right_hand_fingering, sequence_timing_from_profile, write_sequence_midi
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv, GeneralRewardConfig
from evaluate_general_one_hand_policy import reward_config_from_profile


ROOT = Path("/home/reece_dev/msc-audio-pianist")
DEFAULT_OUT = ROOT / "experiments" / "cleanliness_reward_audit"
DEFAULT_DROQ = ROOT / "artifacts/hex_runs/droq_stage3c_fair_1m_20260720T160108Z/output/checkpoints/droq_stage3c_fair_hexcloud_resume_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_1000000/checkpoint_1000000_steps.pt"
FALLBACK_DROQ = ROOT / "experiments/general_one_hand/droq/checkpoints/droq_stage3c_fair_1m_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_1000000/checkpoint_300000_steps.pt"
DEFAULT_SAC = ROOT / "experiments/general_one_hand/stage3c_adjacent_74_75_general_one_hand_sac_sequence_cleanup_pitches73-74-75_lookahead1_directx1_transition_cleanup_seed13_500000.zip"


def weighted_components(components: dict[str, float], cfg: GeneralRewardConfig) -> dict[str, float]:
    return {
        "target_reward": cfg.target_travel_weight * components["target_key_state"]
        + cfg.target_activation_bonus * components["target_activation"],
        "transition_cleanup_existing_penalty": -(
            cfg.wrong_travel_weight * components["max_unintended_key_state"]
            + cfg.wrong_pressed_weight * components["wrong_pressed_key_count"]
            + cfg.high_unintended_weight * components["high_unintended"]
            + cfg.gated_unintended_weight * components["cleanup_gate"] * components["max_unintended_key_state"]
            + cfg.gated_wrong_pressed_weight * components["cleanup_gate"] * components["wrong_pressed_key_count"]
            + cfg.nearby_wrong_key_weight * components["cleanup_gate"] * components["nearby_wrong_key_state"]
            + cfg.csharp_dsharp_key54_weight * components["cleanup_gate"] * components["csharp_dsharp_key54_state"]
            + cfg.dsharp_csharp_key52_weight * components["cleanup_gate"] * components["dsharp_csharp_key52_state"]
            + cfg.csharp_dsharp_pressed_weight * components["cleanup_gate"] * components["csharp_dsharp_key54_pressed"]
            + cfg.dsharp_csharp_pressed_weight * components["cleanup_gate"] * components["dsharp_csharp_key52_pressed"]
            + cfg.release_previous_key_weight * components["release_gate"] * components["release_previous_key_state"]
            + cfg.transition_stray_key_weight * components["transition_stray_key_state"]
            + cfg.transition_stray_pressed_weight * components["transition_stray_pressed_count"]
        ),
        "continuous_unintended_travel_penalty": -cfg.unintended_travel_weight
        * components["unintended_continuous_travel"],
        "near_press_penalty": -cfg.unintended_near_press_weight
        * components["unintended_near_press_barrier"],
        "wrong_key_press_penalty": -cfg.unintended_press_weight
        * components["unintended_pressed_event_count"],
        "late_release_penalty": -cfg.late_release_weight * components["late_release_travel"],
        "early_activation_penalty": -cfg.early_activation_weight * components["early_activation_travel"],
        "duration_penalty": -cfg.duration_weight * components["unintended_integrated_duration"],
        "action_smoothness_penalty": -(
            cfg.action_weight * components["action_magnitude"]
            + cfg.smoothness_weight * components["smoothness"]
        ),
        "native_reward_contribution": cfg.native_reward_weight * components["native_reward"],
    }


def make_env(sequence: list[int], out_dir: Path, reward_config: GeneralRewardConfig) -> GeneralOneHandGoalEnv:
    midi_path = write_sequence_midi(
        sequence,
        out_dir / "audit_midi" / f"{'_'.join(map(str, sequence))}.mid",
        midi_min=min(sequence),
        midi_max=max(sequence),
        timing=sequence_timing_from_profile("aligned"),
        fingering_fn=assign_right_hand_fingering,
        title="cleanliness reward audit",
    )
    return GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(sequence),
        midi_max=max(sequence),
        seed=23,
        lookahead=1,
        horizon_steps=96,
        action_mode="direct",
        action_repeat=1,
        reward_config=reward_config,
    )


def load_policy(kind: str, path: Path):
    if kind == "droq":
        return DroQPolicy.load(path, device="cpu")
    if kind == "sac":
        return SAC.load(path)
    raise ValueError(kind)


def action_for_case(case: str, env: GeneralOneHandGoalEnv, obs, policy=None, rng=None):
    if case == "zero_noop":
        return np.zeros(22, dtype=np.float32)
    if case == "random_actions":
        return rng.uniform(-1.0, 1.0, size=22).astype(np.float32)
    if case in {"competent_droq_policy", "competent_sac_policy"}:
        action, _ = policy.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    raise ValueError(case)


def run_case(
    *,
    case: str,
    sequence: list[int],
    out_dir: Path,
    reward_config: GeneralRewardConfig,
    policy=None,
    seed: int = 23,
) -> dict[str, Any]:
    env = make_env(sequence, out_dir, reward_config)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    raw = defaultdict(list)
    weighted = defaultdict(list)
    total = 0.0
    nonfinite = False
    for _ in range(env.horizon_steps):
        action = action_for_case(case, env, obs, policy=policy, rng=rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        components = info["reward_components"]
        for key, value in components.items():
            value = float(value)
            raw[key].append(value)
            nonfinite = nonfinite or not np.isfinite(value)
        for key, value in weighted_components(components, reward_config).items():
            weighted[key].append(float(value))
        nonfinite = nonfinite or not np.isfinite(reward)
        if terminated or truncated:
            break
    summary: dict[str, Any] = {
        "case": case,
        "sequence": json.dumps(sequence),
        "total_return": total,
        "nonfinite": nonfinite,
    }
    for source_name, values in {**raw, **weighted}.items():
        arr = np.asarray(values, dtype=float)
        summary[f"{source_name}_mean"] = float(np.mean(arr))
        summary[f"{source_name}_max"] = float(np.max(arr))
        summary[f"{source_name}_total"] = float(np.sum(arr))
    return summary


def synthetic_case_row(case: str, components: dict[str, float], cfg: GeneralRewardConfig) -> dict[str, Any]:
    weighted = weighted_components(components, cfg)
    total = sum(weighted.values())
    row: dict[str, Any] = {
        "case": case,
        "sequence": json.dumps(["synthetic"]),
        "total_return": float(total),
        "nonfinite": not np.isfinite(total),
    }
    for key, value in {**components, **weighted}.items():
        numeric = float(value)
        row[f"{key}_mean"] = numeric
        row[f"{key}_max"] = numeric
        row[f"{key}_total"] = numeric
        row["nonfinite"] = row["nonfinite"] or not np.isfinite(numeric)
    return row


def base_component_template() -> dict[str, float]:
    return {
        "target_key_state": 0.0,
        "max_unintended_key_state": 0.0,
        "wrong_pressed_key_count": 0.0,
        "action_magnitude": 0.0,
        "smoothness": 0.0,
        "native_reward": 0.0,
        "fingering_score": 0.0,
        "target_activation": 0.0,
        "high_unintended": 0.0,
        "cleanup_gate": 0.0,
        "nearby_wrong_key_state": 0.0,
        "csharp_dsharp_key54_state": 0.0,
        "dsharp_csharp_key52_state": 0.0,
        "csharp_dsharp_key54_pressed": 0.0,
        "dsharp_csharp_key52_pressed": 0.0,
        "release_gate": 0.0,
        "release_previous_key_state": 0.0,
        "transition_stray_key_state": 0.0,
        "transition_stray_pressed_count": 0.0,
        "unintended_continuous_travel": 0.0,
        "unintended_near_press_barrier": 0.0,
        "unintended_pressed_event_count": 0.0,
        "late_release_travel": 0.0,
        "early_activation_travel": 0.0,
        "unintended_integrated_duration": 0.0,
    }


def synthetic_mask_audit(cfg: GeneralRewardConfig) -> dict[str, Any]:
    states = np.zeros(88, dtype=float)
    states[52] = 1.0
    target_only = unintended_penalty_components(
        states,
        current_target_keys={52},
        previous_target_keys={54},
        future_target_keys={53},
        soft_threshold=cfg.unintended_soft_threshold,
        press_threshold=cfg.press_threshold,
    )
    states[51] = 1.0
    wrong_neighbor = unintended_penalty_components(
        states,
        current_target_keys={52},
        previous_target_keys=set(),
        future_target_keys=set(),
        soft_threshold=cfg.unintended_soft_threshold,
        press_threshold=cfg.press_threshold,
    )
    classifications = {
        "current_target": "excluded_from_unintended_components",
        "previous": classify_unintended_key(
            54,
            value=1.0,
            current_target_keys={52},
            previous_target_keys={54},
            future_target_keys=set(),
            press_threshold=cfg.press_threshold,
        ).category,
        "future": classify_unintended_key(
            53,
            value=1.0,
            current_target_keys={52},
            previous_target_keys=set(),
            future_target_keys={53},
            press_threshold=cfg.press_threshold,
        ).category,
        "unrelated": classify_unintended_key(
            60,
            value=1.0,
            current_target_keys={52},
            previous_target_keys=set(),
            future_target_keys=set(),
            press_threshold=cfg.press_threshold,
        ).category,
    }
    wrong_components = base_component_template()
    wrong_components.update(
        {
            "max_unintended_key_state": 1.0,
            "wrong_pressed_key_count": 1.0,
            "high_unintended": max(0.0, 1.0 - cfg.high_unintended_threshold),
            "nearby_wrong_key_state": 1.0,
            "csharp_dsharp_key54_state": 1.0,
            "csharp_dsharp_key54_pressed": 1.0,
            **wrong_neighbor,
        }
    )
    one_key_components = unintended_penalty_components(
        np.eye(1, 88, 60, dtype=float).reshape(88),
        current_target_keys={52},
        previous_target_keys=set(),
        future_target_keys=set(),
        soft_threshold=cfg.unintended_soft_threshold,
        press_threshold=cfg.press_threshold,
    )
    max_one_key_penalty = (
        cfg.unintended_travel_weight * one_key_components["unintended_continuous_travel"]
        + cfg.unintended_near_press_weight * one_key_components["unintended_near_press_barrier"]
        + cfg.unintended_press_weight * one_key_components["unintended_pressed_event_count"]
        + cfg.duration_weight * one_key_components["unintended_integrated_duration"]
    )
    return {
        "target_only_components": target_only,
        "wrong_neighbor_components": wrong_neighbor,
        "synthetic_wrong_neighbor_reward_row": synthetic_case_row(
            "synthetic_deliberately_wrong_neighbour",
            wrong_components,
            cfg,
        ),
        "classifications": classifications,
        "max_possible_added_penalty_for_one_unrelated_key_one_timestep": max_one_key_penalty,
        "one_key_components": one_key_components,
        "overlap_note": "The same wrong key can intentionally contribute to continuous travel, near-press barrier, press event, and duration penalties.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--droq-policy", default=str(DEFAULT_DROQ if DEFAULT_DROQ.exists() else FALLBACK_DROQ))
    parser.add_argument("--sac-policy", default=str(DEFAULT_SAC))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = reward_config_from_profile("transition_cleanup_sensitive_v1")
    droq = load_policy("droq", Path(args.droq_policy)) if Path(args.droq_policy).exists() else None
    sac = load_policy("sac", Path(args.sac_policy)) if Path(args.sac_policy).exists() else None
    rows = [
        run_case(case="zero_noop", sequence=[75], out_dir=out_dir, reward_config=cfg),
        run_case(case="random_actions", sequence=[75], out_dir=out_dir, reward_config=cfg),
    ]
    if droq is not None:
        rows.append(
            run_case(
                case="competent_droq_policy",
                sequence=[75],
                out_dir=out_dir,
                reward_config=cfg,
                policy=droq,
            )
        )
    if sac is not None:
        rows.append(
            run_case(
                case="competent_sac_policy",
                sequence=[75],
                out_dir=out_dir,
                reward_config=cfg,
                policy=sac,
            )
        )
    mask = synthetic_mask_audit(cfg)
    rows.append(mask["synthetic_wrong_neighbor_reward_row"])
    write_csv(out_dir / "reward_scale_audit_cases.csv", rows)
    competent_rows = [
        row for row in rows if row["case"] in {"competent_droq_policy", "competent_sac_policy"}
    ]
    target_beats_noop = None
    if competent_rows:
        target_beats_noop = max(row["total_return"] for row in competent_rows) > next(
            row["total_return"] for row in rows if row["case"] == "zero_noop"
        )
    payload = {
        "cases": rows,
        "mask_audit": mask,
        "reward_config": cfg.__dict__,
        "checks": {
            "target_beats_noop": target_beats_noop,
            "noop_beats_random_wrong_activity": next(row["total_return"] for row in rows if row["case"] == "zero_noop")
            > next(row["total_return"] for row in rows if row["case"] == "random_actions"),
            "noop_beats_synthetic_wrong_neighbour": next(row["total_return"] for row in rows if row["case"] == "zero_noop")
            > next(row["total_return"] for row in rows if row["case"] == "synthetic_deliberately_wrong_neighbour"),
            "all_components_finite": not any(row["nonfinite"] for row in rows),
        },
    }
    (out_dir / "reward_scale_audit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["checks"], indent=2, sort_keys=True))
    print(f"summary_path={out_dir / 'reward_scale_audit_summary.json'}")


if __name__ == "__main__":
    main()
