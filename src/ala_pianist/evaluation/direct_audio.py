"""Formal evaluation helpers for Pipeline 2 direct raw-audio policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from ala_pianist.evaluation.metrics import binary_key_vector, pressed_key_metrics, timestep_key_metrics
from ala_pianist.pipelines.indirect import BENCHMARK_SEQUENCE_PITCHES
from ala_pianist.rl.direct_audio_droq import DirectDroQAgent, load_direct_droq_checkpoint
from ala_pianist.rl.direct_audio_env import (
    DirectAudioClip,
    DirectAudioGoalEnv,
    build_direct_audio_reference_bank,
)
from ala_pianist.rl.general_one_hand_env import GeneralRewardConfig


PHASE_A_SEQUENCES: tuple[tuple[int, ...], ...] = BENCHMARK_SEQUENCE_PITCHES
PHASE_A_WEIGHTS: tuple[float, ...] = (0.07,) * 5 + (0.08125,) * 8
EVAL_AUDIO_MODES: tuple[str, ...] = ("correct", "zero", "mismatched")
CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)_steps\.pt$")


@dataclass(frozen=True)
class EvaluationClipSelection:
    sequence_to_clip_index: dict[tuple[int, ...], int]
    mismatch_clip_id_by_sequence: dict[tuple[int, ...], int]
    mismatch_sequence_by_sequence: dict[tuple[int, ...], tuple[int, ...]]
    mismatch_mapping: dict[str, str]


def checkpoint_step(path: str | Path) -> int:
    match = CHECKPOINT_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot infer checkpoint step from {path}.")
    return int(match.group(1))


def discover_lightweight_checkpoints(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    paths = sorted(run_dir.glob("**/checkpoint_*_steps.pt"), key=checkpoint_step)
    if not paths:
        raise FileNotFoundError(f"No lightweight checkpoint_*_steps.pt files found under {run_dir}.")
    return paths


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_name(sequence: Iterable[int]) -> str:
    values = tuple(int(pitch) for pitch in sequence)
    if len(values) == 1:
        return f"anchor_{values[0]}"
    return "transition_" + "_".join(str(pitch) for pitch in values)


def parse_sequences(raw: str | None) -> tuple[tuple[int, ...], ...]:
    if not raw:
        return PHASE_A_SEQUENCES
    return tuple(tuple(int(part) for part in chunk.split(",")) for chunk in raw.split(";") if chunk)


def build_pipeline2_evaluation_env(
    *,
    generated_root: str | Path,
    sequences: tuple[tuple[int, ...], ...] = PHASE_A_SEQUENCES,
    sample_rate: int = 16_000,
    variants_per_sequence: int = 1,
    split: str = "canonical_eval",
    seed: int = 13,
    device_horizon_steps: int = 64,
) -> DirectAudioGoalEnv:
    _bank, clips = build_direct_audio_reference_bank(
        generated_root=generated_root,
        sequences=sequences,
        sample_rate=sample_rate,
        variants_per_sequence=variants_per_sequence,
        split=split,
    )
    bank, clips = _bank, tuple(clips)
    return DirectAudioGoalEnv(
        audio_bank=bank,
        clips=clips,
        sequences=sequences,
        sequence_sampling_weights=tuple(1.0 / len(sequences) for _ in sequences),
        reward_config=transition_cleanup_sensitive_reward_config(),
        horizon_steps=device_horizon_steps,
        seed=seed,
        sampling_split=split,
    )


def transition_cleanup_sensitive_reward_config() -> GeneralRewardConfig:
    return GeneralRewardConfig(
        target_travel_weight=4.0,
        wrong_travel_weight=0.25,
        wrong_pressed_weight=0.30,
        action_weight=0.002,
        smoothness_weight=0.001,
        target_activation_bonus=3.0,
        target_activation_threshold=0.9,
        high_unintended_weight=0.5,
        high_unintended_threshold=0.75,
        cleanup_gate_threshold=0.75,
        gated_unintended_weight=2.0,
        gated_wrong_pressed_weight=1.5,
        nearby_wrong_key_weight=1.0,
        release_previous_key_weight=1.0,
        transition_stray_key_weight=1.0,
        transition_stray_pressed_weight=1.0,
        unintended_soft_threshold=0.2,
        unintended_travel_weight=2.0,
        unintended_near_press_weight=1.0,
        unintended_press_weight=1.5,
        late_release_weight=1.0,
        early_activation_weight=0.8,
        duration_weight=0.5,
    )


def build_clip_selection(env: DirectAudioGoalEnv) -> EvaluationClipSelection:
    sequence_to_clip_index: dict[tuple[int, ...], int] = {}
    for clip_index, clip in enumerate(env.clips):
        sequence_to_clip_index.setdefault(tuple(clip.sequence), int(clip_index))
    missing = [seq for seq in env.sequences if tuple(seq) not in sequence_to_clip_index]
    if missing:
        raise ValueError(f"Evaluation audio is missing sequences: {missing}.")

    anchors = [tuple(seq) for seq in env.sequences if len(seq) == 1]
    transitions = [tuple(seq) for seq in env.sequences if len(seq) > 1]
    mismatch_sequence_by_sequence: dict[tuple[int, ...], tuple[int, ...]] = {}
    for group in (anchors, transitions):
        if len(group) < 2:
            raise ValueError("Need at least two anchors and two transitions for deterministic mismatch mapping.")
        for index, sequence in enumerate(group):
            mismatch_sequence_by_sequence[sequence] = group[(index + 1) % len(group)]
    mismatch_clip_id_by_sequence = {
        sequence: int(env.clips[sequence_to_clip_index[mismatch_sequence]].clip_id)
        for sequence, mismatch_sequence in mismatch_sequence_by_sequence.items()
    }
    mismatch_mapping = {
        sequence_name(sequence): sequence_name(mismatch)
        for sequence, mismatch in mismatch_sequence_by_sequence.items()
    }
    return EvaluationClipSelection(
        sequence_to_clip_index=sequence_to_clip_index,
        mismatch_clip_id_by_sequence=mismatch_clip_id_by_sequence,
        mismatch_sequence_by_sequence=mismatch_sequence_by_sequence,
        mismatch_mapping=mismatch_mapping,
    )


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    env: DirectAudioGoalEnv,
    selection: EvaluationClipSelection,
    seed: int,
    device: str = "cpu",
    audio_modes: tuple[str, ...] = EVAL_AUDIO_MODES,
) -> tuple[list[dict], list[dict]]:
    checkpoint_path = Path(checkpoint_path)
    payload = load_direct_droq_checkpoint(checkpoint_path, device=device)
    config = dict(payload["config"])
    validate_checkpoint_against_env(config, env, checkpoint_path=checkpoint_path)
    agent = DirectDroQAgent.load(checkpoint_path, device=device)
    step = checkpoint_step(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    sequence_rows: list[dict] = []
    action_rows: list[dict] = []
    for sequence_index, sequence in enumerate(env.sequences):
        clip_index = selection.sequence_to_clip_index[tuple(sequence)]
        mismatch_sequence = selection.mismatch_sequence_by_sequence[tuple(sequence)]
        mismatch_clip_id = selection.mismatch_clip_id_by_sequence[tuple(sequence)]
        action_rows.append(
            action_dependence_row(
                agent=agent,
                env=env,
                sequence=tuple(sequence),
                clip_index=clip_index,
                mismatch_sequence=mismatch_sequence,
                mismatch_clip_id=mismatch_clip_id,
                checkpoint_step_value=step,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                seed=seed + sequence_index,
            )
        )
        for mode in audio_modes:
            sequence_rows.append(
                rollout_sequence(
                    agent=agent,
                    env=env,
                    sequence=tuple(sequence),
                    clip_index=clip_index,
                    audio_mode=mode,
                    mismatch_sequence=mismatch_sequence,
                    mismatch_clip_id=mismatch_clip_id,
                    checkpoint_step_value=step,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_hash,
                    seed=seed + sequence_index,
                )
            )
    return sequence_rows, action_rows


def validate_checkpoint_against_env(config: dict, env: DirectAudioGoalEnv, *, checkpoint_path: Path) -> None:
    expected = {
        "audio_window_size": int(env.observation_space["audio"].shape[0]),
        "physical_dim": int(env.observation_space["physical"].shape[0]),
        "action_dim": int(env.action_space.shape[0]),
    }
    for key, value in expected.items():
        if int(config.get(key, -1)) != int(value):
            raise ValueError(
                f"Checkpoint {checkpoint_path} has {key}={config.get(key)!r}, "
                f"but evaluation environment expects {value}."
            )


def rollout_sequence(
    *,
    agent: DirectDroQAgent,
    env: DirectAudioGoalEnv,
    sequence: tuple[int, ...],
    clip_index: int,
    audio_mode: str,
    mismatch_sequence: tuple[int, ...],
    mismatch_clip_id: int,
    checkpoint_step_value: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
) -> dict:
    obs, info = env.reset_to_clip_index(clip_index, seed=seed)
    target_key_set = {int(pitch) - 21 for pitch in sequence}
    pressed_keys_seen: set[int] = set()
    target_vectors = []
    pressed_vectors = []
    max_target_state = 0.0
    max_unintended = 0.0
    integrated_unintended = 0.0
    shaped_return = 0.0
    native_reward_sum = 0.0
    action_saturation_values = []
    for _ in range(env.horizon_steps):
        policy_obs = env.observation_for_audio_mode(
            obs,
            mode=audio_mode,
            mismatched_clip_id=mismatch_clip_id,
        )
        action = agent.act(policy_obs, deterministic=True)
        action_saturation_values.append(float(np.mean(np.abs(action) > 0.95)))
        obs, reward, terminated, truncated, info = env.step(action)
        current_targets = tuple(int(key) for key in info.get("target_keys", ()))
        current_pressed = tuple(int(key) for key in info.get("pressed_keys", ()))
        pressed_keys_seen.update(current_pressed)
        target_vectors.append(binary_key_vector(current_targets))
        pressed_vectors.append(binary_key_vector(current_pressed))
        shaped_return += float(reward)
        native_reward_sum += float(info.get("native_reward", 0.0))
        max_target_state = max(max_target_state, float(info.get("target_key_state", 0.0)))
        step_unintended = float(info.get("max_unintended_key_state", 0.0))
        max_unintended = max(max_unintended, step_unintended)
        integrated_unintended += step_unintended * env.control_timestep_seconds
        if terminated or truncated:
            break
    pressed = pressed_key_metrics(target_key_set, pressed_keys_seen)
    timestep = timestep_key_metrics(target_vectors, pressed_vectors)
    wrong_pressed = sorted(key for key in pressed_keys_seen if key not in target_key_set)
    return {
        "checkpoint_step": checkpoint_step_value,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "sequence_name": sequence_name(sequence),
        "sequence": "-".join(str(pitch) for pitch in sequence),
        "sequence_group": "anchor" if len(sequence) == 1 else "transition",
        "audio_mode": audio_mode,
        "audio_split": env.clips[clip_index].split,
        "variant_index": int(env.clips[clip_index].variant_index),
        "mismatched_sequence": "-".join(str(pitch) for pitch in mismatch_sequence) if audio_mode == "mismatched" else "",
        "target_keys": "-".join(str(key) for key in sorted(target_key_set)),
        "pressed_keys": "-".join(str(key) for key in sorted(pressed_keys_seen)),
        "wrong_pressed_keys": "-".join(str(key) for key in wrong_pressed),
        "wrong_press_count": int(len(wrong_pressed)),
        "pressed_key_precision": pressed.precision,
        "pressed_key_recall": pressed.recall,
        "pressed_key_f1": pressed.f1,
        "timestep_precision": timestep.precision,
        "timestep_recall": timestep.recall,
        "timestep_f1": timestep.f1,
        "max_target_key_state": float(max_target_state),
        "max_unintended_key_state": float(max_unintended),
        "integrated_unintended_key_state": float(integrated_unintended),
        "shaped_return": float(shaped_return),
        "native_reward_sum": float(native_reward_sum),
        "mean_action_saturation": float(np.mean(action_saturation_values)) if action_saturation_values else 0.0,
        "strict_outcome": strict_outcome(target_key_set, pressed_keys_seen, max_target_state, max_unintended),
    }


def action_dependence_row(
    *,
    agent: DirectDroQAgent,
    env: DirectAudioGoalEnv,
    sequence: tuple[int, ...],
    clip_index: int,
    mismatch_sequence: tuple[int, ...],
    mismatch_clip_id: int,
    checkpoint_step_value: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    seed: int,
) -> dict:
    obs, _info = env.reset_to_clip_index(clip_index, seed=seed)
    actions = {}
    for mode in EVAL_AUDIO_MODES:
        policy_obs = env.observation_for_audio_mode(obs, mode=mode, mismatched_clip_id=mismatch_clip_id)
        actions[mode] = agent.act(policy_obs, deterministic=True)
    zero_diff = np.abs(actions["correct"] - actions["zero"])
    mismatch_diff = np.abs(actions["correct"] - actions["mismatched"])
    return {
        "checkpoint_step": checkpoint_step_value,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "sequence_name": sequence_name(sequence),
        "sequence": "-".join(str(pitch) for pitch in sequence),
        "sequence_group": "anchor" if len(sequence) == 1 else "transition",
        "mismatched_sequence": "-".join(str(pitch) for pitch in mismatch_sequence),
        "mean_abs_action_diff_correct_zero": float(np.mean(zero_diff)),
        "max_abs_action_diff_correct_zero": float(np.max(zero_diff)),
        "mean_abs_action_diff_correct_mismatched": float(np.mean(mismatch_diff)),
        "max_abs_action_diff_correct_mismatched": float(np.max(mismatch_diff)),
    }


def strict_outcome(
    target_keys: set[int],
    pressed_keys: set[int],
    max_target_state: float,
    max_unintended: float,
) -> str:
    if target_keys and target_keys.issubset(pressed_keys) and not (pressed_keys - target_keys) and max_unintended < 0.25:
        return "clean_low_unintended"
    if target_keys and target_keys.issubset(pressed_keys) and not (pressed_keys - target_keys):
        return "clean_high_unintended"
    if target_keys and target_keys & pressed_keys and pressed_keys - target_keys:
        return "dirty_pressed_wrong_key"
    if max_target_state >= 0.25 and max_unintended < 0.25:
        return "near_clean_partial"
    return "missed"


def aggregate_checkpoint_rows(sequence_rows: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({(int(row["checkpoint_step"]), row["audio_mode"]) for row in sequence_rows})
    for step, mode in keys:
        subset = [row for row in sequence_rows if int(row["checkpoint_step"]) == step and row["audio_mode"] == mode]
        if not subset:
            continue
        anchors = [row for row in subset if row["sequence_group"] == "anchor"]
        transitions = [row for row in subset if row["sequence_group"] == "transition"]
        rows.append(
            {
                "checkpoint_step": step,
                "audio_mode": mode,
                "sequence_count": len(subset),
                "pressed_key_precision_mean": mean(subset, "pressed_key_precision"),
                "pressed_key_recall_mean": mean(subset, "pressed_key_recall"),
                "pressed_key_f1_mean": mean(subset, "pressed_key_f1"),
                "pressed_key_f1_min": min_value(subset, "pressed_key_f1"),
                "timestep_f1_mean": mean(subset, "timestep_f1"),
                "timestep_f1_min": min_value(subset, "timestep_f1"),
                "max_unintended_mean": mean(subset, "max_unintended_key_state"),
                "max_unintended_max": max_value(subset, "max_unintended_key_state"),
                "integrated_unintended_mean": mean(subset, "integrated_unintended_key_state"),
                "wrong_press_count_mean": mean(subset, "wrong_press_count"),
                "wrong_press_count_total": int(sum(int(row["wrong_press_count"]) for row in subset)),
                "anchor_pressed_key_f1_mean": mean(anchors, "pressed_key_f1"),
                "anchor_pressed_key_f1_min": min_value(anchors, "pressed_key_f1"),
                "transition_pressed_key_f1_mean": mean(transitions, "pressed_key_f1"),
                "transition_pressed_key_f1_min": min_value(transitions, "pressed_key_f1"),
                "transition_timestep_f1_mean": mean(transitions, "timestep_f1"),
                "transition_timestep_f1_min": min_value(transitions, "timestep_f1"),
            }
        )
    add_audio_dependence_deltas(rows)
    return rows


def add_audio_dependence_deltas(rows: list[dict]) -> None:
    by_step = {}
    for row in rows:
        by_step.setdefault(int(row["checkpoint_step"]), {})[row["audio_mode"]] = row
    for modes in by_step.values():
        correct = modes.get("correct")
        if not correct:
            continue
        for other_mode in ("zero", "mismatched"):
            other = modes.get(other_mode)
            if not other:
                continue
            suffix = "mismatch" if other_mode == "mismatched" else other_mode
            correct[f"delta_pressed_key_f1_correct_minus_{suffix}"] = (
                float(correct["pressed_key_f1_mean"]) - float(other["pressed_key_f1_mean"])
            )
            correct[f"delta_timestep_f1_correct_minus_{suffix}"] = (
                float(correct["timestep_f1_mean"]) - float(other["timestep_f1_mean"])
            )
            correct[f"delta_transition_f1_correct_minus_{suffix}"] = (
                float(correct["transition_pressed_key_f1_mean"]) - float(other["transition_pressed_key_f1_mean"])
            )


def write_evaluation_outputs(
    *,
    output_dir: str | Path,
    checkpoint_rows: list[dict],
    sequence_rows: list[dict],
    action_rows: list[dict],
    manifest: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_checkpoint_metrics.csv", checkpoint_rows)
    write_csv(output_dir / "evaluation_sequence_metrics.csv", sequence_rows)
    write_csv(output_dir / "audio_dependence.csv", action_rows)
    learning_curve = [row for row in checkpoint_rows if row["audio_mode"] == "correct"]
    write_csv(output_dir / "learning_curve.csv", learning_curve)
    summary = {
        "best_checkpoint_by_correct_pressed_key_f1": best_row(learning_curve, "pressed_key_f1_mean"),
        "best_checkpoint_by_correct_timestep_f1": best_row(learning_curve, "timestep_f1_mean"),
        "checkpoint_count": len({row["checkpoint_step"] for row in sequence_rows}),
        "sequence_count": len({row["sequence_name"] for row in sequence_rows}),
        "audio_modes": sorted({row["audio_mode"] for row in sequence_rows}),
        "manifest_path": str(output_dir / "evaluation_manifest.json"),
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def write_pipeline_comparison(
    *,
    output_dir: str | Path,
    checkpoint_rows: list[dict],
    pipeline1_summary_path: str | Path | None,
) -> Path | None:
    if pipeline1_summary_path is None or not Path(pipeline1_summary_path).exists():
        return None
    payload = json.loads(Path(pipeline1_summary_path).read_text(encoding="utf-8"))
    degradation = payload.get("controller_degradation_summary", {})
    rows = []
    for label, prefix in (
        ("Pipeline1 oracle symbolic", "oracle"),
        ("Pipeline1 raw Basic Pitch", "predicted"),
    ):
        rows.append(
            {
                "system": label,
                "pressed_key_precision": degradation.get(f"{prefix}_pressed_key_precision_mean"),
                "pressed_key_recall": degradation.get(f"{prefix}_pressed_key_recall_mean"),
                "pressed_key_f1": degradation.get(f"{prefix}_pressed_key_f1_mean"),
                "timestep_f1": degradation.get(f"{prefix}_timestep_f1_mean"),
                "max_unintended": degradation.get(f"{prefix}_max_unintended_key_state_mean"),
            }
        )
    correct_rows = [row for row in checkpoint_rows if row["audio_mode"] == "correct"]
    if correct_rows:
        best = best_row(correct_rows, "pressed_key_f1_mean")
        rows.append(
            {
                "system": f"Pipeline2 best checkpoint {best['checkpoint_step']}",
                "pressed_key_precision": best["pressed_key_precision_mean"],
                "pressed_key_recall": best["pressed_key_recall_mean"],
                "pressed_key_f1": best["pressed_key_f1_mean"],
                "timestep_f1": best["timestep_f1_mean"],
                "max_unintended": best["max_unintended_mean"],
            }
        )
    path = Path(output_dir) / "pipeline_comparison.csv"
    write_csv(path, rows)
    return path


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))


def min_value(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(min(float(row[key]) for row in rows))


def max_value(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(max(float(row[key]) for row in rows))


def best_row(rows: list[dict], key: str) -> dict:
    if not rows:
        return {}
    return dict(max(rows, key=lambda row: (float(row[key]), int(row.get("checkpoint_step", 0)))))


def manifest_for_env(env: DirectAudioGoalEnv, selection: EvaluationClipSelection) -> dict:
    return {
        "evaluation_audio_policy": (
            "canonical_eval audio is rendered from the same FluidSynth/TimGM6mb benchmark generator; "
            "no separate held-out acoustic split exists in the current Phase-A training artifacts"
        ),
        "true_heldout_acoustic_set_exists": False,
        "sampling_policy": "deterministic per-sequence canonical clip for evaluation",
        "policy_observation_fields": ["audio", "physical"],
        "logical_sequences": [list(seq) for seq in env.sequences],
        "audio_split": env.sampling_split,
        "clip_count": len(env.clips),
        "mismatch_mapping": selection.mismatch_mapping,
        "metrics_reused": [
            "pressed_key_metrics",
            "timestep_key_metrics",
            "binary_key_vector",
        ],
    }
