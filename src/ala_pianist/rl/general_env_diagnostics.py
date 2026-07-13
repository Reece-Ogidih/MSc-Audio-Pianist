"""Signal diagnostics for the general one-hand native-goal env."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.music import write_curriculum_midi
from ala_pianist.rl.general_one_hand_env import GeneralOneHandGoalEnv
from ala_pianist.rl.residual_env import (
    ResidualSingleNoteEnv,
    action_ramp,
    get_dirty_dsharp5_base_action,
    infer_wrong_key,
    make_residual_observation,
    note_name,
)


D_SHARP_5_MIDI = 75
D_SHARP_5_KEY = 54
DEFAULT_DSHARP5_RESIDUAL_MODEL = (
    Path("/home/reece_dev/msc-audio-pianist")
    / "experiments"
    / "residual_single_note"
    / "residual_sac_cleanliness_scale_0.1"
)


@dataclass(frozen=True)
class RolloutDiagnostic:
    name: str
    max_target_key_state: float
    max_unintended_key_state: float
    pressed_keys: tuple[int, ...]
    shaped_return: float
    native_reward_sum: float
    positive_target_signal: bool
    final_reward_breakdown: dict[str, float]
    steps: tuple[dict[str, Any], ...]
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_dsharp5_diagnostic_midi(
    path: str | Path,
    *,
    midi_min: int = 73,
    midi_max: int = 75,
) -> Path:
    """Write the generated single-note D#5 clip used by the signal diagnostic."""

    if not midi_min <= D_SHARP_5_MIDI <= midi_max:
        raise ValueError("D#5/MIDI 75 must be inside the diagnostic MIDI range.")
    clip_index = D_SHARP_5_MIDI - int(midi_min)
    clip = write_curriculum_midi(
        path,
        mode="single_notes",
        midi_min=midi_min,
        midi_max=midi_max,
        seed=0,
        clip_index=clip_index,
        note_count=1,
    )
    if clip.pitches != (D_SHARP_5_MIDI,):
        raise RuntimeError(f"Expected generated D#5 clip, got pitches {clip.pitches}.")
    return clip.midi_path


def goal_timing_diagnostic(env: GeneralOneHandGoalEnv, *, steps: int = 8) -> dict[str, Any]:
    """Record active native goal frames before and during a short zero-action rollout."""

    obs, info = env.reset()
    del obs
    records = [_goal_record(env, step=0, info=info)]
    for step in range(1, steps + 1):
        _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        records.append(_goal_record(env, step=step, info=info))
        if terminated or truncated:
            break
    active_steps = [
        record["step"]
        for record in records
        if D_SHARP_5_KEY in record["current_target_keys"]
        or D_SHARP_5_KEY in record["future_target_keys"]
    ]
    return {
        "native_goal_shape": tuple(env.native_goal_shape),
        "observation_shape": tuple(env.observation_space.shape),
        "expected_target_midi": D_SHARP_5_MIDI,
        "expected_target_key": D_SHARP_5_KEY,
        "expected_target_note": note_name(D_SHARP_5_MIDI),
        "target_key_seen_in_goal": bool(active_steps),
        "target_key_goal_steps": active_steps,
        "records": records,
    }


def run_zero_action_diagnostic(env: GeneralOneHandGoalEnv, *, horizon_steps: int = 24) -> RolloutDiagnostic:
    return rollout_diagnostic(
        env,
        name="zero_action",
        policy=lambda step, env: np.zeros(env.action_space.shape, dtype=np.float32),
        horizon_steps=horizon_steps,
    )


def run_random_action_diagnostics(
    midi_path: str | Path,
    *,
    midi_min: int,
    midi_max: int,
    lookahead: int,
    seed: int = 0,
    horizon_steps: int = 24,
) -> tuple[RolloutDiagnostic, ...]:
    rng = np.random.default_rng(seed)
    constant = rng.uniform(-1.0, 1.0, size=22).astype(np.float32)
    ramped = rng.uniform(-1.0, 1.0, size=22).astype(np.float32)

    def make_env() -> GeneralOneHandGoalEnv:
        return GeneralOneHandGoalEnv(
            midi_path=midi_path,
            midi_min=midi_min,
            midi_max=midi_max,
            lookahead=lookahead,
            horizon_steps=horizon_steps,
        )

    return (
        rollout_diagnostic(
            make_env(),
            name="random_per_step",
            policy=lambda step, env: rng.uniform(-1.0, 1.0, size=22).astype(np.float32),
            horizon_steps=horizon_steps,
        ),
        rollout_diagnostic(
            make_env(),
            name="random_constant",
            policy=lambda step, env: constant,
            horizon_steps=horizon_steps,
        ),
        rollout_diagnostic(
            make_env(),
            name="random_ramped_held",
            policy=lambda step, env: ramped * action_ramp(step),
            horizon_steps=horizon_steps,
        ),
    )


def run_known_dsharp5_base_diagnostic(
    env: GeneralOneHandGoalEnv,
    *,
    horizon_steps: int = 24,
    ramped: bool = True,
) -> RolloutDiagnostic:
    """Apply the known dirty D#5 base action inside the general normalized env."""

    try:
        native_base_action = get_dirty_dsharp5_base_action()
    except Exception as exc:  # pragma: no cover - exercised only when local artifact setup breaks.
        return RolloutDiagnostic(
            name="known_dsharp5_base_action",
            max_target_key_state=0.0,
            max_unintended_key_state=0.0,
            pressed_keys=(),
            shaped_return=0.0,
            native_reward_sum=0.0,
            positive_target_signal=False,
            final_reward_breakdown={},
            steps=(),
            skipped_reason=str(exc),
        )
    normalized = env.normalize_native_action(native_base_action)
    return rollout_diagnostic(
        env,
        name="known_dsharp5_base_action_ramped" if ramped else "known_dsharp5_base_action_held",
        policy=lambda step, env: normalized * action_ramp(step) if ramped else normalized,
        horizon_steps=horizon_steps,
    )


def run_dsharp5_residual_policy_diagnostic(
    env: GeneralOneHandGoalEnv,
    *,
    model_path: str | Path = DEFAULT_DSHARP5_RESIDUAL_MODEL,
    horizon_steps: int = 24,
    residual_scale: float = 0.1,
) -> RolloutDiagnostic:
    """Apply the trained D#5 residual policy through the general env action interface."""

    model_path = Path(model_path)
    if not model_path.exists() and not model_path.with_suffix(".zip").exists():
        return RolloutDiagnostic(
            name="known_dsharp5_residual_policy",
            max_target_key_state=0.0,
            max_unintended_key_state=0.0,
            pressed_keys=(),
            shaped_return=0.0,
            native_reward_sum=0.0,
            positive_target_signal=False,
            final_reward_breakdown={},
            steps=(),
            skipped_reason=f"Residual model not found at {model_path}",
        )
    try:
        model = SAC.load(model_path)
        helper_env = ResidualSingleNoteEnv(
            target_midi=D_SHARP_5_MIDI,
            wrong_key=infer_wrong_key(D_SHARP_5_MIDI),
            horizon_steps=horizon_steps,
            residual_scale=residual_scale,
            reward_mode="cleanliness",
        )
    except Exception as exc:  # pragma: no cover - model loading depends on ignored artifacts.
        return RolloutDiagnostic(
            name="known_dsharp5_residual_policy",
            max_target_key_state=0.0,
            max_unintended_key_state=0.0,
            pressed_keys=(),
            shaped_return=0.0,
            native_reward_sum=0.0,
            positive_target_signal=False,
            final_reward_breakdown={},
            steps=(),
            skipped_reason=str(exc),
        )

    def policy(step: int, general_env: GeneralOneHandGoalEnv) -> np.ndarray:
        obs = make_residual_observation(
            general_env,
            target_key=D_SHARP_5_KEY,
            wrong_key=infer_wrong_key(D_SHARP_5_MIDI),
            step_count=step,
            horizon_steps=horizon_steps,
        )
        residual, _ = model.predict(obs, deterministic=True)
        normalized = np.clip(
            helper_env.base_normalized_action
            + residual_scale * np.asarray(residual, dtype=np.float32),
            -1.0,
            1.0,
        )
        native = helper_env.rescale_action(normalized)
        ramped_native = native.astype(np.float32) * action_ramp(step)
        return general_env.normalize_native_action(ramped_native)

    return rollout_diagnostic(
        env,
        name="known_dsharp5_residual_policy",
        policy=policy,
        horizon_steps=horizon_steps,
    )


def rollout_diagnostic(
    env: GeneralOneHandGoalEnv,
    *,
    name: str,
    policy,
    horizon_steps: int,
) -> RolloutDiagnostic:
    obs, info = env.reset()
    del obs
    max_target = float(info["target_key_state"])
    max_unintended = float(info["max_unintended_key_state"])
    pressed = set(info["pressed_keys"])
    shaped_return = 0.0
    native_reward_sum = 0.0
    steps = []
    final_breakdown = reward_breakdown(env, info)
    for step in range(horizon_steps):
        action = np.asarray(policy(step, env), dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        breakdown = reward_breakdown(env, info)
        final_breakdown = breakdown
        shaped_return += float(reward)
        native_reward_sum += float(info["native_reward"])
        max_target = max(max_target, float(info["target_key_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        pressed.update(info["pressed_keys"])
        steps.append(
            {
                "step": step,
                "target_keys": tuple(info["target_keys"]),
                "pressed_keys": tuple(info["pressed_keys"]),
                "target_key_state": float(info["target_key_state"]),
                "max_unintended_key_state": float(info["max_unintended_key_state"]),
                "shaped_reward": float(reward),
                "native_reward": float(info["native_reward"]),
                "reward_breakdown": breakdown,
            }
        )
        if terminated or truncated:
            break
    return RolloutDiagnostic(
        name=name,
        max_target_key_state=max_target,
        max_unintended_key_state=max_unintended,
        pressed_keys=tuple(sorted(pressed)),
        shaped_return=shaped_return,
        native_reward_sum=native_reward_sum,
        positive_target_signal=bool(max_target > 0.0 or D_SHARP_5_KEY in pressed),
        final_reward_breakdown=final_breakdown,
        steps=tuple(steps),
    )


def reward_breakdown(env: GeneralOneHandGoalEnv, info: dict[str, Any]) -> dict[str, float]:
    components = info["reward_components"]
    cfg = env.reward_config
    target_reward = cfg.target_travel_weight * components["target_key_state"]
    wrong_penalty = cfg.wrong_travel_weight * components["max_unintended_key_state"]
    wrong_pressed_penalty = cfg.wrong_pressed_weight * components["wrong_pressed_key_count"]
    action_penalty = cfg.action_weight * components["action_magnitude"]
    smoothness_penalty = cfg.smoothness_weight * components["smoothness"]
    fingering_reward = cfg.fingering_weight * components["fingering_score"]
    native_reward = cfg.native_reward_weight * components["native_reward"]
    total = (
        target_reward
        - wrong_penalty
        - wrong_pressed_penalty
        - action_penalty
        - smoothness_penalty
        + fingering_reward
        + native_reward
    )
    return {
        "target_travel_reward": float(target_reward),
        "wrong_key_penalty": float(wrong_penalty),
        "wrong_pressed_key_penalty": float(wrong_pressed_penalty),
        "action_magnitude_penalty": float(action_penalty),
        "smoothness_penalty": float(smoothness_penalty),
        "fingering_reward": float(fingering_reward),
        "native_reward_component": float(native_reward),
        "total_shaped_reward": float(total),
    }


def _goal_record(env: GeneralOneHandGoalEnv, *, step: int, info: dict[str, Any]) -> dict[str, Any]:
    goal = np.asarray(env._last_timestep.observation["goal"], dtype=np.float32)
    frames = goal.reshape(env.lookahead + 1, env.task.piano.n_keys + 1)
    active = [
        tuple(int(key) for key in np.flatnonzero(frame[:-1] > 0.5))
        for frame in frames
    ]
    return {
        "step": int(step),
        "current_target_keys": active[0],
        "future_target_keys": tuple(sorted({key for frame in active[1:] for key in frame})),
        "all_goal_frames": tuple(active),
        "env_current_target_keys": tuple(info["target_keys"]),
    }
