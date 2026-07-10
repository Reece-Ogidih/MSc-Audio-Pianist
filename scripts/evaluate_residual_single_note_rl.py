import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.learning.random_search import write_single_note_learning_midi
from ala_pianist.rl.residual_env import ResidualSingleNoteEnv, evaluate_residual_action


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "residual_single_note"
MODEL_PATH = OUT_DIR / "residual_sac_cleanliness_scale_0.1"
SUMMARY_PATH = OUT_DIR / "residual_single_note_eval_summary.json"


def evaluate_model(model_path: Path, *, deterministic: bool, residual_scale: float = 0.1):
    model = SAC.load(model_path)
    env = ResidualSingleNoteEnv(
        midi_path=ROOT / "tmp" / "residual_single_note_eval.mid",
        horizon_steps=24,
        residual_scale=residual_scale,
        reward_mode="cleanliness",
    )
    obs, _ = env.reset()
    total_reward = 0.0
    native_reward = 0.0
    max_target = 0.0
    max_key50 = 0.0
    max_unintended = 0.0
    pressed = set()
    actions = []
    for _ in range(24):
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(np.asarray(action, dtype=float))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        native_reward += 0.0 if info["native_reward"] is None else float(info["native_reward"])
        max_target = max(max_target, float(info["target_key_state"]))
        max_key50 = max(max_key50, float(info["key50_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        pressed.update(info["pressed_keys"])
        if terminated or truncated:
            break
    arr = np.asarray(actions, dtype=float)
    return {
        "deterministic": deterministic,
        "max_target_key_state": max_target,
        "max_key50_state": max_key50,
        "max_unintended_key_state": max_unintended,
        "pressed_keys": sorted(pressed),
        "clean_press": pressed == {54},
        "near_clean_partial": bool(max_target >= 0.25 and max_unintended <= max_target + 0.02),
        "dirty_press": bool(54 in pressed and pressed != {54}),
        "missed": 54 not in pressed and max_target < 0.25,
        "shaped_return": total_reward,
        "native_reward_sum": native_reward,
        "action_mean_abs": float(np.mean(np.abs(arr))) if arr.size else 0.0,
        "action_saturation_fraction": float(np.mean(np.abs(arr) >= 0.95)) if arr.size else 0.0,
    }


def evaluate_native_zero():
    midi_path = ROOT / "tmp" / "residual_single_note_zero_eval.mid"
    write_single_note_learning_midi(midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    action = np.zeros(env.action_spec().shape, dtype=env.action_spec().dtype)
    max_target = 0.0
    max_key50 = 0.0
    max_unintended = 0.0
    pressed = set()
    shaped_return = 0.0
    native_reward = 0.0
    for _ in range(24):
        timestep = env.step(action)
        target = env.target_key_state(54) or 0.0
        key50 = env.target_key_state(50) or 0.0
        unintended = env.max_unintended_key_state(54)
        pressed.update(env.current_pressed_keys())
        shaped_return += 10.0 * target - 8.0 * key50 - 5.0 * unintended
        if env.current_reward() is not None:
            native_reward += float(env.current_reward())
        max_target = max(max_target, target)
        max_key50 = max(max_key50, key50)
        max_unintended = max(max_unintended, unintended)
        if timestep.last():
            break
    return {
        "max_target_key_state": max_target,
        "max_key50_state": max_key50,
        "max_unintended_key_state": max_unintended,
        "pressed_keys": sorted(pressed),
        "clean_press": pressed == {54},
        "near_clean_partial": bool(max_target >= 0.25 and max_unintended <= max_target + 0.02),
        "dirty_press": bool(54 in pressed and pressed != {54}),
        "missed": 54 not in pressed and max_target < 0.25,
        "shaped_return": shaped_return,
        "native_reward_sum": native_reward,
    }


def main() -> None:
    model_path = MODEL_PATH if MODEL_PATH.exists() else MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".zip")
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained residual model at {MODEL_PATH} or {model_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zero_metrics = evaluate_native_zero()
    base_metrics = evaluate_residual_action(residual_action=np.zeros(22), residual_scale=0.1, reward_mode="cleanliness")
    deterministic = evaluate_model(model_path, deterministic=True)
    stochastic = evaluate_model(model_path, deterministic=False)
    summary = {
        "model_path": str(model_path),
        "zero_action_proxy": zero_metrics,
        "dirty_base_action": base_metrics,
        "residual_policy_deterministic": deterministic,
        "residual_policy_stochastic": stochastic,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary_path={SUMMARY_PATH}")
    print(f"model_path={model_path}")
    print(f"zero_action_proxy={zero_metrics}")
    print(f"dirty_base_action={base_metrics}")
    print(f"residual_policy_deterministic={deterministic}")
    print(f"residual_policy_stochastic={stochastic}")


if __name__ == "__main__":
    main()
