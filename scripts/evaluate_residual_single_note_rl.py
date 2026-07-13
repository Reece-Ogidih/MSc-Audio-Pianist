import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.evaluation import record_action_rollout, save_trajectory_json
from ala_pianist.rl.residual_env import (
    REWARD_MODES,
    ResidualSingleNoteEnv,
    action_ramp,
    evaluate_residual_action,
    infer_wrong_key,
    infer_wrong_keys,
    note_name,
    write_single_note_residual_midi,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "residual_single_note"
DEMO_DIR = ROOT / "experiments" / "demos"


def evaluate_model(
    model_path: Path,
    *,
    target_midi: int,
    wrong_key: int,
    wrong_keys: tuple[int, ...],
    deterministic: bool,
    residual_scale: float = 0.1,
    reward_mode: str = "cleanliness",
    base_action_penalty: float = 0.0,
    save_demo_path: Path | None = None,
):
    model = SAC.load(model_path)
    env = ResidualSingleNoteEnv(
        midi_path=ROOT / "tmp" / f"residual_midi{target_midi}_eval.mid",
        target_midi=target_midi,
        wrong_key=wrong_key,
        wrong_keys=wrong_keys,
        horizon_steps=24,
        residual_scale=residual_scale,
        reward_mode=reward_mode,
        base_action_penalty=base_action_penalty,
    )
    obs, _ = env.reset()
    total_reward = 0.0
    native_reward = 0.0
    max_target = 0.0
    max_wrong = 0.0
    max_key_states = {52: 0.0, 53: 0.0, 54: 0.0, 55: 0.0, 56: 0.0}
    max_unintended = 0.0
    max_residual_magnitude = 0.0
    max_action_deviation = 0.0
    pressed = set()
    actions = []
    native_actions = []
    for step in range(24):
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(np.asarray(action, dtype=float))
        normalized = np.clip(
            env.base_normalized_action + residual_scale * np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )
        native_actions.append(env.rescale_action(normalized) * action_ramp(step))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        native_reward += 0.0 if info["native_reward"] is None else float(info["native_reward"])
        max_target = max(max_target, float(info["target_key_state"]))
        max_wrong = max(max_wrong, float(info["wrong_key_state"]))
        for key in max_key_states:
            max_key_states[key] = max(max_key_states[key], float(info[f"key_{key}_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        max_residual_magnitude = max(max_residual_magnitude, float(info["residual_magnitude"]))
        max_action_deviation = max(max_action_deviation, float(info["action_deviation_from_base"]))
        pressed.update(info["pressed_keys"])
        if terminated or truncated:
            break

    if save_demo_path is not None and native_actions:
        demo_env = ALAOneHandEnv(env.midi_path)
        demo_env.reset()
        records = []
        for action in native_actions:
            records.extend(
                record_action_rollout(
                    demo_env,
                    target_midi=target_midi,
                    action=np.asarray(action, dtype=demo_env.action_spec().dtype),
                    horizon_steps=1,
                    ramp=False,
                )
            )
        save_trajectory_json(records, save_demo_path)

    arr = np.asarray(actions, dtype=float)
    target_key = target_midi - 21
    return {
        "target_midi": target_midi,
        "target_key": target_key,
        "target_note": note_name(target_midi),
        "wrong_key": wrong_key,
        "wrong_keys": wrong_keys,
        "deterministic": deterministic,
        "max_target_key_state": max_target,
        "max_wrong_key_state": max_wrong,
        **{f"max_key_{key}_state": value for key, value in max_key_states.items()},
        "max_unintended_key_state": max_unintended,
        "pressed_keys": sorted(pressed),
        "clean_press": pressed == {target_key},
        "near_clean_partial": bool(max_target >= 0.25 and max_unintended <= max_target + 0.02),
        "dirty_press": bool(target_key in pressed and pressed != {target_key}),
        "missed": target_key not in pressed and max_target < 0.25,
        "shaped_return": total_reward,
        "native_reward_sum": native_reward,
        "action_mean_abs": float(np.mean(np.abs(arr))) if arr.size else 0.0,
        "action_saturation_fraction": float(np.mean(np.abs(arr) >= 0.95)) if arr.size else 0.0,
        "max_residual_magnitude": max_residual_magnitude,
        "max_action_deviation_from_base": max_action_deviation,
    }


def evaluate_native_zero(target_midi: int, wrong_key: int):
    midi_path = ROOT / "tmp" / f"residual_midi{target_midi}_zero_eval.mid"
    write_single_note_residual_midi(midi_path, target_midi=target_midi)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    action = np.zeros(env.action_spec().shape, dtype=env.action_spec().dtype)
    max_target = 0.0
    max_wrong = 0.0
    max_key_states = {52: 0.0, 53: 0.0, 54: 0.0, 55: 0.0, 56: 0.0}
    max_unintended = 0.0
    pressed = set()
    shaped_return = 0.0
    native_reward = 0.0
    target_key = target_midi - 21
    for _ in range(24):
        timestep = env.step(action)
        target = env.target_key_state(target_key) or 0.0
        wrong = env.target_key_state(wrong_key) or 0.0
        unintended = env.max_unintended_key_state(target_key)
        pressed.update(env.current_pressed_keys())
        shaped_return += 10.0 * target - 8.0 * wrong - 5.0 * unintended
        if env.current_reward() is not None:
            native_reward += float(env.current_reward())
        max_target = max(max_target, target)
        max_wrong = max(max_wrong, wrong)
        for key in max_key_states:
            max_key_states[key] = max(max_key_states[key], env.target_key_state(key) or 0.0)
        max_unintended = max(max_unintended, unintended)
        if timestep.last():
            break
    return {
        "target_midi": target_midi,
        "target_key": target_key,
        "target_note": note_name(target_midi),
        "wrong_key": wrong_key,
        "max_target_key_state": max_target,
        "max_wrong_key_state": max_wrong,
        **{f"max_key_{key}_state": value for key, value in max_key_states.items()},
        "max_unintended_key_state": max_unintended,
        "pressed_keys": sorted(pressed),
        "clean_press": pressed == {target_key},
        "near_clean_partial": bool(max_target >= 0.25 and max_unintended <= max_target + 0.02),
        "dirty_press": bool(target_key in pressed and pressed != {target_key}),
        "missed": target_key not in pressed and max_target < 0.25,
        "shaped_return": shaped_return,
        "native_reward_sum": native_reward,
    }


def default_model_path(
    target_midi: int,
    residual_scale: float,
    *,
    reward_mode: str = "cleanliness",
    base_action_penalty: float = 0.0,
) -> Path:
    if int(target_midi) == 75:
        legacy = OUT_DIR / "residual_sac_cleanliness_scale_0.1"
        if reward_mode == "cleanliness" and residual_scale == 0.1 and base_action_penalty == 0.0 and legacy.exists():
            return legacy
    scale_label = str(residual_scale).replace(".", "p")
    penalty_label = str(base_action_penalty).replace(".", "p")
    if base_action_penalty:
        path = (
            OUT_DIR
            / f"residual_sac_midi{target_midi}_{reward_mode}_scale_{scale_label}_penalty_{penalty_label}"
        )
    else:
        path = OUT_DIR / f"residual_sac_midi{target_midi}_{reward_mode}_scale_{scale_label}"
    return path if path.exists() else path.with_suffix(path.suffix + ".zip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-midi", type=int, default=75)
    parser.add_argument("--wrong-key", type=int, default=None)
    parser.add_argument("--wrong-keys", default=None, help="Comma-separated key indices to report/penalise.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="cleanliness")
    parser.add_argument("--base-action-penalty", type=float, default=0.0)
    parser.add_argument("--base-action-source", default="auto")
    parser.add_argument("--save-demo", action="store_true")
    args = parser.parse_args()

    wrong_key = infer_wrong_key(args.target_midi) if args.wrong_key is None else args.wrong_key
    wrong_keys = _parse_wrong_keys(args.wrong_keys, args.target_midi)
    model_path = args.model_path or default_model_path(
        args.target_midi,
        args.residual_scale,
        reward_mode=args.reward_mode,
        base_action_penalty=args.base_action_penalty,
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained residual model at {model_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    zero_metrics = evaluate_native_zero(args.target_midi, wrong_key)
    base_metrics = evaluate_residual_action(
        residual_action=np.zeros(22),
        target_midi=args.target_midi,
        wrong_key=wrong_key,
        wrong_keys=wrong_keys,
        base_action_source=args.base_action_source,
        residual_scale=args.residual_scale,
        reward_mode=args.reward_mode,
        base_action_penalty=args.base_action_penalty,
    )
    demo_path = None
    if args.save_demo:
        demo_path = DEMO_DIR / f"residual_midi{args.target_midi}_deterministic_demo.json"
    deterministic = evaluate_model(
        model_path,
        target_midi=args.target_midi,
        wrong_key=wrong_key,
        wrong_keys=wrong_keys,
        deterministic=True,
        residual_scale=args.residual_scale,
        reward_mode=args.reward_mode,
        base_action_penalty=args.base_action_penalty,
        save_demo_path=demo_path,
    )
    stochastic = evaluate_model(
        model_path,
        target_midi=args.target_midi,
        wrong_key=wrong_key,
        wrong_keys=wrong_keys,
        deterministic=False,
        residual_scale=args.residual_scale,
        reward_mode=args.reward_mode,
        base_action_penalty=args.base_action_penalty,
    )
    summary = {
        "target_midi": args.target_midi,
        "target_key": args.target_midi - 21,
        "target_note": note_name(args.target_midi),
        "wrong_key": wrong_key,
        "wrong_keys": wrong_keys,
        "reward_mode": args.reward_mode,
        "residual_scale": args.residual_scale,
        "base_action_penalty": args.base_action_penalty,
        "model_path": str(model_path),
        "demo_path": None if demo_path is None else str(demo_path),
        "zero_action_proxy": zero_metrics,
        "dirty_base_action": base_metrics,
        "residual_policy_deterministic": deterministic,
        "residual_policy_stochastic": stochastic,
    }
    summary_path = OUT_DIR / f"residual_midi{args.target_midi}_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary_path={summary_path}")
    print(f"model_path={model_path}")
    print(f"demo_path={demo_path}")
    print(f"zero_action_proxy={zero_metrics}")
    print(f"dirty_base_action={base_metrics}")
    print(f"residual_policy_deterministic={deterministic}")
    print(f"residual_policy_stochastic={stochastic}")


def _parse_wrong_keys(value: str | None, target_midi: int) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return infer_wrong_keys(target_midi)
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


if __name__ == "__main__":
    main()
