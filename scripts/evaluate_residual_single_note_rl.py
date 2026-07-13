import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.evaluation import record_action_rollout, save_trajectory_json
from ala_pianist.rl.residual_env import (
    ResidualSingleNoteEnv,
    action_ramp,
    evaluate_residual_action,
    infer_wrong_key,
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
    deterministic: bool,
    residual_scale: float = 0.1,
    save_demo_path: Path | None = None,
):
    model = SAC.load(model_path)
    env = ResidualSingleNoteEnv(
        midi_path=ROOT / "tmp" / f"residual_midi{target_midi}_eval.mid",
        target_midi=target_midi,
        wrong_key=wrong_key,
        horizon_steps=24,
        residual_scale=residual_scale,
        reward_mode="cleanliness",
    )
    obs, _ = env.reset()
    total_reward = 0.0
    native_reward = 0.0
    max_target = 0.0
    max_wrong = 0.0
    max_unintended = 0.0
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
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
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
        "deterministic": deterministic,
        "max_target_key_state": max_target,
        "max_wrong_key_state": max_wrong,
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
    }


def evaluate_native_zero(target_midi: int, wrong_key: int):
    midi_path = ROOT / "tmp" / f"residual_midi{target_midi}_zero_eval.mid"
    write_single_note_residual_midi(midi_path, target_midi=target_midi)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    action = np.zeros(env.action_spec().shape, dtype=env.action_spec().dtype)
    max_target = 0.0
    max_wrong = 0.0
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
        "max_unintended_key_state": max_unintended,
        "pressed_keys": sorted(pressed),
        "clean_press": pressed == {target_key},
        "near_clean_partial": bool(max_target >= 0.25 and max_unintended <= max_target + 0.02),
        "dirty_press": bool(target_key in pressed and pressed != {target_key}),
        "missed": target_key not in pressed and max_target < 0.25,
        "shaped_return": shaped_return,
        "native_reward_sum": native_reward,
    }


def default_model_path(target_midi: int, residual_scale: float) -> Path:
    if int(target_midi) == 75:
        legacy = OUT_DIR / "residual_sac_cleanliness_scale_0.1"
        if legacy.exists():
            return legacy
    scale_label = str(residual_scale).replace(".", "p")
    path = OUT_DIR / f"residual_sac_midi{target_midi}_cleanliness_scale_{scale_label}"
    return path if path.exists() else path.with_suffix(path.suffix + ".zip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-midi", type=int, default=75)
    parser.add_argument("--wrong-key", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--base-action-source", default="auto")
    parser.add_argument("--save-demo", action="store_true")
    args = parser.parse_args()

    wrong_key = infer_wrong_key(args.target_midi) if args.wrong_key is None else args.wrong_key
    model_path = args.model_path or default_model_path(args.target_midi, args.residual_scale)
    if not model_path.exists():
        raise FileNotFoundError(f"Expected trained residual model at {model_path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    zero_metrics = evaluate_native_zero(args.target_midi, wrong_key)
    base_metrics = evaluate_residual_action(
        residual_action=np.zeros(22),
        target_midi=args.target_midi,
        wrong_key=wrong_key,
        base_action_source=args.base_action_source,
        residual_scale=args.residual_scale,
        reward_mode="cleanliness",
    )
    demo_path = None
    if args.save_demo:
        demo_path = DEMO_DIR / f"residual_midi{args.target_midi}_deterministic_demo.json"
    deterministic = evaluate_model(
        model_path,
        target_midi=args.target_midi,
        wrong_key=wrong_key,
        deterministic=True,
        residual_scale=args.residual_scale,
        save_demo_path=demo_path,
    )
    stochastic = evaluate_model(
        model_path,
        target_midi=args.target_midi,
        wrong_key=wrong_key,
        deterministic=False,
        residual_scale=args.residual_scale,
    )
    summary = {
        "target_midi": args.target_midi,
        "target_key": args.target_midi - 21,
        "target_note": note_name(args.target_midi),
        "wrong_key": wrong_key,
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


if __name__ == "__main__":
    main()
