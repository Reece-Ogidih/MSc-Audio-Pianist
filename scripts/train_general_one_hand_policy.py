import argparse
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from ala_pianist.rl import GeneralOneHandGoalEnv, GeneralRewardConfig


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "general_one_hand"


def evaluate_policy(model, env_kwargs: dict, *, episodes: int = 3, deterministic: bool = True) -> dict:
    episode_results = []
    all_actions = []
    for episode in range(episodes):
        env = GeneralOneHandGoalEnv(**env_kwargs, clip_index=episode)
        obs, info = env.reset(seed=env_kwargs.get("seed", 0) + episode)
        total_reward = 0.0
        native_reward_sum = 0.0
        max_target = 0.0
        max_unintended = 0.0
        pressed_keys = set()
        for _ in range(env.horizon_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            all_actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            native_reward_sum += float(info["native_reward"])
            max_target = max(max_target, float(info["target_key_state"]))
            max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
            pressed_keys.update(info["pressed_keys"])
            if terminated or truncated:
                break
        episode_results.append(
            {
                "clip_pitches": list(env.curriculum_clip.pitches if env.curriculum_clip else ()),
                "debug_return": total_reward,
                "native_reward_sum": native_reward_sum,
                "max_target_key_state": max_target,
                "max_unintended_key_state": max_unintended,
                "pressed_keys": sorted(pressed_keys),
                "final_info": dict(info),
            }
        )
    return {
        "deterministic": deterministic,
        "episodes": episode_results,
        "action_stats": _action_stats(all_actions),
        "mean_debug_return": float(np.mean([item["debug_return"] for item in episode_results])),
        "mean_max_target_key_state": float(
            np.mean([item["max_target_key_state"] for item in episode_results])
        ),
        "mean_max_unintended_key_state": float(
            np.mean([item["max_unintended_key_state"] for item in episode_results])
        ),
    }


def _action_stats(actions) -> dict:
    if not actions:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean_abs": 0.0,
            "saturation_fraction": 0.0,
        }
    arr = np.asarray(actions, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
        "saturation_fraction": float(np.mean(np.abs(arr) >= 0.95)),
    }


def parse_reward_config(raw: str | None) -> GeneralRewardConfig:
    if raw is None:
        return GeneralRewardConfig()
    path = Path(raw)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(raw)
    return GeneralRewardConfig(**payload)


def reward_config_from_profile(profile: str, override: str | None = None) -> GeneralRewardConfig:
    if override is not None:
        return parse_reward_config(override)
    if profile == "default":
        return GeneralRewardConfig()
    if profile == "press_bonus":
        return GeneralRewardConfig(
            target_travel_weight=4.0,
            wrong_travel_weight=1.5,
            wrong_pressed_weight=1.0,
            action_weight=0.002,
            smoothness_weight=0.001,
            target_activation_bonus=5.0,
            target_activation_threshold=0.9,
            high_unintended_weight=1.0,
            high_unintended_threshold=0.75,
        )
    if profile in {"cleanup", "gated_cleanliness"}:
        return GeneralRewardConfig(
            target_travel_weight=4.0,
            wrong_travel_weight=0.25,
            wrong_pressed_weight=0.25,
            action_weight=0.002,
            smoothness_weight=0.001,
            target_activation_bonus=3.0,
            target_activation_threshold=0.9,
            high_unintended_weight=0.5,
            high_unintended_threshold=0.75,
            cleanup_gate_threshold=0.75,
            gated_unintended_weight=3.0,
            gated_wrong_pressed_weight=2.0,
            nearby_wrong_key_weight=2.0,
        )
    if profile == "anti_coupling":
        return GeneralRewardConfig(
            target_travel_weight=4.0,
            wrong_travel_weight=0.20,
            wrong_pressed_weight=0.25,
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
            csharp_dsharp_key54_weight=5.0,
            csharp_dsharp_pressed_weight=3.0,
            dsharp_csharp_key52_weight=2.0,
            dsharp_csharp_pressed_weight=1.0,
        )
    raise ValueError(f"Unknown reward profile {profile!r}.")


def parse_midi_pitches(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw == "":
        return None
    pitches = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not pitches:
        raise ValueError("--midi-pitches must include at least one MIDI pitch.")
    if len(set(pitches)) != len(pitches):
        raise ValueError("--midi-pitches must not contain duplicates.")
    return pitches


def parse_pitch_sampling_weights(raw: str | None) -> tuple[float, ...] | None:
    if raw is None or raw == "":
        return None
    weights = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not weights:
        raise ValueError("--pitch-sampling-weights must include at least one value.")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("--pitch-sampling-weights must be non-negative.")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("--pitch-sampling-weights must contain at least one positive value.")
    return tuple(weight / total for weight in weights)


def parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value {raw!r}.")


def load_or_create_model(args, env):
    if args.resume_model_path is None:
        return SAC(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=0,
            learning_rate=3e-4,
            buffer_size=max(5000, args.timesteps),
            learning_starts=min(1000, max(100, args.timesteps // 5)),
            batch_size=64,
            gamma=0.95,
            train_freq=1,
            gradient_steps=1,
        )
    model = SAC.load(args.resume_model_path, env=env)
    check_model_compatibility(model, env, lookahead=args.lookahead)
    return model


def check_model_compatibility(model, env, *, lookahead: int) -> None:
    if model.action_space.shape != env.action_space.shape:
        raise ValueError(
            f"Resume model action shape {model.action_space.shape} does not match "
            f"env action shape {env.action_space.shape}."
        )
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"Resume model observation shape {model.observation_space.shape} does not match "
            f"env observation shape {env.observation_space.shape}."
        )
    if tuple(env.native_goal_shape) != ((int(lookahead) + 1) * 89,):
        raise ValueError(
            f"Env native goal shape {env.native_goal_shape} is incompatible with lookahead {lookahead}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--midi-min", type=int, default=73)
    parser.add_argument("--midi-max", type=int, default=75)
    parser.add_argument("--midi-pitches", default=None)
    parser.add_argument("--pitch-sampling-weights", default=None)
    parser.add_argument("--curriculum", default="single_notes")
    parser.add_argument("--reward-config", default=None)
    parser.add_argument(
        "--reward-profile",
        default="default",
        choices=["default", "press_bonus", "cleanup", "gated_cleanliness", "anti_coupling"],
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizon-steps", type=int, default=64)
    parser.add_argument("--action-mode", default="direct", choices=["direct", "hold", "ramp_hold"])
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--ramp-steps", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-model-path", default=None)
    parser.add_argument("--reset-num-timesteps", default="false")
    parser.add_argument("--stage-name", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_config = reward_config_from_profile(args.reward_profile, args.reward_config)
    midi_pitches = parse_midi_pitches(args.midi_pitches)
    pitch_sampling_weights = parse_pitch_sampling_weights(args.pitch_sampling_weights)
    midi_min = min(midi_pitches) if midi_pitches is not None else args.midi_min
    midi_max = max(midi_pitches) if midi_pitches is not None else args.midi_max
    if pitch_sampling_weights is not None:
        pitch_count = len(midi_pitches) if midi_pitches is not None else midi_max - midi_min + 1
        if len(pitch_sampling_weights) != pitch_count:
            raise ValueError("--pitch-sampling-weights must match the configured pitch count.")
    env_kwargs = {
        "generated_midi_dir": str(output_dir / "generated_midi"),
        "curriculum": args.curriculum,
        "midi_min": midi_min,
        "midi_max": midi_max,
        "midi_pitches": midi_pitches,
        "pitch_sampling_weights": pitch_sampling_weights,
        "seed": args.seed,
        "note_count": 4,
        "lookahead": args.lookahead,
        "horizon_steps": args.horizon_steps,
        "reward_config": reward_config,
        "action_mode": args.action_mode,
        "action_repeat": args.action_repeat,
        "ramp_steps": args.ramp_steps,
    }

    env = GeneralOneHandGoalEnv(**env_kwargs)
    obs, info = env.reset(seed=args.seed)
    print("env=GeneralOneHandGoalEnv")
    print(f"action_space={env.action_space}")
    print(f"observation_shape={env.observation_space.shape}")
    print(f"native_goal_shape={env.native_goal_shape}")
    print(f"lookahead={env.lookahead}")
    print(f"curriculum_clip_pitches={env.curriculum_clip.pitches if env.curriculum_clip else ()}")
    print(f"reset_target_keys={info['target_keys']}")
    print(f"midi_pitches={midi_pitches if midi_pitches is not None else tuple(range(midi_min, midi_max + 1))}")
    print(f"pitch_sampling_weights={pitch_sampling_weights}")
    print(f"resume_model_path={args.resume_model_path}")
    print(f"action_mode={args.action_mode}")
    print(f"action_repeat={args.action_repeat}")
    print(f"ramp_steps={args.ramp_steps}")
    print(f"reward_profile={args.reward_profile}")
    if args.dry_run:
        return

    model = load_or_create_model(args, env)
    callbacks = []
    if args.checkpoint_freq > 0:
        checkpoint_dir = output_dir / "checkpoints" / (
            f"{args.stage_name or 'general'}_{args.action_mode}x{args.action_repeat}_{args.reward_profile}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="checkpoint",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )
        )
    start = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=parse_bool(args.reset_num_timesteps),
        callback=callbacks or None,
    )
    runtime_seconds = time.time() - start

    pitch_part = "-".join(str(pitch) for pitch in (midi_pitches or tuple(range(midi_min, midi_max + 1))))
    stage_prefix = f"{args.stage_name}_" if args.stage_name else ""
    run_name = (
        f"{stage_prefix}general_one_hand_sac_{args.curriculum}_pitches{pitch_part}_"
        f"lookahead{args.lookahead}_{args.action_mode}x{args.action_repeat}_"
        f"{args.reward_profile}_seed{args.seed}_{args.timesteps}"
    )
    model_path = output_dir / run_name
    model.save(model_path)
    deterministic_eval = evaluate_policy(model, env_kwargs, deterministic=True)
    stochastic_eval = evaluate_policy(model, env_kwargs, deterministic=False)
    summary = {
        "model_path": str(model_path) + ".zip",
        "runtime_seconds": runtime_seconds,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "lookahead": args.lookahead,
        "midi_min": midi_min,
        "midi_max": midi_max,
        "midi_pitches": midi_pitches,
        "pitch_sampling_weights": pitch_sampling_weights,
        "curriculum": args.curriculum,
        "action_mode": args.action_mode,
        "action_repeat": args.action_repeat,
        "ramp_steps": args.ramp_steps,
        "reward_profile": args.reward_profile,
        "checkpoint_freq": args.checkpoint_freq,
        "resume_model_path": args.resume_model_path,
        "reset_num_timesteps": parse_bool(args.reset_num_timesteps),
        "stage_name": args.stage_name,
        "compatibility_checks": {
            "action_space_shape": env.action_space.shape,
            "observation_space_shape": env.observation_space.shape,
            "lookahead": args.lookahead,
            "action_mode": args.action_mode,
            "action_repeat": args.action_repeat,
        },
        "reward_config": reward_config.__dict__,
        "native_goal_shape": tuple(env.native_goal_shape),
        "observation_shape": env.observation_space.shape,
        "action_space": "Box(-1, 1, shape=(22,))",
        "deterministic_eval": deterministic_eval,
        "stochastic_eval": stochastic_eval,
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"model_path={summary['model_path']}")
    print(f"summary_path={summary_path}")
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print(
        "deterministic "
        f"mean_target={deterministic_eval['mean_max_target_key_state']:.6f} "
        f"mean_unintended={deterministic_eval['mean_max_unintended_key_state']:.6f} "
        f"mean_return={deterministic_eval['mean_debug_return']:.6f} "
        f"action_stats={deterministic_eval['action_stats']}"
    )
    print(
        "stochastic "
        f"mean_target={stochastic_eval['mean_max_target_key_state']:.6f} "
        f"mean_unintended={stochastic_eval['mean_max_unintended_key_state']:.6f} "
        f"mean_return={stochastic_eval['mean_debug_return']:.6f} "
        f"action_stats={stochastic_eval['action_stats']}"
    )


if __name__ == "__main__":
    main()
