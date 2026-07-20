import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from ala_pianist.rl import DroQAgent, DroQConfig, GeneralOneHandGoalEnv, GeneralRewardConfig, ReplayBuffer
from train_general_one_hand_policy import (
    parse_midi_pitches,
    parse_pitch_sampling_weights,
    parse_reward_config,
    parse_sequence_pitches,
    reward_config_from_profile,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "general_one_hand" / "droq"


def evaluate_agent(
    agent: DroQAgent,
    env_kwargs: dict,
    *,
    episodes: int = 3,
    deterministic: bool = True,
) -> dict:
    results = []
    all_actions = []
    for episode in range(episodes):
        env = GeneralOneHandGoalEnv(**env_kwargs, clip_index=episode)
        observation, _ = env.reset(seed=env_kwargs.get("seed", 0) + episode)
        total_reward = 0.0
        native_reward_sum = 0.0
        max_target = 0.0
        max_unintended = 0.0
        pressed_keys = set()
        for _ in range(env.horizon_steps):
            action = agent.act(observation, deterministic=deterministic)
            all_actions.append(action.copy())
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            native_reward_sum += float(info["native_reward"])
            max_target = max(max_target, float(info["target_key_state"]))
            max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
            pressed_keys.update(info["pressed_keys"])
            if terminated or truncated:
                break
        results.append(
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
        "episodes": results,
        "action_stats": _action_stats(all_actions),
        "mean_debug_return": float(np.mean([result["debug_return"] for result in results])),
        "mean_max_target_key_state": float(
            np.mean([result["max_target_key_state"] for result in results])
        ),
        "mean_max_unintended_key_state": float(
            np.mean([result["max_unintended_key_state"] for result in results])
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


def build_env_kwargs(args, output_dir: Path, reward_config: GeneralRewardConfig) -> dict:
    midi_pitches = parse_midi_pitches(args.midi_pitches)
    pitch_sampling_weights = parse_pitch_sampling_weights(args.pitch_sampling_weights)
    sequence_pitches = parse_sequence_pitches(args.sequence_pitches)
    sequence_sampling_weights = parse_pitch_sampling_weights(args.sequence_sampling_weights)
    midi_min = min(midi_pitches) if midi_pitches is not None else args.midi_min
    midi_max = max(midi_pitches) if midi_pitches is not None else args.midi_max
    if pitch_sampling_weights is not None:
        pitch_count = len(midi_pitches) if midi_pitches is not None else midi_max - midi_min + 1
        if len(pitch_sampling_weights) != pitch_count:
            raise ValueError("--pitch-sampling-weights must match the configured pitch count.")
    if sequence_sampling_weights is not None:
        if sequence_pitches is None:
            raise ValueError("--sequence-sampling-weights requires --sequence-pitches.")
        if len(sequence_sampling_weights) != len(sequence_pitches):
            raise ValueError("--sequence-sampling-weights must match --sequence-pitches.")
    return {
        "generated_midi_dir": str(output_dir / "generated_midi"),
        "curriculum": args.curriculum,
        "midi_min": midi_min,
        "midi_max": midi_max,
        "midi_pitches": midi_pitches,
        "pitch_sampling_weights": pitch_sampling_weights,
        "sequence_pitches": sequence_pitches,
        "sequence_sampling_weights": sequence_sampling_weights,
        "sequence_timing_profile": args.sequence_timing_profile,
        "note_duration": args.note_duration,
        "note_gap": args.note_gap,
        "note_velocity": args.note_velocity,
        "timing_jitter": args.timing_jitter,
        "seed": args.seed,
        "note_count": 4,
        "lookahead": args.lookahead,
        "horizon_steps": args.horizon_steps,
        "reward_config": reward_config,
        "action_mode": args.action_mode,
        "action_repeat": args.action_repeat,
        "ramp_steps": args.ramp_steps,
    }


def _safe_stage_name(value: str | None) -> str:
    return value or "droq_general_one_hand"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--midi-min", type=int, default=73)
    parser.add_argument("--midi-max", type=int, default=75)
    parser.add_argument("--midi-pitches", default=None)
    parser.add_argument("--pitch-sampling-weights", default=None)
    parser.add_argument("--sequence-pitches", default=None)
    parser.add_argument("--sequence-sampling-weights", default=None)
    parser.add_argument(
        "--sequence-timing-profile",
        default="legacy_curriculum",
        choices=["legacy_curriculum", "aligned"],
    )
    parser.add_argument("--note-duration", type=float, default=None)
    parser.add_argument("--note-gap", type=float, default=None)
    parser.add_argument("--note-velocity", type=int, default=None)
    parser.add_argument("--timing-jitter", type=float, default=0.0)
    parser.add_argument("--curriculum", default="single_notes")
    parser.add_argument("--reward-config", default=None)
    parser.add_argument(
        "--reward-profile",
        default="default",
        choices=[
            "default",
            "press_bonus",
            "cleanup",
            "gated_cleanliness",
            "anti_coupling",
            "transition_cleanup",
        ],
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--horizon-steps", type=int, default=64)
    parser.add_argument("--action-mode", default="direct", choices=["direct", "hold", "ramp_hold"])
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--ramp-steps", type=int, default=1)
    parser.add_argument("--stage-name", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--utd-ratio", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--critic-ensemble-size", type=int, default=2)
    parser.add_argument("--critic-dropout", type=float, default=0.01)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--fixed-alpha", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    reward_config = (
        parse_reward_config(args.reward_config)
        if args.reward_config is not None
        else reward_config_from_profile(args.reward_profile)
    )
    env_kwargs = build_env_kwargs(args, output_dir, reward_config)
    env = GeneralOneHandGoalEnv(**env_kwargs)
    observation, info = env.reset(seed=args.seed)

    print("env=GeneralOneHandGoalEnv")
    print("trainer=DroQAgent")
    print(f"action_space={env.action_space}")
    print(f"observation_shape={env.observation_space.shape}")
    print(f"native_goal_shape={env.native_goal_shape}")
    print(f"lookahead={env.lookahead}")
    print(f"curriculum_clip_pitches={env.curriculum_clip.pitches if env.curriculum_clip else ()}")
    print(f"reset_target_keys={info['target_keys']}")
    print(f"sequence_pitches={env.sequence_pitches}")
    print(f"sequence_sampling_weights={env.sequence_sampling_weights}")
    print(f"sequence_timing_profile={args.sequence_timing_profile}")
    print(f"action_mode={args.action_mode}")
    print(f"action_repeat={args.action_repeat}")
    print(f"reward_profile={args.reward_profile}")
    if args.dry_run:
        return

    config = DroQConfig(
        observation_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        hidden_dim=args.hidden_dim,
        critic_ensemble_size=args.critic_ensemble_size,
        critic_dropout=args.critic_dropout,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=not args.fixed_alpha,
        batch_size=args.batch_size,
        utd_ratio=args.utd_ratio,
        buffer_size=args.buffer_size,
        device=args.device,
    )
    agent = DroQAgent(config)
    replay_buffer = ReplayBuffer(config.observation_dim, config.action_dim, config.buffer_size)

    run_name = (
        f"{_safe_stage_name(args.stage_name)}_droq_{args.curriculum}_"
        f"lookahead{args.lookahead}_{args.action_mode}x{args.action_repeat}_"
        f"{args.reward_profile}_seed{args.seed}_{args.timesteps}"
    )
    checkpoint_dir = output_dir / "checkpoints" / run_name
    start = time.time()
    losses: list[dict[str, float]] = []
    episode_return = 0.0
    for step in range(1, args.timesteps + 1):
        if step <= args.learning_starts:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(observation, deterministic=False)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        replay_buffer.add(observation, action, float(reward), next_observation, done)
        episode_return += float(reward)
        observation = next_observation

        if replay_buffer.size >= args.batch_size and step > args.learning_starts:
            for _ in range(args.utd_ratio):
                losses.append(
                    agent.update(
                        replay_buffer.sample(args.batch_size, device=agent.device)
                    )
                )

        if done:
            observation, info = env.reset()
            episode_return = 0.0

        if args.checkpoint_freq > 0 and step % args.checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{step}_steps.pt"
            agent.save(
                checkpoint_path,
                replay_buffer=replay_buffer,
                extra={"step": step, "episode_return": episode_return},
            )
            print(f"checkpoint_path={checkpoint_path}")

    runtime_seconds = time.time() - start
    model_path = output_dir / f"{run_name}.pt"
    agent.save(model_path, replay_buffer=replay_buffer, extra={"step": args.timesteps})
    config_path = output_dir / f"{run_name}_config.json"
    config_payload = {
        "droq_config": asdict(config),
        "env_kwargs": {k: v for k, v in env_kwargs.items() if k != "reward_config"},
        "reward_config": asdict(reward_config),
        "reward_profile": args.reward_profile,
    }
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    deterministic_eval = evaluate_agent(agent, env_kwargs, deterministic=True)
    stochastic_eval = evaluate_agent(agent, env_kwargs, deterministic=False)
    recent_losses = losses[-100:] if losses else []
    loss_summary = {
        key: float(np.mean([item[key] for item in recent_losses])) if recent_losses else None
        for key in ("critic_loss", "actor_loss", "alpha_loss", "alpha", "mean_q")
    }
    summary = {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "runtime_seconds": runtime_seconds,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "lookahead": args.lookahead,
        "curriculum": args.curriculum,
        "sequence_pitches": env.sequence_pitches,
        "sequence_sampling_weights": env.sequence_sampling_weights,
        "sequence_timing_profile": args.sequence_timing_profile,
        "action_mode": args.action_mode,
        "action_repeat": args.action_repeat,
        "reward_profile": args.reward_profile,
        "checkpoint_freq": args.checkpoint_freq,
        "loss_summary": loss_summary,
        "deterministic_eval": deterministic_eval,
        "stochastic_eval": stochastic_eval,
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"model_path={model_path}")
    print(f"config_path={config_path}")
    print(f"summary_path={summary_path}")
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print(f"loss_summary={loss_summary}")
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
