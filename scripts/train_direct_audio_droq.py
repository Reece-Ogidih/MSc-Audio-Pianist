"""Short local smoke and future launcher scaffold for Pipeline 2 direct DroQ."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from ala_pianist.rl import (
    DirectAudioGoalEnv,
    DirectDroQAgent,
    DirectDroQConfig,
    IndexedDirectReplayBuffer,
    indexed_replay_from_checkpoint,
    load_direct_droq_checkpoint,
    restore_direct_rng_state,
    set_direct_droq_seed,
)
from ala_pianist.pipelines.indirect import BENCHMARK_SEQUENCE_PITCHES
from train_general_one_hand_policy import reward_config_from_profile


ROOT = Path("/home/reece_dev/msc-audio-pianist")


def parse_sequences(raw: str | None):
    if not raw:
        return BENCHMARK_SEQUENCE_PITCHES
    return tuple(tuple(int(part) for part in chunk.split(",")) for chunk in raw.split(";") if chunk)


def parse_weights(raw: str | None):
    if not raw:
        return None
    return tuple(float(part) for part in raw.split(",") if part)


def parse_steps(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--additional-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--stage-name", default="pipeline2_direct_audio_droq_smoke")
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "pipeline2_direct_audio"))
    parser.add_argument("--generated-root", default=str(ROOT / "experiments" / "pipeline2_direct_audio" / "audio_bank"))
    parser.add_argument("--sequence-pitches", default=None)
    parser.add_argument("--sequence-sampling-weights", default=None)
    parser.add_argument("--variants-per-sequence", type=int, default=1)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--past-context", type=float, default=0.10)
    parser.add_argument("--future-context", type=float, default=0.40)
    parser.add_argument("--horizon-steps", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--utd-ratio", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--lightweight-checkpoint-steps", default="")
    parser.add_argument("--full-checkpoint-steps", default="")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.resume_checkpoint and args.additional_timesteps is None:
        raise ValueError("--resume-checkpoint requires --additional-timesteps.")
    if args.resume_checkpoint and args.timesteps != 1000:
        raise ValueError("Use --additional-timesteps instead of --timesteps when resuming.")
    if args.additional_timesteps is not None and not args.resume_checkpoint:
        raise ValueError("--additional-timesteps requires --resume-checkpoint.")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {args.device!r}, but torch.cuda.is_available() is False.")

    set_direct_droq_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_utc = utc_now()
    checkpoint_payload = (
        load_direct_droq_checkpoint(args.resume_checkpoint, device=args.device)
        if args.resume_checkpoint
        else None
    )
    env = DirectAudioGoalEnv(
        generated_root=args.generated_root,
        sequences=parse_sequences(args.sequence_pitches),
        sequence_sampling_weights=parse_weights(args.sequence_sampling_weights),
        audio_sample_rate=args.audio_sample_rate,
        past_context_seconds=args.past_context,
        future_context_seconds=args.future_context,
        sequence_timing_profile="aligned",
        lookahead=1,
        horizon_steps=args.horizon_steps,
        reward_config=reward_config_from_profile("transition_cleanup_sensitive_v1"),
        seed=args.seed,
        variants_per_sequence=args.variants_per_sequence,
    )
    env.assert_no_forbidden_observation_fields()
    observation, info = env.reset(seed=args.seed)
    if checkpoint_payload:
        config_payload = dict(checkpoint_payload["config"])
        config_payload["device"] = args.device
        config = DirectDroQConfig(**config_payload)
        if config.audio_window_size != int(env.observation_space["audio"].shape[0]):
            raise ValueError("Resume checkpoint audio_window_size does not match current env.")
        if config.physical_dim != int(env.observation_space["physical"].shape[0]):
            raise ValueError("Resume checkpoint physical_dim does not match current env.")
        if config.action_dim != int(env.action_space.shape[0]):
            raise ValueError("Resume checkpoint action_dim does not match current env.")
        agent = DirectDroQAgent.load(args.resume_checkpoint, device=args.device)
        replay = indexed_replay_from_checkpoint(
            checkpoint_payload,
            physical_dim=config.physical_dim,
            action_dim=config.action_dim,
            fallback_capacity=config.buffer_size,
        )
        restore_direct_rng_state(checkpoint_payload.get("rng_state"))
        start_step = int(checkpoint_payload.get("extra", {}).get("step", 0))
        steps_to_run = int(args.additional_timesteps)
        resume_semantics = "full_training_state_resume"
    else:
        config = DirectDroQConfig(
            audio_window_size=int(env.observation_space["audio"].shape[0]),
            physical_dim=int(env.observation_space["physical"].shape[0]),
            action_dim=int(env.action_space.shape[0]),
            batch_size=args.batch_size,
            utd_ratio=args.utd_ratio,
            buffer_size=args.buffer_size,
            device=args.device,
        )
        agent = DirectDroQAgent(config)
        replay = IndexedDirectReplayBuffer(
            physical_dim=config.physical_dim,
            action_dim=config.action_dim,
            capacity=args.buffer_size,
        )
        start_step = 0
        steps_to_run = int(args.timesteps)
        resume_semantics = "fresh_actor_critics_optimizers_replay_rng_from_seed"
    lightweight_steps = parse_steps(args.lightweight_checkpoint_steps)
    full_steps = parse_steps(args.full_checkpoint_steps)
    checkpoint_dir = output_dir / "checkpoints"
    lightweight_dir = output_dir / "lightweight_checkpoints"
    losses = []
    print(f"resume_semantics={resume_semantics}")
    print(f"start_step={start_step}")
    print(f"steps_to_run={steps_to_run}")
    print(f"final_target_step={start_step + steps_to_run}")
    start = time.time()
    last_info = info
    last_full_checkpoint_path: Path | None = None
    for local_step in range(1, steps_to_run + 1):
        global_step = start_step + local_step
        metadata = env.replay_metadata()
        if replay.size < args.learning_starts:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(observation, deterministic=False)
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_metadata = env.replay_metadata()
        done = bool(terminated or truncated)
        replay.add(observation, metadata, action, reward, next_observation, next_metadata, done)
        observation = next_observation
        last_info = info
        if replay.size >= args.batch_size and replay.size > args.learning_starts:
            for _ in range(args.utd_ratio):
                losses.append(agent.update(replay.sample(args.batch_size, bank=env.audio_bank, device=agent.device)))
        if done:
            observation, last_info = env.reset()
        if global_step in lightweight_steps:
            lightweight_path = lightweight_dir / f"checkpoint_{global_step}_steps.pt"
            agent.save_lightweight(
                lightweight_path,
                extra={
                    "step": global_step,
                    "stage_name": args.stage_name,
                    "seed": args.seed,
                    "checkpoint_class": "lightweight_policy",
                    "policy_observation_fields": ("audio", "physical"),
                    "audio_bank_provenance": _audio_bank_provenance(env),
                },
            )
            print(f"lightweight_checkpoint_path={lightweight_path}")
        if global_step in full_steps:
            full_path = checkpoint_dir / f"full_checkpoint_{global_step}_steps.pt"
            agent.save(
                full_path,
                replay_buffer=replay,
                extra={
                    "step": global_step,
                    "stage_name": args.stage_name,
                    "seed": args.seed,
                    "checkpoint_class": "full_resumable",
                    "resume_semantics": "full_training_state_resume",
                    "policy_observation_fields": ("audio", "physical"),
                    "physical_observation_names": env.physical_observation_names,
                    "audio_bank_provenance": _audio_bank_provenance(env),
                    "training_config": vars(args),
                },
            )
            last_full_checkpoint_path = full_path
            print(f"full_checkpoint_path={full_path}")

    runtime_seconds = time.time() - start
    final_step = start_step + steps_to_run
    if final_step in full_steps and last_full_checkpoint_path is not None:
        checkpoint_path = last_full_checkpoint_path
    else:
        checkpoint_path = output_dir / f"{args.stage_name}_checkpoint.pt"
        agent.save(
            checkpoint_path,
            replay_buffer=replay,
            extra={
                "step": int(final_step),
                "stage_name": args.stage_name,
                "seed": args.seed,
                "checkpoint_class": "full_resumable",
                "resume_semantics": "full_training_state_resume",
                "policy_observation_fields": ("audio", "physical"),
                "physical_observation_names": env.physical_observation_names,
                "audio_bank_size": len(env.audio_bank),
                "audio_bank_provenance": _audio_bank_provenance(env),
                "audio_window_size": config.audio_window_size,
                "selected_training_route": "DIRECT_RL",
                "training_config": vars(args),
            },
        )
    reloaded = DirectDroQAgent.load(checkpoint_path, device=args.device)
    eval_action = reloaded.act(observation, deterministic=True)
    loss_summary = {
        key: float(np.mean([loss[key] for loss in losses[-20:]])) if losses else None
        for key in ("critic_loss", "actor_loss", "alpha_loss", "alpha", "mean_q")
    }
    summary = {
        "selected_training_route": "DIRECT_RL",
        "start_utc": start_utc,
        "end_utc": utc_now(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "seed": int(args.seed),
        "start_step": int(start_step),
        "timesteps": int(steps_to_run),
        "final_step": int(final_step),
        "resume_checkpoint": args.resume_checkpoint,
        "resume_semantics": resume_semantics,
        "runtime_seconds": runtime_seconds,
        "throughput_steps_per_second": float(steps_to_run / max(runtime_seconds, 1e-9)),
        "checkpoint_path": str(checkpoint_path),
        "lightweight_checkpoint_steps": sorted(lightweight_steps),
        "full_checkpoint_steps": sorted(full_steps),
        "loss_summary": loss_summary,
        "replay_size": replay.size,
        "replay_schema": "indexed_audio_references",
        "estimated_replay_bytes_for_1m": replay.estimated_bytes(1_000_000),
        "estimated_replay_bytes_for_2m": replay.estimated_bytes(2_000_000),
        "audio_window_shape": list(observation["audio"].shape),
        "physical_shape": list(observation["physical"].shape),
        "action_shape": list(eval_action.shape),
        "last_info": {
            "sequence": list(last_info.get("sequence", ())),
            "pressed_keys": list(last_info.get("pressed_keys", ())),
            "hidden_target_keys": list(last_info.get("hidden_target_keys", ())),
        },
        "audio_bank_provenance": _audio_bank_provenance(env),
        "sequence_distribution": {
            "sequences": [list(seq) for seq in parse_sequences(args.sequence_pitches)],
            "weights": list(parse_weights(args.sequence_sampling_weights) or []),
        },
        "architecture": {
            "audio": "raw waveform -> Conv1D stack -> GRU -> 128-D latent",
            "fusion": "audio latent + 118-D physical state -> MLP",
            "action_dim": 22,
        },
        "config": asdict(config),
    }
    summary_path = output_dir / f"{args.stage_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"selected_training_route=DIRECT_RL")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"summary_path={summary_path}")
    print(f"runtime_seconds={runtime_seconds:.2f}")
    print(f"loss_summary={loss_summary}")
    print(f"audio_shape={observation['audio'].shape}")
    print(f"physical_shape={observation['physical'].shape}")
    print(f"action_shape={eval_action.shape}")
    print(f"replay_schema=indexed_audio_references")


def _audio_bank_provenance(env: DirectAudioGoalEnv) -> dict:
    variants_per_sequence = {
        "_".join(str(pitch) for pitch in sequence): len(indices)
        for sequence, indices in zip(env.sequences, env._sequence_clip_indices, strict=True)
    }
    return {
        "sample_rate": env.audio_bank.sample_rate,
        "past_context_seconds": env.audio_bank.past_context_seconds,
        "future_context_seconds": env.audio_bank.future_context_seconds,
        "window_size": env.audio_bank.window_size,
        "clip_count": len(env.audio_bank),
        "logical_sequence_count": len(env.sequences),
        "training_variant_counts_per_sequence": variants_per_sequence,
        "training_sampling_policy": "sample logical sequence by sequence_sampling_weights, then sample one train split acoustic variant uniformly",
        "clips": [
            {
                "clip_id": clip.clip_id,
                "sequence": list(clip.sequence),
                "variant_index": clip.variant_index,
                "velocity": clip.velocity,
                "gain": clip.gain,
                "split": clip.split,
                "midi_path": str(clip.midi_path),
                "wav_path": str(clip.wav_path),
            }
            for clip in env.clips
        ],
    }


if __name__ == "__main__":
    main()
