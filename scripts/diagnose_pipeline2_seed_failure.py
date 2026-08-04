#!/usr/bin/env python3
"""Activation diagnostics for successful vs collapsed Pipeline 2 direct policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ala_pianist.evaluation.direct_audio import build_clip_selection, build_pipeline2_evaluation_env
from ala_pianist.evaluation.final_experiments import write_csv, write_json
from ala_pianist.rl import DirectDroQAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed13-checkpoint", type=Path, required=True)
    parser.add_argument("--seed37-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-audio-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    audio_root = args.evaluation_audio_root or args.output_dir / "diagnostic_audio"
    env = build_pipeline2_evaluation_env(generated_root=audio_root, seed=5151)
    selection = build_clip_selection(env)
    rows = []
    for label, checkpoint in (("seed13", args.seed13_checkpoint), ("seed37", args.seed37_checkpoint)):
        agent = DirectDroQAgent.load(checkpoint, device=args.device)
        rows.append(diagnose_agent(label, checkpoint, agent, env, selection))
    write_csv(args.output_dir / "seed_failure_diagnostics.csv", rows)
    write_json(
        args.output_dir / "seed_failure_diagnostics_summary.json",
        {
            "rows": rows,
            "interpretation_limits": "Activation statistics can identify obvious collapse/suppression patterns but do not prove causal training failure mechanisms.",
        },
    )
    for row in rows:
        print(
            f"{row['model_id']} latent_var={row['audio_latent_variance_mean']:.6f} "
            f"action_diff_mismatch={row['mean_abs_action_diff_correct_mismatched']:.6f}"
        )
    print(f"output_dir={args.output_dir}")
    print("PIPELINE2_SEED_DIAGNOSTIC_COMPLETE=true")


def diagnose_agent(label: str, checkpoint: Path, agent: DirectDroQAgent, env, selection) -> dict:
    latents = []
    zero_diffs = []
    mismatch_diffs = []
    latent_zero_diffs = []
    latent_mismatch_diffs = []
    action_norms = []
    saturation = []
    with torch.no_grad():
        for index, sequence in enumerate(env.sequences):
            clip_index = selection.sequence_to_clip_index[tuple(sequence)]
            obs, _ = env.reset_to_clip_index(clip_index, seed=6000 + index)
            mismatch_id = selection.mismatch_clip_id_by_sequence[tuple(sequence)]
            correct = env.observation_for_audio_mode(obs, mode="correct")
            zero = env.observation_for_audio_mode(obs, mode="zero")
            mismatch = env.observation_for_audio_mode(obs, mode="mismatched", mismatched_clip_id=mismatch_id)
            correct_latent = latent(agent, correct)
            zero_latent = latent(agent, zero)
            mismatch_latent = latent(agent, mismatch)
            latents.append(correct_latent)
            latent_zero_diffs.append(np.abs(correct_latent - zero_latent))
            latent_mismatch_diffs.append(np.abs(correct_latent - mismatch_latent))
            correct_action = agent.act(correct, deterministic=True)
            zero_action = agent.act(zero, deterministic=True)
            mismatch_action = agent.act(mismatch, deterministic=True)
            zero_diffs.append(np.abs(correct_action - zero_action))
            mismatch_diffs.append(np.abs(correct_action - mismatch_action))
            action_norms.append(float(np.linalg.norm(correct_action)))
            saturation.append(float(np.mean(np.abs(correct_action) > 0.95)))
    latent_array = np.stack(latents)
    first_linear = agent.actor.fusion[0]
    audio_dim = agent.config.audio_latent_dim
    audio_weight_norm = float(first_linear.weight[:, :audio_dim].norm().detach().cpu())
    physical_weight_norm = float(first_linear.weight[:, audio_dim:].norm().detach().cpu())
    param_norms = {
        name: float(param.detach().norm().cpu())
        for name, param in agent.actor.named_parameters()
        if any(token in name for token in ("audio_encoder", "fusion.0", "mean"))
    }
    return {
        "model_id": label,
        "checkpoint_path": str(checkpoint),
        "audio_latent_variance_mean": float(np.var(latent_array, axis=0).mean()),
        "audio_latent_norm_mean": float(np.linalg.norm(latent_array, axis=1).mean()),
        "mean_abs_latent_diff_correct_zero": float(np.mean(np.stack(latent_zero_diffs))),
        "mean_abs_latent_diff_correct_mismatched": float(np.mean(np.stack(latent_mismatch_diffs))),
        "mean_abs_action_diff_correct_zero": float(np.mean(np.stack(zero_diffs))),
        "max_abs_action_diff_correct_zero": float(np.max(np.stack(zero_diffs))),
        "mean_abs_action_diff_correct_mismatched": float(np.mean(np.stack(mismatch_diffs))),
        "max_abs_action_diff_correct_mismatched": float(np.max(np.stack(mismatch_diffs))),
        "actor_action_norm_mean": float(np.mean(action_norms)),
        "actor_action_saturation_mean": float(np.mean(saturation)),
        "fusion_audio_weight_norm": audio_weight_norm,
        "fusion_physical_weight_norm": physical_weight_norm,
        "fusion_audio_to_physical_weight_norm_ratio": audio_weight_norm / max(physical_weight_norm, 1e-9),
        "selected_parameter_norms": param_norms,
    }


def latent(agent: DirectDroQAgent, obs: dict) -> np.ndarray:
    audio, _physical = agent._obs_tensors(obs)
    return agent.actor.audio_encoder(audio).squeeze(0).detach().cpu().numpy()


if __name__ == "__main__":
    main()
