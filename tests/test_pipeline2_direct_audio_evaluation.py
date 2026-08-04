from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ala_pianist.audio import AudioReferenceBank
from ala_pianist.evaluation.direct_audio import (
    aggregate_checkpoint_rows,
    build_clip_selection,
    checkpoint_step,
    discover_lightweight_checkpoints,
    evaluate_checkpoint,
    write_evaluation_outputs,
)
from ala_pianist.music import write_sequence_midi
from ala_pianist.rl import DirectAudioClip, DirectAudioGoalEnv, DirectDroQAgent, DirectDroQConfig, GeneralRewardConfig


def _eval_bank(tmp_path: Path, sequences=((72,), (73,), (72, 73), (73, 72))):
    bank = AudioReferenceBank(sample_rate=16000, past_context_seconds=0.1, future_context_seconds=0.4)
    clips = []
    for seq_idx, sequence in enumerate(tuple(tuple(seq) for seq in sequences)):
        midi_path = tmp_path / f"seq_{seq_idx}.mid"
        write_sequence_midi(sequence, midi_path, midi_min=72, midi_max=76)
        waveform = np.full(16000, fill_value=(seq_idx + 1) / 20.0, dtype=np.float32)
        clip_id = bank.add_waveform(
            waveform,
            name=f"seq_{seq_idx}",
            source_sample_rate=16000,
            metadata={"sequence": sequence, "split": "canonical_eval"},
        )
        clips.append(
            DirectAudioClip(
                sequence=sequence,
                midi_path=midi_path,
                wav_path=tmp_path / f"seq_{seq_idx}.wav",
                clip_id=clip_id,
                variant_index=0,
                velocity=90,
                gain=0.5,
                split="canonical_eval",
            )
        )
    env = DirectAudioGoalEnv(
        audio_bank=bank,
        clips=tuple(clips),
        sequences=tuple(tuple(seq) for seq in sequences),
        sequence_sampling_weights=tuple(1.0 / len(sequences) for _ in sequences),
        reward_config=GeneralRewardConfig(),
        horizon_steps=4,
        seed=13,
        sampling_split="canonical_eval",
    )
    return env


def test_checkpoint_ordering_discovers_lightweight_checkpoints(tmp_path: Path) -> None:
    root = tmp_path / "run"
    ckpt_dir = root / "lightweight_checkpoints"
    ckpt_dir.mkdir(parents=True)
    for step in (50000, 10000, 25000):
        (ckpt_dir / f"checkpoint_{step}_steps.pt").write_bytes(b"fake")

    paths = discover_lightweight_checkpoints(root)

    assert [checkpoint_step(path) for path in paths] == [10000, 25000, 50000]


def test_lightweight_checkpoint_loading_and_deterministic_actor_eval(tmp_path: Path) -> None:
    env = _eval_bank(tmp_path)
    agent = DirectDroQAgent(DirectDroQConfig(audio_window_size=8000, physical_dim=118, action_dim=22))
    checkpoint = tmp_path / "checkpoint_10000_steps.pt"
    agent.save_lightweight(checkpoint, extra={"step": 10000})
    loaded = DirectDroQAgent.load(checkpoint)
    obs, _ = env.reset_to_clip_index(0, seed=13)

    first = loaded.act(obs, deterministic=True)
    second = loaded.act(obs, deterministic=True)

    np.testing.assert_allclose(first, second)
    assert first.shape == (22,)


def test_audio_override_modes_leave_underlying_task_unchanged(tmp_path: Path) -> None:
    env = _eval_bank(tmp_path)
    selection = build_clip_selection(env)
    obs, info = env.reset_to_clip_index(selection.sequence_to_clip_index[(72, 73)], seed=13)
    original_target = info["hidden_target_keys"]
    zero = env.observation_for_audio_mode(obs, mode="zero")
    mismatch = env.observation_for_audio_mode(
        obs,
        mode="mismatched",
        mismatched_clip_id=selection.mismatch_clip_id_by_sequence[(72, 73)],
    )
    after = env._info(env._base_env_for_clip_index(env._active_clip_index))

    assert after["hidden_target_keys"] == original_target
    np.testing.assert_allclose(zero["physical"], obs["physical"])
    np.testing.assert_allclose(mismatch["physical"], obs["physical"])
    assert np.allclose(zero["audio"], 0.0)
    assert not np.allclose(mismatch["audio"], obs["audio"])


def test_mismatched_mapping_is_deterministic_and_different(tmp_path: Path) -> None:
    env = _eval_bank(tmp_path)
    first = build_clip_selection(env)
    second = build_clip_selection(env)

    assert first.mismatch_mapping == second.mismatch_mapping
    for sequence, mismatch in first.mismatch_sequence_by_sequence.items():
        assert sequence != mismatch
        assert len(sequence) == len(mismatch)


def test_evaluation_observation_has_no_target_or_clip_leakage(tmp_path: Path) -> None:
    env = _eval_bank(tmp_path)
    obs, _ = env.reset_to_clip_index(0, seed=13)

    env.assert_no_forbidden_observation_fields()
    assert set(obs) == {"audio", "physical"}
    assert "clip_id" not in obs
    assert "sequence" not in obs
    assert "midi" not in obs


def test_evaluate_checkpoint_writes_required_metric_fields(tmp_path: Path) -> None:
    env = _eval_bank(tmp_path)
    selection = build_clip_selection(env)
    agent = DirectDroQAgent(DirectDroQConfig(audio_window_size=8000, physical_dim=118, action_dim=22))
    checkpoint = tmp_path / "checkpoint_10000_steps.pt"
    agent.save_lightweight(checkpoint, extra={"step": 10000})

    sequence_rows, action_rows = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        env=env,
        selection=selection,
        seed=13,
        device="cpu",
        audio_modes=("correct", "zero"),
    )

    assert sequence_rows
    assert action_rows
    row = sequence_rows[0]
    for key in (
        "pressed_key_precision",
        "pressed_key_recall",
        "pressed_key_f1",
        "timestep_f1",
        "max_unintended_key_state",
        "strict_outcome",
    ):
        assert key in row
    assert "mean_abs_action_diff_correct_zero" in action_rows[0]


def test_evaluation_artifact_serialization_and_summary_rows(tmp_path: Path) -> None:
    sequence_rows = [
        {
            "checkpoint_step": 10000,
            "audio_mode": "correct",
            "sequence_name": "anchor_72",
            "sequence_group": "anchor",
            "pressed_key_precision": 1.0,
            "pressed_key_recall": 1.0,
            "pressed_key_f1": 1.0,
            "timestep_f1": 0.5,
            "max_unintended_key_state": 0.0,
            "integrated_unintended_key_state": 0.0,
            "wrong_press_count": 0,
        },
        {
            "checkpoint_step": 10000,
            "audio_mode": "zero",
            "sequence_name": "anchor_72",
            "sequence_group": "anchor",
            "pressed_key_precision": 0.0,
            "pressed_key_recall": 0.0,
            "pressed_key_f1": 0.0,
            "timestep_f1": 0.0,
            "max_unintended_key_state": 0.0,
            "integrated_unintended_key_state": 0.0,
            "wrong_press_count": 0,
        },
    ]
    checkpoint_rows = aggregate_checkpoint_rows(sequence_rows)
    summary = write_evaluation_outputs(
        output_dir=tmp_path / "evaluation",
        checkpoint_rows=checkpoint_rows,
        sequence_rows=sequence_rows,
        action_rows=[],
        manifest={"test": True},
    )

    assert (tmp_path / "evaluation" / "evaluation_summary.json").exists()
    assert (tmp_path / "evaluation" / "evaluation_checkpoint_metrics.csv").exists()
    assert summary["best_checkpoint_by_correct_pressed_key_f1"]["checkpoint_step"] == 10000
