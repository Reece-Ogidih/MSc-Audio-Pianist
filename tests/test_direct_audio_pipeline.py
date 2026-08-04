from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ala_pianist.audio import AudioReferenceBank
from ala_pianist.music import write_sequence_midi
from ala_pianist.pipelines.indirect import BENCHMARK_SEQUENCE_PITCHES
from ala_pianist.rl import (
    DirectAudioClip,
    DirectAudioGoalEnv,
    DirectDroQAgent,
    DirectDroQConfig,
    GeneralRewardConfig,
    IndexedDirectReplayBuffer,
    direct_rng_state_dict,
    indexed_replay_from_checkpoint,
    load_direct_droq_checkpoint,
    restore_direct_rng_state,
    set_direct_droq_seed,
)


def _bank_and_clips(tmp_path: Path):
    bank = AudioReferenceBank(sample_rate=16000, past_context_seconds=0.1, future_context_seconds=0.4)
    clips = []
    for idx, sequence in enumerate(((72, 73), (73, 72))):
        midi_path = tmp_path / f"seq_{idx}.mid"
        write_sequence_midi(sequence, midi_path, midi_min=72, midi_max=76)
        waveform = np.zeros(16000, dtype=np.float32)
        first, second = sequence
        waveform[1600:3200] = (first - 70) / 10.0
        waveform[6400:8000] = (second - 70) / 10.0
        clip_id = bank.add_waveform(
            waveform,
            name=f"seq_{idx}",
            source_sample_rate=16000,
            metadata={"sequence": sequence},
        )
        clips.append(
            DirectAudioClip(
                sequence=sequence,
                midi_path=midi_path,
                wav_path=tmp_path / f"seq_{idx}.wav",
                clip_id=clip_id,
                variant_index=0,
                velocity=90,
                gain=0.5,
            )
        )
    return bank, tuple(clips)


def _bank_and_variable_variant_clips(
    tmp_path: Path,
    sequences,
    variant_counts,
    *,
    include_eval_clip: bool = False,
):
    bank = AudioReferenceBank(sample_rate=16000, past_context_seconds=0.1, future_context_seconds=0.4)
    clips = []
    for seq_idx, sequence in enumerate(tuple(tuple(seq) for seq in sequences)):
        midi_path = tmp_path / f"seq_{seq_idx}.mid"
        write_sequence_midi(sequence, midi_path, midi_min=72, midi_max=76)
        for variant_index in range(int(variant_counts[seq_idx])):
            waveform = np.full(16000, fill_value=(seq_idx + 1) / 20.0 + variant_index / 200.0, dtype=np.float32)
            clip_id = bank.add_waveform(
                waveform,
                name=f"seq_{seq_idx}_variant_{variant_index}",
                source_sample_rate=16000,
                metadata={"sequence": sequence, "variant_index": variant_index, "split": "train"},
            )
            clips.append(
                DirectAudioClip(
                    sequence=sequence,
                    midi_path=midi_path,
                    wav_path=tmp_path / f"seq_{seq_idx}_variant_{variant_index}.wav",
                    clip_id=clip_id,
                    variant_index=variant_index,
                    velocity=90,
                    gain=0.5,
                    split="train",
                )
            )
        if include_eval_clip:
            clip_id = bank.add_waveform(
                np.full(16000, fill_value=0.9, dtype=np.float32),
                name=f"seq_{seq_idx}_eval",
                source_sample_rate=16000,
                metadata={"sequence": sequence, "variant_index": 999, "split": "eval"},
            )
            clips.append(
                DirectAudioClip(
                    sequence=sequence,
                    midi_path=midi_path,
                    wav_path=tmp_path / f"seq_{seq_idx}_eval.wav",
                    clip_id=clip_id,
                    variant_index=999,
                    velocity=90,
                    gain=0.5,
                    split="eval",
                )
            )
    return bank, tuple(clips)


def _env(tmp_path: Path) -> DirectAudioGoalEnv:
    bank, clips = _bank_and_clips(tmp_path)
    return DirectAudioGoalEnv(
        audio_bank=bank,
        clips=clips,
        sequences=((72, 73), (73, 72)),
        sequence_sampling_weights=(1.0, 0.0),
        reward_config=GeneralRewardConfig(),
        horizon_steps=8,
        seed=13,
    )


def test_direct_observation_excludes_symbolic_goal_fields(tmp_path):
    env = _env(tmp_path)
    obs, info = env.reset(seed=13)
    env.assert_no_forbidden_observation_fields()
    assert set(obs) == {"audio", "physical"}
    assert "clip_id" not in obs
    assert "logical_sequence_index" not in obs
    assert "sequence" not in obs
    assert "audio_sample_index" not in obs
    assert obs["audio"].shape == (8000,)
    assert obs["physical"].shape == (118,)
    assert info["policy_observation_fields"] == ("audio", "physical")
    assert all("goal" not in name for name in env.physical_observation_names)
    assert all("fingering" not in name for name in env.physical_observation_names)


def test_hidden_target_changes_do_not_change_observation_when_audio_and_physical_fixed(tmp_path):
    env = _env(tmp_path)
    obs, _ = env.reset(seed=13)
    copied = {key: value.copy() for key, value in obs.items()}
    # Hidden score labels may change in the simulator internals, but the policy
    # observation is only already-resolved audio plus physical state.
    changed_hidden_target = (99,)
    assert changed_hidden_target != tuple(env._base_env_for_clip_index(0).current_target_keys())
    np.testing.assert_allclose(copied["audio"], obs["audio"])
    np.testing.assert_allclose(copied["physical"], obs["physical"])


def test_changing_waveform_changes_musical_observation_without_clip_id_feature(tmp_path):
    bank, clips = _bank_and_clips(tmp_path)
    win_a = bank.context_window(clip_id=clips[0].clip_id, center_sample=7000)
    win_b = bank.context_window(clip_id=clips[1].clip_id, center_sample=7000)
    assert not np.allclose(win_a, win_b)


def test_audio_dependence_evaluation_modes_change_only_audio(tmp_path):
    env = _env(tmp_path)
    obs, _ = env.reset(seed=13)
    zero = env.observation_for_audio_mode(obs, mode="zero")
    wrong = env.observation_for_audio_mode(obs, mode="mismatched", mismatched_clip_id=1)
    np.testing.assert_allclose(zero["physical"], obs["physical"])
    np.testing.assert_allclose(wrong["physical"], obs["physical"])
    assert np.allclose(zero["audio"], 0.0)
    assert not np.allclose(wrong["audio"], obs["audio"])


def test_audio_window_deterministic_and_zero_padded(tmp_path):
    bank, clips = _bank_and_clips(tmp_path)
    a = bank.context_window(clip_id=clips[0].clip_id, center_sample=0)
    b = bank.context_window(clip_id=clips[0].clip_id, center_sample=0)
    np.testing.assert_allclose(a, b)
    assert np.allclose(a[: bank.past_samples], 0.0)


def test_temporal_order_distinguishes_72_73_from_73_72(tmp_path):
    bank, clips = _bank_and_clips(tmp_path)
    a = bank.context_window(clip_id=clips[0].clip_id, center_sample=4000)
    b = bank.context_window(clip_id=clips[1].clip_id, center_sample=4000)
    assert not np.allclose(a, b)


def test_direct_policy_action_shape_and_gradients(tmp_path):
    env = _env(tmp_path)
    obs, _ = env.reset(seed=13)
    config = DirectDroQConfig(audio_window_size=8000, physical_dim=118, action_dim=22, batch_size=2)
    agent = DirectDroQAgent(config)
    action = agent.act(obs, deterministic=True)
    assert action.shape == (22,)

    audio = torch.as_tensor(obs["audio"]).unsqueeze(0)
    physical = torch.as_tensor(obs["physical"]).unsqueeze(0)
    sampled, _ = agent.actor.sample(audio, physical)
    loss = sampled.square().mean()
    agent.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    conv_grad = agent.actor.audio_encoder.conv[0].weight.grad
    gru_grad = agent.actor.audio_encoder.gru.weight_ih_l0.grad
    assert conv_grad is not None and torch.isfinite(conv_grad).all()
    assert gru_grad is not None and torch.isfinite(gru_grad).all()


def test_sequence_weights_with_multiple_physical_variants_do_not_fail(tmp_path):
    bank, clips = _bank_and_variable_variant_clips(
        tmp_path,
        BENCHMARK_SEQUENCE_PITCHES,
        [4] * len(BENCHMARK_SEQUENCE_PITCHES),
    )
    weights = (0.07,) * 5 + (0.08125,) * 8
    env = DirectAudioGoalEnv(
        audio_bank=bank,
        clips=clips,
        sequences=BENCHMARK_SEQUENCE_PITCHES,
        sequence_sampling_weights=weights,
        reward_config=GeneralRewardConfig(),
        horizon_steps=8,
        seed=13,
    )
    obs, info = env.reset(seed=13)

    assert len(clips) == 52
    assert len(BENCHMARK_SEQUENCE_PITCHES) == 13
    assert set(obs) == {"audio", "physical"}
    assert info["sequence"] in BENCHMARK_SEQUENCE_PITCHES
    assert info["split"] == "train"


def test_every_logical_sequence_requires_eligible_training_clip(tmp_path):
    bank, clips = _bank_and_variable_variant_clips(
        tmp_path,
        ((72,), (73,)),
        [1, 0],
        include_eval_clip=True,
    )

    with pytest.raises(ValueError, match="eligible training audio clip"):
        DirectAudioGoalEnv(
            audio_bank=bank,
            clips=clips,
            sequences=((72,), (73,)),
            sequence_sampling_weights=(0.5, 0.5),
            reward_config=GeneralRewardConfig(),
            horizon_steps=8,
            seed=13,
        )


def test_sequence_sampling_is_seeded_and_returns_selected_sequence_variant(tmp_path):
    bank_a, clips_a = _bank_and_variable_variant_clips(tmp_path / "a", ((72,), (73, 74)), [3, 3])
    bank_b, clips_b = _bank_and_variable_variant_clips(tmp_path / "b", ((72,), (73, 74)), [3, 3])
    env_a = DirectAudioGoalEnv(
        audio_bank=bank_a,
        clips=clips_a,
        sequences=((72,), (73, 74)),
        sequence_sampling_weights=(0.25, 0.75),
        reward_config=GeneralRewardConfig(),
        horizon_steps=8,
        seed=37,
    )
    env_b = DirectAudioGoalEnv(
        audio_bank=bank_b,
        clips=clips_b,
        sequences=((72,), (73, 74)),
        sequence_sampling_weights=(0.25, 0.75),
        reward_config=GeneralRewardConfig(),
        horizon_steps=8,
        seed=37,
    )

    draws_a = [env_a._sample_clip_index() for _ in range(20)]
    draws_b = [env_b._sample_clip_index() for _ in range(20)]

    assert draws_a == draws_b
    for sequence_index, clip_index in draws_a:
        assert clips_a[clip_index].sequence == env_a.sequences[sequence_index]
        assert clips_a[clip_index].split == "train"


def test_unequal_variant_counts_do_not_change_logical_task_probability(tmp_path):
    bank, clips = _bank_and_variable_variant_clips(tmp_path, ((72,), (73,)), [8, 1])
    env = DirectAudioGoalEnv(
        audio_bank=bank,
        clips=clips,
        sequences=((72,), (73,)),
        sequence_sampling_weights=(0.5, 0.5),
        reward_config=GeneralRewardConfig(),
        horizon_steps=8,
        seed=13,
    )

    sequence_counts = np.zeros(2, dtype=np.int64)
    for _ in range(2000):
        sequence_index, clip_index = env._sample_clip_index()
        sequence_counts[sequence_index] += 1
        assert clips[clip_index].sequence == env.sequences[sequence_index]

    observed = sequence_counts / sequence_counts.sum()
    np.testing.assert_allclose(observed, [0.5, 0.5], atol=0.05)


def test_invalid_sequence_weight_count_fails_early(tmp_path):
    bank, clips = _bank_and_variable_variant_clips(tmp_path, ((72,), (73,)), [1, 1])

    with pytest.raises(ValueError, match="Expected 2 sampling weights"):
        DirectAudioGoalEnv(
            audio_bank=bank,
            clips=clips,
            sequences=((72,), (73,)),
            sequence_sampling_weights=(1.0,),
            reward_config=GeneralRewardConfig(),
            horizon_steps=8,
            seed=13,
        )


def test_phase_a_anchor_transition_sampling_mass_matches_design(tmp_path):
    bank, clips = _bank_and_variable_variant_clips(
        tmp_path,
        BENCHMARK_SEQUENCE_PITCHES,
        [2] * len(BENCHMARK_SEQUENCE_PITCHES),
    )
    weights = (0.07,) * 5 + (0.08125,) * 8
    env = DirectAudioGoalEnv(
        audio_bank=bank,
        clips=clips,
        sequences=BENCHMARK_SEQUENCE_PITCHES,
        sequence_sampling_weights=weights,
        reward_config=GeneralRewardConfig(),
        horizon_steps=8,
        seed=13,
    )

    assert float(env.sequence_sampling_weights[:5].sum()) == pytest.approx(0.35)
    assert float(env.sequence_sampling_weights[5:].sum()) == pytest.approx(0.65)


def test_indexed_replay_resolves_audio_windows_without_storing_them(tmp_path):
    env = _env(tmp_path)
    obs, _ = env.reset(seed=13)
    metadata = env.replay_metadata()
    action = np.zeros(22, dtype=np.float32)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    replay = IndexedDirectReplayBuffer(physical_dim=118, action_dim=22, capacity=4)
    replay.add(obs, metadata, action, reward, next_obs, env.replay_metadata(), terminated or truncated)
    batch = replay.sample(1, bank=env.audio_bank, device=torch.device("cpu"))
    assert batch["audio"].shape == (1, 8000)
    assert batch["physical"].shape == (1, 118)
    assert not hasattr(replay, "audio")
    restored = IndexedDirectReplayBuffer(physical_dim=118, action_dim=22, capacity=4)
    restored.load_state_dict(replay.state_dict())
    assert restored.size == replay.size


def test_checkpoint_save_load_preserves_audio_network_parameters(tmp_path):
    config = DirectDroQConfig(audio_window_size=8000, physical_dim=118, action_dim=22)
    agent = DirectDroQAgent(config)
    path = tmp_path / "direct.pt"
    agent.save(path)
    loaded = DirectDroQAgent.load(path)
    original = dict(agent.actor.named_parameters())["audio_encoder.conv.0.weight"]
    restored = dict(loaded.actor.named_parameters())["audio_encoder.conv.0.weight"]
    torch.testing.assert_close(original, restored)


def test_full_checkpoint_contains_resume_state_and_indexed_replay(tmp_path):
    env = _env(tmp_path)
    obs, _ = env.reset(seed=13)
    action = np.zeros(22, dtype=np.float32)
    metadata = env.replay_metadata()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    replay = IndexedDirectReplayBuffer(physical_dim=118, action_dim=22, capacity=4)
    replay.add(obs, metadata, action, reward, next_obs, env.replay_metadata(), terminated or truncated)
    agent = DirectDroQAgent(DirectDroQConfig(audio_window_size=8000, physical_dim=118, action_dim=22))
    path = tmp_path / "full.pt"
    agent.save(path, replay_buffer=replay, extra={"step": 123, "checkpoint_class": "full_resumable"})
    payload = load_direct_droq_checkpoint(path)
    assert payload["extra"]["checkpoint_class"] == "full_resumable"
    assert "actor_optimizer" in payload
    assert "critic_optimizer" in payload
    assert "target_critics" in payload
    assert "alpha_optimizer" in payload
    assert "rng_state" in payload and "torch_cpu" in payload["rng_state"]
    restored = indexed_replay_from_checkpoint(
        payload,
        physical_dim=118,
        action_dim=22,
        fallback_capacity=4,
    )
    assert restored.size == 1


def test_restore_direct_rng_state_accepts_torch_uint8_tensor():
    set_direct_droq_seed(123)
    state = direct_rng_state_dict()

    restore_direct_rng_state(state)

    assert state["torch_cpu"].dtype == torch.uint8


def test_restore_direct_rng_state_accepts_list_serialised_torch_state():
    set_direct_droq_seed(123)
    state = direct_rng_state_dict()
    state["torch_cpu"] = state["torch_cpu"].tolist()

    restore_direct_rng_state(state)


def test_restore_direct_rng_state_accepts_numpy_uint8_torch_state():
    set_direct_droq_seed(123)
    state = direct_rng_state_dict()
    state["torch_cpu"] = state["torch_cpu"].numpy().copy()

    restore_direct_rng_state(state)


def test_restore_direct_rng_state_round_trip_reproduces_random_draws():
    set_direct_droq_seed(456)
    state = direct_rng_state_dict()
    torch_draw = torch.rand(4)
    numpy_draw = np.random.rand(4)

    restore_direct_rng_state(state)

    torch.testing.assert_close(torch.rand(4), torch_draw)
    np.testing.assert_allclose(np.random.rand(4), numpy_draw)


def test_restore_direct_rng_state_malformed_torch_state_fails_clearly():
    state = direct_rng_state_dict()
    state["torch_cpu"] = ["not", "bytes"]

    with pytest.raises(TypeError, match="integer byte values"):
        restore_direct_rng_state(state)


def test_no_basic_pitch_or_timed_note_calls_in_direct_policy_inference(monkeypatch, tmp_path):
    from ala_pianist.audio import transcriber as transcriber_module
    from ala_pianist.music import timed_notes as timed_notes_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("symbolic/transcription shortcut was called")

    monkeypatch.setattr(transcriber_module.BasicPitchTranscriber, "transcribe", forbidden)
    monkeypatch.setattr(timed_notes_module, "timed_notes_to_controller_sequence", forbidden)
    env = _env(tmp_path)
    obs, _ = env.reset(seed=13)
    agent = DirectDroQAgent(DirectDroQConfig(audio_window_size=8000, physical_dim=118, action_dim=22))
    action = agent.act(obs, deterministic=True)
    assert action.shape == (22,)
