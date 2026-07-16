import numpy as np

from ala_pianist.music import (
    assign_right_hand_fingering,
    generate_curriculum_events,
    write_curriculum_midi,
)
from ala_pianist.rl import GeneralOneHandGoalEnv, GeneralRewardConfig


def test_curriculum_generates_local_fingered_midi(tmp_path):
    clip = write_curriculum_midi(
        tmp_path / "clip.mid",
        mode="two_note_transitions",
        midi_min=73,
        midi_max=75,
        seed=7,
        clip_index=0,
    )

    assert clip.midi_path.exists()
    assert all(73 <= event.pitch <= 75 for event in clip.events)
    assert all(event.fingering is not None for event in clip.events)
    assert all(0 <= event.fingering <= 4 for event in clip.events)
    assert clip.key_indices == tuple(pitch - 21 for pitch in clip.pitches)


def test_curriculum_modes_are_deterministic():
    first = generate_curriculum_events(
        mode="short_phrases",
        midi_min=69,
        midi_max=75,
        seed=5,
        clip_index=2,
    )
    second = generate_curriculum_events(
        mode="short_phrases",
        midi_min=69,
        midi_max=75,
        seed=5,
        clip_index=2,
    )

    assert first == second
    assert assign_right_hand_fingering(69, 69, 75) == 0
    assert assign_right_hand_fingering(75, 69, 75) == 4


def test_curriculum_modes_respect_configured_range():
    for mode in ("repeated_notes", "two_note_transitions"):
        events = generate_curriculum_events(
            mode=mode,
            midi_min=73,
            midi_max=75,
            seed=5,
            clip_index=4,
        )
        assert events
        assert all(73 <= event.pitch <= 75 for event in events)


def test_general_one_hand_env_reset_and_random_step(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_min=73,
        midi_max=75,
        seed=3,
        lookahead=1,
        horizon_steps=2,
        reward_config=GeneralRewardConfig(fingering_weight=0.1),
    )

    obs, info = env.reset(seed=3)
    assert env.action_space.shape == (22,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    assert env.native_goal_shape == (178,)
    assert env.observation_space.contains(obs)
    assert info["lookahead"] == 1
    assert info["sustain_state"] == 0.0
    assert info["sampled_midi_pitch"] in (73, 74, 75)
    assert info["sampled_midi_pitch"] - 21 in info["target_keys"]
    assert info["trajectory_quality"] in {
        "gold_demo_candidate",
        "weak_demo_candidate",
        "not_demo_candidate",
    }

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["sustain_state"] == 0.0
    assert "reward_components" in info
    assert "target_key_state" in info["reward_components"]
    assert "max_unintended_key_state" in info["reward_components"]


def test_single_note_curriculum_cycles_across_resets(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_min=73,
        midi_max=75,
        seed=11,
        lookahead=1,
        horizon_steps=1,
    )

    sampled = []
    target_keys = []
    for index in range(6):
        _, info = env.reset(seed=11 if index == 0 else None)
        sampled.append(info["sampled_midi_pitch"])
        target_keys.extend(info["target_keys"])
        assert info["sampled_midi_pitch"] - 21 in info["target_keys"]
        assert info["native_goal_shape"] == (178,)

    assert set(sampled) == {73, 74, 75}
    assert set(target_keys) == {52, 53, 54}


def test_explicit_pitch_curriculum_cycles_only_requested_notes(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_min=73,
        midi_max=75,
        midi_pitches=(73, 75),
        seed=11,
        lookahead=1,
        horizon_steps=1,
    )

    sampled = []
    for index in range(6):
        _, info = env.reset(seed=11 if index == 0 else None)
        sampled.append(info["sampled_midi_pitch"])
        assert info["sampled_midi_pitch"] - 21 in info["target_keys"]

    assert sampled == [73, 75, 73, 75, 73, 75]


def test_weighted_pitch_sampling_biases_reset_distribution(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_pitches=(73, 75),
        pitch_sampling_weights=(0.7, 0.3),
        seed=11,
        lookahead=1,
        horizon_steps=1,
    )

    sampled = [env.reset(seed=11 if index == 0 else None)[1]["sampled_midi_pitch"] for index in range(100)]

    assert sampled.count(73) == 70
    assert sampled.count(75) == 30


def test_curriculum_cycle_is_deterministic_with_seed(tmp_path):
    kwargs = dict(
        curriculum="single_notes",
        midi_min=73,
        midi_max=75,
        seed=13,
        lookahead=1,
        horizon_steps=1,
    )
    env_a = GeneralOneHandGoalEnv(generated_midi_dir=tmp_path / "a", **kwargs)
    env_b = GeneralOneHandGoalEnv(generated_midi_dir=tmp_path / "b", **kwargs)

    seq_a = [env_a.reset(seed=13 if i == 0 else None)[1]["sampled_midi_pitch"] for i in range(6)]
    seq_b = [env_b.reset(seed=13 if i == 0 else None)[1]["sampled_midi_pitch"] for i in range(6)]

    assert seq_a == seq_b == [73, 74, 75, 73, 74, 75]


def test_general_one_hand_env_lookahead_changes_shape(tmp_path):
    env1 = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path / "l1",
        curriculum="single_notes",
        midi_min=73,
        midi_max=75,
        lookahead=1,
    )
    env2 = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path / "l2",
        curriculum="single_notes",
        midi_min=73,
        midi_max=75,
        lookahead=2,
    )

    assert env1.native_goal_shape == (178,)
    assert env2.native_goal_shape == (267,)
    assert env2.observation_space.shape[0] > env1.observation_space.shape[0]


def test_general_one_hand_env_rescales_normalized_actions(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_min=73,
        midi_max=75,
        lookahead=1,
    )
    native_low, native_high = env.native_action_bounds

    assert np.allclose(env.rescale_action(np.full(22, -1.0, dtype=np.float32)), native_low)
    assert np.allclose(env.rescale_action(np.full(22, 1.0, dtype=np.float32)), native_high)
    midpoint = env.rescale_action(np.zeros(22, dtype=np.float32))
    assert np.all(midpoint >= native_low)
    assert np.all(midpoint <= native_high)


def test_direct_action_mode_preserves_single_internal_step(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_pitches=(73,),
        lookahead=1,
        horizon_steps=4,
        action_mode="direct",
        action_repeat=4,
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros(22, dtype=np.float32))

    assert info["action_mode"] == "direct"
    assert info["internal_steps"] == 1


def test_hold_action_mode_repeats_internal_steps(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_pitches=(73,),
        lookahead=1,
        horizon_steps=4,
        action_mode="hold",
        action_repeat=3,
    )
    env.reset()
    _, reward, terminated, truncated, info = env.step(np.zeros(22, dtype=np.float32))

    assert np.isfinite(reward)
    assert info["action_mode"] == "hold"
    assert info["action_repeat"] == 3
    assert info["internal_steps"] == 3
    assert not terminated
    assert not truncated


def test_hold_action_mode_respects_horizon(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_pitches=(73,),
        lookahead=1,
        horizon_steps=2,
        action_mode="hold",
        action_repeat=4,
    )
    env.reset()
    _, _, terminated, truncated, info = env.step(np.zeros(22, dtype=np.float32))

    assert info["internal_steps"] == 2
    assert terminated or truncated


def test_press_bonus_reward_component_exists(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_pitches=(73,),
        lookahead=1,
        horizon_steps=1,
        reward_config=GeneralRewardConfig(target_activation_bonus=5.0),
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros(22, dtype=np.float32))

    assert "target_activation" in info["reward_components"]
    assert "high_unintended" in info["reward_components"]


def test_cleanup_reward_components_are_logged(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_pitches=(73,),
        lookahead=1,
        horizon_steps=1,
        reward_config=GeneralRewardConfig(
            cleanup_gate_threshold=0.75,
            gated_unintended_weight=3.0,
            gated_wrong_pressed_weight=2.0,
            nearby_wrong_key_weight=2.0,
        ),
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros(22, dtype=np.float32))

    assert "cleanup_gate" in info["reward_components"]
    assert "nearby_wrong_key_state" in info["reward_components"]


def test_anti_coupling_reward_components_and_penalties(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_pitches=(73,),
        lookahead=1,
        horizon_steps=1,
        reward_config=GeneralRewardConfig(
            cleanup_gate_threshold=0.75,
            csharp_dsharp_key54_weight=5.0,
            dsharp_csharp_key52_weight=2.0,
        ),
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros(22, dtype=np.float32))

    components = info["reward_components"]
    assert "csharp_dsharp_key54_state" in components
    assert "dsharp_csharp_key52_state" in components

    base = dict(components)
    base.update(
        {
            "cleanup_gate": 1.0,
            "csharp_dsharp_key54_state": 0.0,
            "dsharp_csharp_key52_state": 0.0,
            "csharp_dsharp_key54_pressed": 0.0,
            "dsharp_csharp_key52_pressed": 0.0,
        }
    )
    csharp_penalized = dict(base, csharp_dsharp_key54_state=0.5)
    dsharp_penalized = dict(base, dsharp_csharp_key52_state=0.5)

    assert env._combine_reward_components(csharp_penalized) < env._combine_reward_components(base)
    assert env._combine_reward_components(dsharp_penalized) < env._combine_reward_components(base)
    assert (
        env._combine_reward_components(csharp_penalized)
        < env._combine_reward_components(dsharp_penalized)
    )
