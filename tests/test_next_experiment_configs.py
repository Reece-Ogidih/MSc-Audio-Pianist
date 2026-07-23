import json
from pathlib import Path

from ala_pianist.music import load_staged_curriculum, phase_for_global_step


ROOT = Path("/home/reece_dev/msc-audio-pianist")


def test_five_note_staged_curriculum_global_steps():
    phases = load_staged_curriculum(ROOT / "configs/droq_five_note_expansion_from_300k_seed13.json")
    assert phase_for_global_step(phases, 300000).name == "phase1_boundary_single_anchors"
    assert phase_for_global_step(phases, 499999).name == "phase1_boundary_single_anchors"
    assert phase_for_global_step(phases, 500000).name == "phase2_boundary_transitions"
    assert phase_for_global_step(phases, 799999).name == "phase2_boundary_transitions"
    assert phase_for_global_step(phases, 800000).name == "phase3_short_compositions"
    assert phase_for_global_step(phases, 1000000).name == "phase3_short_compositions"


def test_sac_fair_config_matches_benchmark():
    payload = json.loads((ROOT / "configs/sac_fair_six_sequence_seed13_1m.json").read_text())
    assert payload["seed"] == 13
    assert payload["timesteps"] == 1000000
    assert payload["sequence_pitches"] == [[73], [74], [75], [73, 75], [75, 73], [74, 75]]
    assert payload["sequence_sampling_weights"] == [0.22, 0.22, 0.22, 0.14, 0.10, 0.10]
    assert payload["action_mode"] == "direct"
    assert payload["action_repeat"] == 1
    assert payload["reward_profile"] == "transition_cleanup"
    assert payload["expected_observation_dim"] == 301
    assert payload["expected_action_dim"] == 22


def test_cleanliness_config_is_fresh_start_and_parameterised():
    payload = json.loads((ROOT / "configs/droq_cleanliness_sensitive_v1_seed13_300k.json").read_text())
    assert payload["fresh_start_required"] is True
    assert payload["reward_profile"] == "transition_cleanup_sensitive_v1"
    assert payload["timesteps"] == 300000
    for key in [
        "unintended_soft_threshold",
        "unintended_travel_weight",
        "unintended_near_press_weight",
        "unintended_press_weight",
        "late_release_weight",
        "early_activation_weight",
        "duration_weight",
        "press_threshold",
    ]:
        assert key in payload["reward_parameters"]


def test_hex_launch_scripts_have_strict_mode_and_env_inputs():
    scripts = [
        "run_sac_fair_1m.sh",
        "run_droq_cleanliness_300k.sh",
        "run_droq_five_note_expansion.sh",
        "verify_hex_inputs.sh",
    ]
    for script in scripts:
        text = (ROOT / "scripts/hex" / script).read_text()
        assert "set -euo pipefail" in text
        assert "HEX_GPU_INDEX" in text or script == "verify_hex_inputs.sh"
        assert "HEX_SCRATCH" in text
        assert "HEX_IMAGE_TAG" in text
