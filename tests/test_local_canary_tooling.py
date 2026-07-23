import importlib.util
import sys
from pathlib import Path


ROOT = Path("/home/reece_dev/msc-audio-pianist")


def _load_script(name: str):
    script_dir = ROOT / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        path = script_dir / name
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_dir))


def test_reward_scale_audit_masks_exclude_current_target():
    audit = _load_script("audit_cleanliness_reward_scale.py")
    cfg = audit.reward_config_from_profile("transition_cleanup_sensitive_v1")
    mask = audit.synthetic_mask_audit(cfg)
    target_components = mask["target_only_components"]
    assert mask["classifications"]["current_target"] == "excluded_from_unintended_components"
    assert target_components["unintended_continuous_travel"] == 0.0
    assert target_components["unintended_pressed_event_count"] == 0.0
    assert target_components["unintended_integrated_duration"] == 0.0
    assert mask["max_possible_added_penalty_for_one_unrelated_key_one_timestep"] > 0.0


def test_canary_evaluator_summarizes_required_metrics():
    canary = _load_script("evaluate_cleanliness_canary.py")
    rows = [
        {
            "checkpoint_step": 5000,
            "pressed_key_f1": 1.0,
            "timestep_f1": 0.5,
            "max_unintended_key_state": 0.2,
            "integrated_unintended_travel": 2.0,
            "shaped_return": 3.0,
        },
        {
            "checkpoint_step": 5000,
            "pressed_key_f1": 0.0,
            "timestep_f1": 0.25,
            "max_unintended_key_state": 0.6,
            "integrated_unintended_travel": 4.0,
            "shaped_return": 1.0,
        },
    ]
    summary = canary.summarize(rows)
    assert summary == [
        {
            "checkpoint_step": 5000,
            "mean_pressed_key_f1": 0.5,
            "mean_timestep_f1": 0.375,
            "mean_max_unintended": 0.4,
            "mean_integrated_unintended_travel": 3.0,
            "mean_shaped_return": 2.0,
        }
    ]
