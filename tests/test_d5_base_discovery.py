import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path("/home/reece_dev/msc-audio-pianist/scripts/find_d5_base_action.py")
_SPEC = importlib.util.spec_from_file_location("find_d5_base_action", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
outcome = _MODULE.outcome
score_result = _MODULE.score_result


def test_d5_base_discovery_outcome_and_score_helpers():
    clean = {
        "max_target_key_state": 1.0,
        "max_unintended_key_state": 0.0,
        "wrong_keys_pressed": [],
        "key_pressed": True,
    }
    dirty = {
        "max_target_key_state": 1.0,
        "max_unintended_key_state": 1.0,
        "wrong_keys_pressed": [54],
        "key_pressed": True,
    }

    assert outcome({53}, 1.0, 0.0) == "clean"
    assert outcome({53, 54}, 1.0, 1.0) == "dirty"
    assert outcome(set(), 0.3, 0.3) == "near_clean"
    assert outcome(set(), 0.1, 0.5) == "partial"
    assert score_result(clean) > score_result(dirty)
