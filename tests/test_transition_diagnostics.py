import importlib.util
from pathlib import Path


def _diagnostic_module():
    script_path = Path("/home/reece_dev/msc-audio-pianist/scripts/diagnose_transition_cleanup.py")
    spec = importlib.util.spec_from_file_location("diagnose_transition_cleanup", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_transition_phase_classifier():
    module = _diagnostic_module()

    assert module.classify_phase((52,), ()) == "csharp5_target_window"
    assert module.classify_phase((54,), ()) == "dsharp5_target_window"
    assert module.classify_phase((), (52,)) == "release_after_csharp5"
    assert module.classify_phase((), (54,)) == "release_after_dsharp5"


def test_transition_note_windows_are_deterministic():
    module = _diagnostic_module()

    windows = module.note_windows([73, 75])

    assert windows[0]["pitch"] == 73
    assert windows[0]["key_index"] == 52
    assert windows[1]["pitch"] == 75
    assert windows[1]["key_index"] == 54
    assert windows[1]["start_seconds"] > windows[0]["start_seconds"]
