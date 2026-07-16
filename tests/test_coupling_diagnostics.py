import importlib.util
from pathlib import Path

import numpy as np


def _diagnostic_module():
    script_path = Path("/home/reece_dev/msc-audio-pianist/scripts/diagnose_csharp_dsharp_coupling.py")
    spec = importlib.util.spec_from_file_location("diagnose_csharp_dsharp_coupling", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ZeroPolicy:
    def predict(self, obs, deterministic=True):
        del obs, deterministic
        return np.zeros(22, dtype=np.float32), None


def test_action_comparison_reports_similarity_fields():
    module = _diagnostic_module()
    actions_a = [[0.0] * 22, [0.5] * 22]
    actions_b = [[0.0] * 22, [-0.5] * 22]

    result = module.compare_actions(tuple(f"a{i}" for i in range(22)), actions_a, actions_b)

    assert result["aligned_steps"] == 2
    assert result["mean_l2_distance"] >= 0.0
    assert -1.0 <= result["mean_cosine_similarity"] <= 1.0
    assert result["top_differing_dimensions"]


def test_coupling_diagnostic_rollout_runs_without_training(tmp_path):
    module = _diagnostic_module()
    module.OUT_DIR = tmp_path

    result = module.rollout(
        _ZeroPolicy(),
        [73],
        action_mode="hold",
        action_repeat=1,
        horizon_steps=1,
        seed=3,
    )

    assert result["pitches"] == [73]
    assert "records" in result
    assert result["records"]
    assert "key52_state" in result["records"][0]
    assert "key54_state" in result["records"][0]
    assert "actions" in result
