from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path("/home/reece_dev/msc-audio-pianist")


def _load_module():
    path = ROOT / "scripts/evaluate_refinement_canary.py"
    spec = importlib.util.spec_from_file_location("evaluate_refinement_canary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_export(root: Path, *, omit_one: bool = False) -> None:
    arms = [
        ("control_continue_sensitive_v1", "transition_cleanup_sensitive_v1"),
        ("release_completion_v2", "transition_cleanup_release_completion_v2"),
        ("release_completion_motion_v2", "transition_cleanup_release_completion_motion_v2"),
    ]
    (root / "launch_manifest.json").write_text(
        json.dumps({"arms": [{"arm": arm, "reward_profile": profile} for arm, profile in arms]}),
        encoding="utf-8",
    )
    for arm, _profile in arms:
        ckpt_dir = root / arm / "lightweight_checkpoints" / f"{arm}_run"
        ckpt_dir.mkdir(parents=True)
        for step in (5000, 10000, 25000, 50000, 75000):
            if omit_one and arm == "release_completion_v2" and step == 25000:
                continue
            (ckpt_dir / f"checkpoint_{step}_steps.pt").write_bytes(b"not a real checkpoint")


def test_discovery_supports_hex_export_layout(tmp_path):
    module = _load_module()
    _write_export(tmp_path)

    discovered = module.discover_checkpoints(tmp_path)

    assert len(discovered) == 15
    assert sorted({item.arm for item in discovered}) == [
        "control_continue_sensitive_v1",
        "release_completion_motion_v2",
        "release_completion_v2",
    ]
    assert sorted({item.step for item in discovered}) == [5000, 10000, 25000, 50000, 75000]
    assert all("/local_smokes/" not in str(item.path) for item in discovered)


def test_dry_run_fails_loudly_for_missing_checkpoint(tmp_path):
    module = _load_module()
    _write_export(tmp_path, omit_one=True)

    with pytest.raises(ValueError, match="missing checkpoint steps"):
        module.discovery_payload(
            tmp_path,
            tmp_path / "out",
            baseline_checkpoint=tmp_path / "baseline.pt",
            expected_checkpoint_count=15,
        )
