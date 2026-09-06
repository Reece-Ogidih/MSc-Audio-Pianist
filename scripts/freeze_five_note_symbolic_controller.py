#!/usr/bin/env python
"""Freeze the selected five-note symbolic controller checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv


ROOT = Path("/home/reece_dev/msc-audio-pianist")
EXPORT_ROOT = ROOT / "artifacts/five_note_factorial_1m_hex_export"
SOURCE_RUN_DIR = (
    EXPORT_ROOT
    / "extracted/garlick/droq_sensitive_v1/"
    "droq_sensitive_v1_five_note_seed13_1m_20260723T201926Z"
)
SOURCE_CHECKPOINT = (
    SOURCE_RUN_DIR
    / "output/lightweight_checkpoints/"
    "droq_sensitive_v1_five_note_seed13_1m_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_1000000/"
    "checkpoint_800000_steps.pt"
)
ANALYSIS_DIR = EXPORT_ROOT / "analysis"
DEFAULT_RELEASE_DIR = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1"
SELECTED_STEP = 800_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_state() -> dict[str, Any]:
    try:
        commit = git_output(["rev-parse", "HEAD"])
        status = git_output(["status", "--short"])
    except Exception as exc:  # pragma: no cover - defensive for non-git copies.
        return {"commit": None, "dirty": None, "status": f"git unavailable: {exc}"}
    return {"commit": commit, "dirty": bool(status), "status": status}


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def package_versions() -> dict[str, str | None]:
    packages = [
        "torch",
        "numpy",
        "gymnasium",
        "stable_baselines3",
        "dm_env",
        "dm_control",
        "mujoco",
        "robopianist",
    ]
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def selected_rows(run_dir: Path, analysis_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "source_summary_800000": run_dir / "evaluation/per_checkpoint_summary.csv",
        "source_per_sequence_800000": run_dir / "evaluation/per_checkpoint_per_sequence_metrics.csv",
        "source_training_health_800000": run_dir / "evaluation/per_checkpoint_training_health.csv",
        "combined_checkpoint_summary_800000": analysis_dir / "combined_checkpoint_summary.csv",
        "best_checkpoint_per_condition": analysis_dir / "best_checkpoint_per_condition.csv",
        "per_condition_comparison": analysis_dir / "per_condition_comparison.csv",
    }
    rows: dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "condition_id" in df.columns:
            df = df[df["condition_id"] == "droq_sensitive_v1"]
        if "checkpoint_step" in df.columns:
            df = df[df["checkpoint_step"] == SELECTED_STEP]
        if "selected_checkpoint_step" in df.columns:
            df = df[df["selected_checkpoint_step"] == SELECTED_STEP]
        if df.empty:
            raise ValueError(f"No selected 800k rows found in {path}.")
        rows[name] = df.copy()
    return rows


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def checkpoint_dims(checkpoint_path: Path) -> dict[str, int]:
    policy = DroQPolicy.load(checkpoint_path, device="cpu")
    config = policy.agent.config
    return {"observation_dim": int(config.observation_dim), "action_dim": int(config.action_dim)}


def env_dims(config: dict[str, Any]) -> dict[str, Any]:
    env = GeneralOneHandGoalEnv(
        curriculum=config["curriculum"],
        midi_min=config["midi_min"],
        midi_max=config["midi_max"],
        sequence_pitches=tuple(tuple(seq) for seq in config["sequence_pitches"]),
        sequence_sampling_weights=tuple(config["sequence_sampling_weights"]),
        sequence_timing_profile=config["sequence_timing_profile"],
        seed=config["seed"],
        lookahead=config["lookahead"],
        horizon_steps=64,
        action_mode=config["action_mode"],
        action_repeat=config["action_repeat"],
    )
    try:
        obs, _ = env.reset(seed=config["seed"])
        return {
            "observation_dim": int(obs.shape[0]),
            "action_dim": int(env.action_space.shape[0]),
            "native_action_shape": list(env._native_action_spec.shape),
            "native_goal_shape": list(env.native_goal_shape),
            "action_names": list(env.action_names),
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def source_output_config(run_dir: Path) -> Path:
    matches = sorted((run_dir / "output").glob("*_config.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one output config under {run_dir / 'output'}, found {matches}.")
    return matches[0]


def source_output_summary(run_dir: Path) -> Path:
    matches = sorted((run_dir / "output").glob("*_summary.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one output summary under {run_dir / 'output'}, found {matches}.")
    return matches[0]


def build_model_card(manifest: dict[str, Any], selection: dict[str, Any]) -> str:
    return f"""# Five-Note Symbolic Controller V1

## Intended Use

This frozen artifact is the selected five-note symbolic-controller baseline for the MSc RoboPianist simulation project. It is a lightweight DroQ actor checkpoint intended for deterministic inference/evaluation in the one-hand five-note MIDI range, not a resumable optimizer/replay-buffer checkpoint.

## Model

- Model kind: {manifest['model_kind']}
- Release: {manifest['release_name']} {manifest['release_version']}
- Selected step: {manifest['selected_step']}
- Seed: {manifest['seed']}
- Reward profile: `{manifest['reward_profile']}`
- MIDI range: {manifest['midi_range']}
- Action mode/repeat: `{manifest['action_mode']}` / {manifest['action_repeat']}
- Lookahead: {manifest['lookahead']}
- Timing profile: `{manifest['sequence_timing_profile']}`
- Observation/action dimensions: {manifest['observation_dim']} / {manifest['action_dim']}

## Selection Evidence

The checkpoint was selected from the five-note 2x2 factorial analysis as the strongest cleanliness-aware trade-off. At selection it had trained pressed-key F1 {selection['selected_trained_pressed_key_f1']:.3f}, trained timestep F1 {selection['selected_trained_timestep_f1']:.3f}, mean integrated unintended travel {selection['selected_integrated_unintended']:.3f}, and mean wrong-key threshold crossings {selection['selected_wrong_key_press_count']:.3f}.

It was selected at 800k instead of 1M because the 1M checkpoint improved some held-out/composition metrics but worsened unintended travel and wrong-key events. The known worst trained transition for the selected checkpoint is `{selection['selected_worst_trained_sequence']}` with pressed-key F1 {selection['selected_worst_trained_f1']:.3f}.

## Trained Material

The training range is MIDI 72-76 with single-note anchors and adjacent trained transitions. The exact sequence list and sampling weights are preserved in `manifest.json` and `provenance/resolved_training_config.json`.

## Limitations

- This is a constrained simulation proof of concept, not hardware-ready robot-control software.
- It is selected from seed 13 only; no multi-seed confidence interval is available.
- Held-out and composition-probe performance remain limited.
- The checkpoint is lightweight/inference-only. It should not be treated as an exact optimizer/replay-buffer resume checkpoint.
"""


def build_release_readme(release_dir: Path) -> str:
    release = str(release_dir)
    py_path = (
        "PYTHONPATH=/home/reece_dev/msc-audio-pianist/src:"
        "/home/reece_dev/msc-audio-pianist/third_party/robopianist"
    )
    return f"""# Frozen Five-Note Symbolic Controller V1

## Recreate the Frozen Release

```bash
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist
{py_path} python scripts/freeze_five_note_symbolic_controller.py --force
```

Omit `--force` to require that the release directory does not already exist.

## Verify SHA-256 Checksums

```bash
cd {release}
sha256sum -c SHA256SUMS.txt
```

## Run Local Reproducibility Evaluation

```bash
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist
{py_path} python scripts/validate_frozen_five_note_symbolic_controller.py
```

Outputs are written under `{release}/validation/`.

## Regenerate the Smoke Rollout

```bash
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist
MUJOCO_GL=egl {py_path} python scripts/render_frozen_five_note_smoke_rollout.py
```

The rollout artifacts are written to `{release}/validation/smoke_rollout/trained_73_72/`.
If EGL is unavailable locally, try `MUJOCO_GL=osmesa` if the local MuJoCo stack supports it.
"""


def freeze_release(
    *,
    release_dir: Path,
    source_checkpoint: Path,
    source_run_dir: Path,
    analysis_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    if release_dir.exists():
        if not force:
            raise FileExistsError(f"{release_dir} already exists; rerun with --force to overwrite.")
        shutil.rmtree(release_dir)
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    if not source_run_dir.exists():
        raise FileNotFoundError(source_run_dir)
    release_dir.mkdir(parents=True)
    provenance_dir = release_dir / "provenance"
    validation_dir = release_dir / "validation"
    provenance_dir.mkdir()
    validation_dir.mkdir()

    frozen_checkpoint = release_dir / "checkpoint_800000_steps.pt"
    shutil.copy2(source_checkpoint, frozen_checkpoint)
    source_hash = sha256_file(source_checkpoint)
    frozen_hash = sha256_file(frozen_checkpoint)
    if source_hash != frozen_hash:
        raise RuntimeError("Frozen checkpoint SHA-256 does not match source checkpoint.")

    training_config = load_json(source_run_dir / "resolved_training_config.json")
    output_config = load_json(source_output_config(source_run_dir))
    metadata = parse_env_file(source_run_dir / "resolved_metadata.env")
    selection_rows = selected_rows(source_run_dir, analysis_dir)
    for name, df in selection_rows.items():
        write_dataframe(provenance_dir / f"{name}.csv", df)
        write_json(provenance_dir / f"{name}.json", df.to_dict(orient="records"))

    for source_name in [
        "resolved_training_config.json",
        "resolved_metadata.env",
        "factorial_manifest_seed13.json",
        "five_note_curriculum_v1.json",
        "evaluation/best_checkpoint_by_metric.json",
        "evaluation/evaluation_config.json",
        "evaluation/evaluation_summary.json",
    ]:
        src = source_run_dir / source_name
        if src.exists():
            dest = provenance_dir / Path(source_name).name
            shutil.copy2(src, dest)
    shutil.copy2(source_output_config(source_run_dir), provenance_dir / source_output_config(source_run_dir).name)
    shutil.copy2(source_output_summary(source_run_dir), provenance_dir / source_output_summary(source_run_dir).name)

    dims = checkpoint_dims(frozen_checkpoint)
    dims.update(env_dims(training_config))
    local_git = git_state()
    selection = selection_rows["per_condition_comparison"].iloc[0].to_dict()
    manifest = {
        "release_name": "five_note_symbolic_controller",
        "release_version": "v1",
        "model_kind": "DroQ",
        "checkpoint_kind": "lightweight_inference_checkpoint",
        "selected_step": SELECTED_STEP,
        "seed": int(training_config["seed"]),
        "reward_profile": training_config["reward_profile"],
        "midi_range": [int(training_config["midi_min"]), int(training_config["midi_max"])],
        "action_mode": training_config["action_mode"],
        "action_repeat": int(training_config["action_repeat"]),
        "lookahead": int(training_config["lookahead"]),
        "sequence_timing_profile": training_config["sequence_timing_profile"],
        "observation_dim": int(dims["observation_dim"]),
        "action_dim": int(dims["action_dim"]),
        "native_action_shape": dims["native_action_shape"],
        "native_goal_shape": dims["native_goal_shape"],
        "source_checkpoint_absolute_path": str(source_checkpoint),
        "frozen_checkpoint_relative_path": frozen_checkpoint.name,
        "source_checkpoint_sha256": source_hash,
        "frozen_checkpoint_sha256": frozen_hash,
        "training_git_commit": metadata.get("git_commit"),
        "expected_training_git_commit_prefix": "b5be771",
        "current_local_git_commit": local_git["commit"],
        "current_local_git_dirty": local_git["dirty"],
        "current_local_git_status": local_git["status"],
        "source_run_name": training_config["run_name"],
        "source_run_dir": str(source_run_dir),
        "source_evaluation_paths": {
            "summary": str(source_run_dir / "evaluation/per_checkpoint_summary.csv"),
            "per_sequence_metrics": str(source_run_dir / "evaluation/per_checkpoint_per_sequence_metrics.csv"),
            "training_health": str(source_run_dir / "evaluation/per_checkpoint_training_health.csv"),
            "best_checkpoint_by_metric": str(source_run_dir / "evaluation/best_checkpoint_by_metric.json"),
            "selection_report": str(analysis_dir.parent / "factorial_results_report.md"),
        },
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_and_package_versions": package_versions(),
        "sequence_pitches": training_config["sequence_pitches"],
        "sequence_sampling_weights": training_config["sequence_sampling_weights"],
        "droq_config": output_config.get("droq_config"),
        "explicit_checkpoint_statement": (
            "This is the selected lightweight inference checkpoint. It is not an exact "
            "optimizer/replay-buffer resume checkpoint."
        ),
    }
    if not str(manifest["training_git_commit"] or "").startswith("b5be771"):
        raise ValueError(f"Unexpected training git commit: {manifest['training_git_commit']!r}")

    write_json(release_dir / "manifest.json", manifest)
    write_json(release_dir / "selection_metrics.json", selection)
    pd.DataFrame([selection]).to_csv(release_dir / "selection_metrics.csv", index=False)
    (release_dir / "model_card.md").write_text(build_model_card(manifest, selection), encoding="utf-8")
    (release_dir / "README.md").write_text(build_release_readme(release_dir), encoding="utf-8")
    sums = [
        f"{frozen_hash}  {frozen_checkpoint.name}",
        f"{sha256_file(release_dir / 'manifest.json')}  manifest.json",
        f"{sha256_file(release_dir / 'selection_metrics.json')}  selection_metrics.json",
        f"{sha256_file(release_dir / 'selection_metrics.csv')}  selection_metrics.csv",
        f"{sha256_file(release_dir / 'model_card.md')}  model_card.md",
        f"{sha256_file(release_dir / 'README.md')}  README.md",
    ]
    (release_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = freeze_release(
        release_dir=args.release_dir,
        source_checkpoint=args.source_checkpoint,
        source_run_dir=args.source_run_dir,
        analysis_dir=args.analysis_dir,
        force=args.force,
    )
    print(json.dumps({"release_dir": str(args.release_dir), "checkpoint_sha256": manifest["frozen_checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
