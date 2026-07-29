#!/usr/bin/env python
"""Prepare the controlled three-arm refinement canary package."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path("/home/reece_dev/msc-audio-pianist")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_general_one_hand_policy import reward_config_from_profile  # noqa: E402


FROZEN = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1"
AUDIT = FROZEN / "audit"
CANARY = FROZEN / "refinement_canary"
CONFIG_DIR = ROOT / "configs/refinement_canary"
CHECKPOINT = FROZEN / "checkpoint_800000_steps.pt"
CHECKPOINT_SHA256 = "927c1050c08769c49568013ead0c69d69d4bd19ff23eb632e89bd89fb735ac4c"
EVALUATION_SCHEDULE = [0, 5000, 10000, 25000, 50000, 75000]
LIGHTWEIGHT_CHECKPOINT_STEPS = [5000, 10000, 25000, 50000, 75000]
FULL_CHECKPOINT_STEPS = [75000]

SEQUENCES = [
    [72],
    [73],
    [74],
    [75],
    [76],
    [72, 73],
    [73, 72],
    [73, 74],
    [74, 73],
    [74, 75],
    [75, 74],
    [75, 76],
    [76, 75],
]

ARMS = {
    "control_continue_sensitive_v1": "transition_cleanup_sensitive_v1",
    "release_completion_v2": "transition_cleanup_release_completion_v2",
    "release_completion_motion_v2": "transition_cleanup_release_completion_motion_v2",
}


def main() -> None:
    CANARY.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CANARY / "design").mkdir(exist_ok=True)
    (CANARY / "local_smokes").mkdir(exist_ok=True)
    (CANARY / "validation").mkdir(exist_ok=True)
    weights, derivation = derive_weights()
    configs = []
    for arm, reward_profile in ARMS.items():
        config = arm_config(arm, reward_profile, weights)
        config_path = CONFIG_DIR / f"{arm}_75k.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        configs.append(str(config_path))
    write_readme(weights, derivation)
    write_design(weights, derivation)
    manifest = launch_manifest(weights, derivation, configs)
    (CANARY / "launch_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_launch_wrappers()
    print(json.dumps({"canary_root": str(CANARY), "configs": configs}, indent=2))


def derive_weights() -> tuple[list[float], dict]:
    metrics = pd.read_csv(AUDIT / "reference_rollout_metrics.csv")
    frozen = metrics[
        (metrics["model_id"] == "droq_sensitive_v1_800k_frozen")
        & (metrics["sequence_group"] == "trained_transition")
    ]
    scores = {}
    for _, row in frozen.iterrows():
        labels = set(json.loads(row["failure_labels"]))
        score = 1.0
        for label in (
            "previous_target_not_released",
            "second_target_missed",
            "transition_incomplete",
            "neighbouring_wrong_key_press",
            "unrelated_wrong_key_press",
        ):
            if label in labels:
                score += 1.0
        scores[tuple(json.loads(row["sequence_pitches"]))] = score
    single_weight = 0.07
    transition_mass = 0.65
    transition_scores = [scores[tuple(seq)] for seq in SEQUENCES if len(seq) == 2]
    total_score = sum(transition_scores)
    transition_weights = [transition_mass * score / total_score for score in transition_scores]
    weights = [single_weight] * 5 + transition_weights
    derivation = {
        "source": str(AUDIT / "reference_rollout_metrics.csv"),
        "model_id": "droq_sensitive_v1_800k_frozen",
        "single_note_policy": "five anchors preserved at 0.07 each, total 0.35",
        "transition_policy": "remaining 0.65 distributed by 1 + observed failure-label count",
        "failure_labels_counted": [
            "previous_target_not_released",
            "second_target_missed",
            "transition_incomplete",
            "neighbouring_wrong_key_press",
            "unrelated_wrong_key_press",
        ],
        "transition_scores": {";".join(map(str, key)): value for key, value in sorted(scores.items())},
        "cap_check": {
            "max_weight": max(weights),
            "one_sequence_dominates": max(weights) > 0.12,
        },
    }
    return [round(float(weight), 10) for weight in weights], derivation


def arm_config(arm: str, reward_profile: str, weights: list[float]) -> dict:
    return {
        "arm": arm,
        "algorithm": "droq",
        "warm_start_policy_path": str(CHECKPOINT),
        "warm_start_semantics": "actor weights only; fresh critics, optimizers, replay buffer and RNG from seed",
        "reward_profile": reward_profile,
        "reward_config": asdict(reward_config_from_profile(reward_profile)),
        "seed": 13,
        "evaluation_seed": 23,
        "timesteps": 75000,
        "evaluation_steps": EVALUATION_SCHEDULE,
        "lightweight_checkpoint_steps": LIGHTWEIGHT_CHECKPOINT_STEPS,
        "full_resumable_checkpoint_steps": FULL_CHECKPOINT_STEPS,
        "checkpoint_freq": 0,
        "lightweight_checkpoint_freq": 0,
        "midi_min": 72,
        "midi_max": 76,
        "lookahead": 1,
        "horizon_steps": 96,
        "curriculum": "sequence_cleanup",
        "sequence_pitches": SEQUENCES,
        "sequence_sampling_weights": weights,
        "sequence_timing_profile": "aligned",
        "action_mode": "direct",
        "action_repeat": 1,
        "utd_ratio": 4,
        "batch_size": 256,
        "learning_starts": 1000,
        "device": "cuda",
        "output_subdir": f"experiments/refinement_canary/{arm}",
    }


def launch_manifest(weights: list[float], derivation: dict, configs: list[str]) -> dict:
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True)
    frozen_manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    return {
        "canary_name": "five_note_symbolic_controller_v1_refinement_canary",
        "frozen_source_checkpoint": str(CHECKPOINT),
        "frozen_source_checkpoint_sha256": sha256(CHECKPOINT),
        "source_frozen_manifest": str(FROZEN / "manifest.json"),
        "source_frozen_manifest_payload": frozen_manifest,
        "arms": [arm_config(arm, profile, weights) for arm, profile in ARMS.items()],
        "sampling_weight_derivation": derivation,
        "training_seed": 13,
        "evaluation_seed": 23,
        "training_budget": 75000,
        "evaluation_schedule": EVALUATION_SCHEDULE,
        "lightweight_checkpoint_schedule": LIGHTWEIGHT_CHECKPOINT_STEPS,
        "full_resumable_checkpoint_schedule": FULL_CHECKPOINT_STEPS,
        "full_checkpoint_policy": "Save lightweight checkpoints at 5k/10k/25k/50k/75k. Save the full resumable checkpoint only at 75k; step 0 references the immutable frozen source checkpoint.",
        "warm_start_semantics": "All arms initialize actor weights from the frozen lightweight checkpoint only. Critics, optimizers, replay buffer and RNG are fresh.",
        "git_commit": commit.stdout.strip(),
        "git_dirty_status": status.stdout,
        "acceptance_gates": {
            "anchors": "all five single-note anchors retained",
            "release_failures": "previous-target-not-released reduced by about 20% or more vs arm A",
            "completion": "second-note completion improves by about 15 percentage points or comparable relative improvement",
            "f1": "worst trained-transition F1 improves or does not materially regress",
            "timestep_f1": "does not fall by more than 0.02 relative to matched control",
            "unintended": "unintended presses and wrong-key crossings do not materially increase",
            "motion": "mean action delta or saturation improves by 5-10%; p95 fingertip jerk improves or remains stable",
        },
        "prelaunch_validation": {
            "reward_scale_audit": str(CANARY / "validation/reward_scale_audit.md"),
            "step0_reproducibility": str(CANARY / "validation/step0_reproducibility.md"),
        },
        "runtime_estimate": "Approximately 3-5 hours per 75k DroQ arm based on prior Hex production performance; estimate only.",
        "config_files": configs,
    }


def write_readme(weights: list[float], derivation: dict) -> None:
    lines = [
        "# Refinement Canary",
        "",
        "Controlled three-arm DroQ warm-start canary for `five_note_symbolic_controller_v1`.",
        "",
        "All arms use actor-weight warm-starting from the frozen DroQ-sensitive 800k lightweight checkpoint. "
        "This is not an exact checkpoint resume: critics, optimizers, replay buffer and RNG are fresh.",
        "",
        "No Hex run has been launched by this package.",
        "",
        "## Arms",
        "",
        "- `control_continue_sensitive_v1`: unchanged `transition_cleanup_sensitive_v1`.",
        "- `release_completion_v2`: adds transition-gated previous-target release and one-shot second-target completion.",
        "- `release_completion_motion_v2`: arm B plus small transition-scoped action-rate and saturation penalties.",
        "",
        "## Sampling Weights",
        "",
        f"- sequences: `{SEQUENCES}`",
        f"- weights: `{weights}`",
        f"- derivation: {derivation['transition_policy']}; {derivation['single_note_policy']}.",
        "",
        "## Schedule",
        "",
        f"- evaluation steps: `{EVALUATION_SCHEDULE}`",
        f"- lightweight checkpoints: `{LIGHTWEIGHT_CHECKPOINT_STEPS}`",
        f"- full resumable checkpoints: `{FULL_CHECKPOINT_STEPS}`",
        "- step 0 is evaluated by referencing the immutable frozen source checkpoint; no duplicate step-0 checkpoint is created.",
        "",
        "## Runtime Estimate",
        "",
        "Expected Hex runtime is approximately 3-5 hours per 75k DroQ arm based on prior production runs. This is an estimate, not a guarantee.",
        "",
        "## Evaluation",
        "",
        "After an arm finishes, rerun aggregation independently with:",
        "",
        "```bash",
        "PYTHONPATH=/app/src:/app/third_party/robopianist python scripts/evaluate_refinement_canary.py --canary-root /app/artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary --include-frozen-baseline",
        "```",
    ]
    (CANARY / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_design(weights: list[float], derivation: dict) -> None:
    lines = [
        "# Canary Design",
        "",
        "The three arms differ only by reward profile. Training seed, warm-start checkpoint, curriculum, timing, action mode, budget and checkpoint schedule are matched.",
        "",
        "## Reward Definitions",
        "",
        "See `launch_manifest.json` for exact coefficients.",
        "",
        "- Control: original `transition_cleanup_sensitive_v1` unchanged.",
        "- Release/completion v2: base control plus `release_completion_release_weight=1.25` and `release_completion_bonus=1.50`.",
        "- Motion v2: release/completion v2 plus `transition_action_rate_weight=0.05`, `transition_saturation_weight=0.03`, `transition_saturation_threshold=0.95`.",
        "",
        "Second-target completion is one-shot per transition and does not require release first; release is separately rewarded/penalized through the previous-key state.",
        "",
        "## Checkpoint Policy",
        "",
        f"- Evaluate: `{EVALUATION_SCHEDULE}`",
        f"- Lightweight: `{LIGHTWEIGHT_CHECKPOINT_STEPS}`",
        f"- Full resumable: `{FULL_CHECKPOINT_STEPS}` only.",
        "- Step 0 uses the immutable frozen checkpoint; no step-0 duplicate is written.",
        "",
        "## Sampling Weight Derivation",
        "",
        f"- source: `{derivation['source']}`",
        f"- counted labels: `{derivation['failure_labels_counted']}`",
        f"- transition scores: `{derivation['transition_scores']}`",
        f"- final weights: `{weights}`",
    ]
    (CANARY / "design/canary_design.md").write_text("\n".join(lines), encoding="utf-8")


def write_launch_wrappers() -> None:
    local = ROOT / "scripts/run_refinement_canary_local_smokes.sh"
    local.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
cd /home/reece_dev/msc-audio-pianist
source /home/reece_dev/miniforge3/etc/profile.d/conda.sh
conda activate pianist

ROOT=/home/reece_dev/msc-audio-pianist
PYTHONPATH="$ROOT/src:$ROOT/third_party/robopianist"
CKPT="$ROOT/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt"
OUT="$ROOT/artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/local_smokes"
COMMON=(--timesteps 1000 --midi-min 72 --midi-max 76 --lookahead 1 --curriculum sequence_cleanup --sequence-pitches "72;73;74;75;76;72,73;73,72;73,74;74,73;74,75;75,74;75,76;76,75" --sequence-sampling-weights "0.07,0.07,0.07,0.07,0.07,0.08125,0.1015625,0.08125,0.08125,0.08125,0.08125,0.0609375,0.08125" --sequence-timing-profile aligned --action-mode direct --action-repeat 1 --horizon-steps 96 --full-checkpoint-steps 1000 --lightweight-checkpoint-steps 1000 --warm-start-policy-path "$CKPT" --seed 13 --learning-starts 100 --batch-size 64 --utd-ratio 1 --device cpu)

for ARM in control_continue_sensitive_v1 release_completion_v2 release_completion_motion_v2; do
  case "$ARM" in
    control_continue_sensitive_v1) PROFILE=transition_cleanup_sensitive_v1 ;;
    release_completion_v2) PROFILE=transition_cleanup_release_completion_v2 ;;
    release_completion_motion_v2) PROFILE=transition_cleanup_release_completion_motion_v2 ;;
  esac
  mkdir -p "$OUT/$ARM"
  PYTHONPATH="$PYTHONPATH" python scripts/train_droq_general_one_hand_policy.py "${COMMON[@]}" --reward-profile "$PROFILE" --stage-name "$ARM"_local_smoke --output-dir "$OUT/$ARM" 2>&1 | tee "$OUT/$ARM/train.log"
done
""",
        encoding="utf-8",
    )
    local.chmod(0o755)
    hex_script = ROOT / "scripts/hex/run_refinement_canary_arm.sh"
    hex_script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
: "${ARM:?Set ARM to one of control_continue_sensitive_v1, release_completion_v2, release_completion_motion_v2}"
: "${HEX_SCRATCH:?Set HEX_SCRATCH to mounted scratch/output storage}"
cd /app

case "$ARM" in
  control_continue_sensitive_v1) PROFILE=transition_cleanup_sensitive_v1 ;;
  release_completion_v2) PROFILE=transition_cleanup_release_completion_v2 ;;
  release_completion_motion_v2) PROFILE=transition_cleanup_release_completion_motion_v2 ;;
  *) echo "Unknown ARM=$ARM" >&2; exit 2 ;;
esac

CKPT=/app/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt
OUT="$HEX_SCRATCH/refinement_canary/$ARM"
EXPECTED_SHA="927c1050c08769c49568013ead0c69d69d4bd19ff23eb632e89bd89fb735ac4c"
ACTUAL_SHA="$(sha256sum "$CKPT" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "Frozen checkpoint SHA mismatch: expected $EXPECTED_SHA got $ACTUAL_SHA" >&2
  exit 3
fi
if [[ -e "$OUT" ]] && [[ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $OUT" >&2
  exit 4
fi
mkdir -p "$OUT/logs"
cp /app/artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/launch_manifest.json "$OUT/launch_manifest.json"
cp "/app/configs/refinement_canary/${ARM}_75k.json" "$OUT/config.json"
{
  echo "arm=$ARM"
  echo "reward_profile=$PROFILE"
  echo "start_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "git_status=$(git status --short 2>/dev/null | base64 -w0 || true)"
  echo "python=$(python --version 2>&1)"
  python - <<'PY'
import json, platform, torch
payload = {
    "platform_node": platform.node(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print("runtime_json=" + json.dumps(payload, sort_keys=True))
PY
  nvidia-smi || true
} > "$OUT/run_metadata_start.txt"
set +e

PYTHONPATH=/app/src:/app/third_party/robopianist python scripts/train_droq_general_one_hand_policy.py \
  --timesteps 75000 \
  --midi-min 72 \
  --midi-max 76 \
  --lookahead 1 \
  --curriculum sequence_cleanup \
  --sequence-pitches "72;73;74;75;76;72,73;73,72;73,74;74,73;74,75;75,74;75,76;76,75" \
  --sequence-sampling-weights "0.07,0.07,0.07,0.07,0.07,0.08125,0.1015625,0.08125,0.08125,0.08125,0.08125,0.0609375,0.08125" \
  --sequence-timing-profile aligned \
  --stage-name "$ARM" \
  --action-mode direct \
  --action-repeat 1 \
  --reward-profile "$PROFILE" \
  --checkpoint-freq 0 \
  --lightweight-checkpoint-freq 0 \
  --lightweight-checkpoint-steps "5000,10000,25000,50000,75000" \
  --full-checkpoint-steps "75000" \
  --warm-start-policy-path "$CKPT" \
  --seed 13 \
  --learning-starts 1000 \
  --batch-size 256 \
  --utd-ratio 4 \
  --device cuda \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/logs/train.log"
STATUS=${PIPESTATUS[0]}
set -e
{
  echo "finish_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code=$STATUS"
} > "$OUT/run_metadata_finish.txt"
exit "$STATUS"
""",
        encoding="utf-8",
    )
    hex_script.chmod(0o755)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
