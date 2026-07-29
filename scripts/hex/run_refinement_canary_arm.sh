#!/usr/bin/env bash
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

PYTHONPATH=/app/src:/app/third_party/robopianist python scripts/train_droq_general_one_hand_policy.py   --timesteps 75000   --midi-min 72   --midi-max 76   --lookahead 1   --curriculum sequence_cleanup   --sequence-pitches "72;73;74;75;76;72,73;73,72;73,74;74,73;74,75;75,74;75,76;76,75"   --sequence-sampling-weights "0.07,0.07,0.07,0.07,0.07,0.08125,0.1015625,0.08125,0.08125,0.08125,0.08125,0.0609375,0.08125"   --sequence-timing-profile aligned   --stage-name "$ARM"   --action-mode direct   --action-repeat 1   --reward-profile "$PROFILE"   --checkpoint-freq 0   --lightweight-checkpoint-freq 0   --lightweight-checkpoint-steps "5000,10000,25000,50000,75000"   --full-checkpoint-steps "75000"   --warm-start-policy-path "$CKPT"   --seed 13   --learning-starts 1000   --batch-size 256   --utd-ratio 4   --device cuda   --output-dir "$OUT"   2>&1 | tee "$OUT/logs/train.log"
STATUS=${PIPESTATUS[0]}
set -e
{
  echo "finish_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "exit_code=$STATUS"
} > "$OUT/run_metadata_finish.txt"
exit "$STATUS"
