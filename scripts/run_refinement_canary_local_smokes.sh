#!/usr/bin/env bash
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
