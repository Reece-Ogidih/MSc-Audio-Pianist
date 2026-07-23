# Hex Five-Note Factorial 1M

This experiment replaces the staged three-note screening plan with a direct five-note factorial benchmark from timestep zero.

## Design

Range: MIDI 72-76, C5 through E5.

Conditions:

- `droq_original`: DroQ with `transition_cleanup`
- `droq_sensitive_v1`: DroQ with `transition_cleanup_sensitive_v1`
- `sac_original`: SB3 SAC with `transition_cleanup`
- `sac_sensitive_v1`: SB3 SAC with `transition_cleanup_sensitive_v1`

All four conditions start fresh from seed 13. Existing three-note runs are historical baselines only and must not initialise any condition.

Common sequences:

- Singles: `[72]`, `[73]`, `[74]`, `[75]`, `[76]`, each weight `0.10`
- Adjacent transitions: `[72,73]`, `[73,72]`, `[73,74]`, `[74,73]`, `[74,75]`, `[75,74]`, `[75,76]`, `[76,75]`, each weight `0.0625`

Weights sum to `1.0`. All five notes are present from timestep zero.

Common settings: `lookahead=1`, `action_mode=direct`, `action_repeat=1`, `sequence_timing_profile=aligned`, expected observation dimension `301`, expected action dimension `22`.

Sensitive reward coefficients are frozen:

- `unintended_soft_threshold = 0.20`
- `press_threshold = 0.50`
- `unintended_travel_weight = 0.75`
- `unintended_near_press_weight = 0.35`
- `unintended_press_weight = 1.0`
- `late_release_weight = 0.75`
- `early_activation_weight = 0.50`
- `duration_weight = 0.25`

## Local Push

```bash
cd /home/reece_dev/msc-audio-pianist
git status
git push origin main
```

## Hex Repository Update

```bash
cd /homes/rgkgo20/msc-audio-pianist
git fetch origin main
git pull --ff-only origin main
test "$(git rev-parse HEAD)" = "<NEW_COMMIT>"
```

## Common Environment

```bash
export HEX_SCRATCH=<scratch-path>
export HEX_IMAGE_TAG=<image-tag>
export AUTO_EVALUATE=1
```

Do not invent a fixed GPU index, node name, or scratch path.

## Preflight

```bash
scripts/hex/verify_five_note_factorial_inputs.sh
scripts/hex/smoke_test_five_note_factorial.sh
```

Useful inspection commands:

```bash
hostname
date -u
nvidia-smi
df -h "$HEX_SCRATCH"
hare me
hare usage
hare ps -a
```

## Launch

Run each condition on a separate node/shell/GPU allocation.

```bash
export HEX_GPU_INDEX=<gpu-index>
scripts/hex/run_five_note_droq_original_1m.sh
```

```bash
export HEX_GPU_INDEX=<gpu-index>
scripts/hex/run_five_note_droq_sensitive_1m.sh
```

```bash
export HEX_GPU_INDEX=<gpu-index>
scripts/hex/run_five_note_sac_original_1m.sh
```

```bash
export HEX_GPU_INDEX=<gpu-index>
scripts/hex/run_five_note_sac_sensitive_1m.sh
```

Each launcher creates a unique timestamped run directory under:

```bash
$HEX_SCRATCH/five_note_factorial_1m/<condition>/<run-name>/
```

Each launcher writes:

- `resolved_command.sh`
- `resolved_metadata.env`
- `started_at.txt`
- `finished_at.txt`
- `exit_code.txt`
- `logs/train.log`
- `resource_usage.csv`
- `output/`
- `evaluation/` when `AUTO_EVALUATE=1`

## Status And Logs

```bash
export CONDITION_ID=<droq_original|droq_sensitive_v1|sac_original|sac_sensitive_v1>
scripts/hex/status_five_note_factorial_job.sh
hare logs -n 120 <hare-name>
tail -f "$HEX_SCRATCH/five_note_factorial_1m/$CONDITION_ID/<run-name>/logs/train.log"
tail -f "$HEX_SCRATCH/five_note_factorial_1m/$CONDITION_ID/<run-name>/resource_usage.csv"
du -sh "$HEX_SCRATCH/five_note_factorial_1m"
```

## Safe Stop

```bash
export CONDITION_ID=<condition>
export HEX_RUN_NAME=<timestamped-run-name>
scripts/hex/stop_five_note_factorial_job.sh
CONFIRM_STOP=1 scripts/hex/stop_five_note_factorial_job.sh
```

Stopping should preserve partial outputs in scratch. Do not delete run directories until outputs have been reviewed.

## Resume

Resume uses full checkpoints only, not lightweight evaluation checkpoints.

```bash
export CONDITION_ID=<condition>
export RESUME_CHECKPOINT=<path-to-full-checkpoint-under-HEX_SCRATCH>
export ADDITIONAL_TIMESTEPS=<remaining-steps>
scripts/hex/resume_five_note_factorial_job.sh
```

DroQ full checkpoints include actor, critics, target critics, optimisers, alpha state, replay buffer, RNG state, timestep and config. SAC full checkpoints are the `.zip` plus matching replay-buffer pickle created by the full checkpoint callback.

## Evaluation

If automatic evaluation did not run:

```bash
export HEX_RUN_DIR="$HEX_SCRATCH/five_note_factorial_1m/<condition>/<run-name>"
export CONFIG_PATH="$HEX_RUN_DIR/resolved_training_config.json"
scripts/hex/evaluate_five_note_factorial_run.sh
```

Evaluate all latest condition runs:

```bash
scripts/hex/evaluate_five_note_factorial_all.sh
```

## Aggregation

```bash
scripts/hex/aggregate_five_note_factorial.sh
```

Outputs are saved under:

```bash
$HEX_SCRATCH/five_note_factorial_1m/aggregate/<timestamp>/
```

The selection report uses gate-and-Pareto interpretation. Shaped return is not used for cross-reward ranking.

## Compact Packaging

```bash
scripts/hex/package_five_note_factorial_results.sh
```

The compact archive includes resolved configs, manifests, launch commands, CSV/JSON metrics, SVG plots, selected logs, resource summaries, lightweight best/final policies where present, checkpoint hashes and transfer manifests.

It excludes replay buffers, optimiser-heavy full checkpoints, rolling resumable checkpoints, Docker caches, Python caches and raw container filesystems.

Preserve on Hex until selection is complete:

- full 500k checkpoint
- full 1M checkpoint
- rolling latest full checkpoint

## Download

```bash
rsync -avz <hex-host>:"$HEX_SCRATCH/five_note_factorial_1m/packages/five_note_factorial_1m_results_compact_<timestamp>.tar.gz" artifacts/hex_runs/
```

## Cleanup

Only after successful transfer and review:

```bash
du -sh "$HEX_SCRATCH/five_note_factorial_1m"
# then remove non-selected intermediate artifacts manually
```

Do not automatically delete selected full checkpoints.
