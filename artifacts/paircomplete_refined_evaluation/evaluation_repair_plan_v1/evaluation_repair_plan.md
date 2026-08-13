# Refined Evaluation Repair Plan v1

## Scope

This is evaluation-only work. It does not train a policy, alter a checkpoint, or overwrite the invalid export under `paircomplete_refined_model_evaluation_v1`.

Three evidence repairs are required:

1. Rerun only the refined Pipeline 1 Basic Pitch condition on `complete_pairwise_v1`, `long_horizon_compositional_v1`, `long_horizon_clean_test_v1`, and `long_horizon_extrapolation_v1`.
2. Run complete refined Pipeline 2 rollouts under correct, zero, and mismatched audio on thirteen prespecified sequences.
3. Evaluate the exact original 13 tasks with one evaluator, timing profile, seed, metric implementation, and rendering path for all four checkpoints.

Validation compositions are omitted from the first repair because they add eight more Basic Pitch/controller rollouts but do not close a distinct evidence gap. They can be added later without changing the design.

## Code and environment changes

- `evaluate_long_horizon_compositional.py` now supports `--pipeline1-condition` so Basic Pitch can be rerun without Oracle, repeated `--sequence-name` filters, and exact repeated `--audio-intervention-sequence` selection.
- Skip flags no longer audit or require checkpoints for a skipped pipeline.
- The Hex image installs Basic Pitch 0.4.0 without its unused TensorFlow Lite dependency, pins the locally proven ONNX Runtime 1.23.2 path and supporting packages, and verifies the bundled ONNX model during image build.
- `evaluate_refined_evidence_repairs.sh` refuses to overwrite a non-empty destination and runs only evaluation.

Defaults remain backward-compatible: without the new flags the evaluator still runs Oracle plus transcribed Pipeline 1 and correct-audio Pipeline 2, while `--include-audio-interventions` without an explicit list retains the old one-sequence-per-length rule.

## Hex execution proposal

- **GPU:** one CUDA GPU; no multi-GPU requirement. Basic Pitch inference uses ONNX Runtime CPU in the prepared image, while policy rollouts use CUDA.
- **Estimated wall time:** about 20-30 minutes on the same Hex class used for the original refined evaluation. The prior five full batteries took 12.8 minutes total while evaluating valid Oracle and Pipeline 2 conditions; the repair adds 76 Basic Pitch rollouts, 39 intervention rollouts, and 52 identical-retention rollouts.
- **New output:** `/workspace/experiments/paircomplete_refined_evaluation_repairs/refined_evidence_repairs_v1/`.
- **No prior output is reused as a destination.** Raw invalid rows remain intact.

## Required checkpoints

| Model | Exact checkpoint |
| --- | --- |
| Original P1 | `/app/artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt` |
| Refined P1 | `/workspace/runs/general_one_hand/droq/pipeline1_symbolic_paircomplete_longhorizon_v1/lightweight_checkpoints/pipeline1_symbolic_paircomplete_longhorizon_v1_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_sensitive_v1_seed13_500000/checkpoint_500000_steps.pt` |
| Original P2 | `/app/artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/lightweight_checkpoints/checkpoint_1000000_steps.pt` |
| Refined P2 | `/workspace/experiments/pipeline2_direct_audio/pipeline2_seed13_paircomplete_longhorizon_v1/lightweight_checkpoints/checkpoint_500000_steps.pt` |

The wrapper validates all four files before starting. The two refined checkpoints are not present in the downloaded evaluation export and must remain available on Hex at the recorded provenance paths, or be supplied through the corresponding environment variables.

## Execution status

Prepared and locally tested only. No Hex job has been launched.
