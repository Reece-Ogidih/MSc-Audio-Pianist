# Refinement Canary e8d8850

## Experiment Question

The refinement canary asked whether short actor-only warm-start refinement could improve the frozen five-note symbolic controller without replacing its useful anchor behaviour. The immutable controller remains:

`artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt`

The canary was run from implementation commit `e8d88501bdc2747b03306c1aac3d5cdcfb82f677`. The Hex export archive SHA-256 is:

`7656e30064c2bc21857f003315097b1f3c15a8e405411f15eafe5846d68f0de5`

## A/B/C Design

The canary compared three short refinement arms against the frozen 800k baseline:

| Arm | Purpose |
| --- | --- |
| A: `control_continue_sensitive_v1` | Continue with the existing sensitive transition-cleanup objective as a control arm. |
| B: `release_completion_v2` | Add reward pressure for completing the second target and releasing the previous target. |
| C: `release_completion_motion_v2` | Add the B release/completion idea plus transition-scoped motion penalties. |

All arms used the same five-note symbolic-controller setting and were evaluated with the same repeated-rollout behavioural metrics. The canary is a refinement diagnostic, not a replacement for the frozen baseline unless a candidate clearly improves behavioural quality.

## Checkpoint Schedule

Each arm was evaluated at:

`5k`, `10k`, `25k`, `50k`, and `75k` environment steps.

The frozen baseline was evaluated as checkpoint step `0`.

## Warm-Start Semantics and Limitations

The canary used actor-only warm-starting from the frozen controller with fresh critics and fresh replay. This makes the experiment useful for testing whether a reward/profile can preserve and refine behaviour, but it is not a pure continuation of the frozen run. The results showed that actor-only warm-starting is itself behaviourally disruptive, especially at early checkpoints.

Limitations:

- single seed;
- short refinement horizon;
- fresh critics and replay buffer;
- no claim that the same reward would behave identically in a full continuation;
- metrics are behavioural rollouts, not proof of human-perceived musical quality.

## Endpoint Metric Table

| Model | Step | Pressed Precision | Pressed Recall | Pressed F1 | Anchor Min F1 | Transition F1 Mean | Transition Completion | Second Target Completion | Late Release Mean | Wrong Crossings | Max Unintended Mean | Integrated Unintended Mean | Timestep F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0 | 0.962 | 0.692 | 0.782 | 1.000 | 0.646 | 0.000 | 0.000 | 1.000 | 0.077 | 0.704 | 0.881 | 0.672 |
| A: control continue | 75k | 0.962 | 0.692 | 0.782 | 1.000 | 0.646 | 0.000 | 0.000 | 0.750 | 0.077 | 0.442 | 0.706 | 0.582 |
| B: release completion | 75k | 0.974 | 0.731 | 0.805 | 1.000 | 0.683 | 0.125 | 0.125 | 0.250 | 0.077 | 0.330 | 0.560 | 0.525 |
| C: release completion + motion | 75k | 0.808 | 0.577 | 0.654 | 0.000 | 0.562 | 0.000 | 0.000 | 0.250 | 0.077 | 0.292 | 0.209 | 0.301 |

## A vs B Interpretation

`release_completion_v2` produced the clearest positive treatment signal. At 75k it improved pressed-key F1, transition F1, transition completion, second-target completion, late release, and unintended-travel metrics relative to the matched control arm.

The reason B was not selected is that the canary did not clearly beat the frozen baseline once timestep alignment and behavioural stability were included. Its endpoint improved some pressed-key and release/completion metrics, but materially reduced timestep F1 relative to the frozen controller. This matters because the next indirect pipeline must preserve note timing, not only eventual pressed-key sets.

## B vs C Interpretation

Adding transition-scoped motion penalties reduced some unintended-travel and action-motion quantities, but it damaged anchor reliability and timing. At the 75k endpoint, C had anchor min F1 `0.000`, transition F1 mean `0.562`, and timestep F1 `0.301`, all worse than B.

This suggests the motion penalty formulation was too blunt for this controller. It suppressed useful behaviour rather than producing cleaner transitions.

## Decision

Retain the immutable frozen 800k controller as the selected five-note symbolic controller:

`artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt`

No canary treatment clearly beat the frozen baseline as a replacement. B remains a promising future direction because it improved release/completion and pressed-key metrics, but it is not selected now because it reduced timestep alignment and came from an actor-only warm-start procedure that disrupted behaviour.

## Artifact Locations

Generated evaluation report:

`artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/hex_evaluation_e8d8850/comparison_report.md`

Selection recommendation:

`artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/hex_evaluation_e8d8850/selection_recommendation.json`

Summary tables:

`artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/hex_evaluation_e8d8850/per_checkpoint_summary.csv`

`artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/hex_evaluation_e8d8850/per_checkpoint_per_sequence_metrics.csv`

`artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/hex_evaluation_e8d8850/reward_component_summary.csv`

Treatment comparisons:

`artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/hex_evaluation_e8d8850/a_vs_b_release_completion_delta.csv`

`artifacts/frozen_models/five_note_symbolic_controller_v1/refinement_canary/hex_evaluation_e8d8850/b_vs_c_motion_delta.csv`

No generated CSV, JSON, checkpoint, archive, or video artifact is copied into this tracked record.
