# Pipeline 2 Behavioural Audio Intervention Plan

## Question

Does rollout performance deteriorate when the refined direct-audio policy receives zero or mismatched audio instead of the correct waveform?

## Fixed model and conditions

Use only the refined Pipeline 2 +500k lightweight checkpoint. For every selected sequence run the complete deterministic episode with:

- `correct`: the sequence's canonical rendered audio;
- `zero`: the audio observation replaced by zeros;
- `mismatched`: canonical audio from a different sequence, selected deterministically by the evaluator.

Physical observations, environment configuration, timing, reset seed, horizon, actor inference, and checkpoint remain unchanged.

## Prespecified subset

Thirteen sequences balance coverage and runtime:

- anchors: `[72]`, `[74]`, `[76]` (low, centre, high register);
- originally trained adjacent pairs: `[72,73]`, `[73,72]` (both directions);
- new nonadjacent pairs: `[73,76]`, `[75,73]` (larger motions and both directions);
- clean compositions: both prespecified clean-test examples at each of lengths 3, 5, and 10. This ensures each mismatched waveform has the same sequence length as its target.

This gives 39 full rollouts. It is deliberately broader than one representative per length but avoids all 76 benchmark sequences. The exact list is encoded in the wrapper, not selected after observing intervention outcomes.

## Metrics and comparison

The existing sequence and event outputs provide timestep precision/recall/F1, pressed-key precision/recall/F1, event hit rate, ordered-event accuracy, transition-pair accuracy, correctly executed prefix, strict success, maximum/integrated unintended activation, unintended press counts, shaped return, and action saturation.

Report paired deltas from `correct` to `zero` and `mismatched` per sequence, plus category summaries. Evidence for behavioural necessity requires consistent degradation in task metrics, not merely action-vector differences. Mismatched audio should also be checked for accidentally matching part of the target sequence.

## Output

`/workspace/experiments/paircomplete_refined_evaluation_repairs/refined_evidence_repairs_v1/audio_interventions/{short,clean}/`

Expected runtime is approximately 3-6 minutes on one Hex GPU.
