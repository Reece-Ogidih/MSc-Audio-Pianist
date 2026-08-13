# Exact Original Short-Benchmark Retention Plan

## Benchmark identity

The original benchmark and the 13-row subset of `complete_pairwise_v1` contain the same five anchors and eight directed adjacent transitions. Both use MIDI 72-76, note duration 0.28 s, gap 0.12 s, velocity 90, no sustain, and the same key mapping.

Existing before/after summaries are informative but not strictly identical evaluations:

- original P1 and P2 were produced by separate benchmark scripts;
- original P2 used evaluation seed 2027, while the refined evaluator used 20260808;
- output schemas and some event/order metrics differ;
- the refined Basic Pitch condition is invalid.

Consequently, archived results can be retained as historical baselines, but they should not be the sole evidence for an unambiguous forgetting claim.

## Identical rerun

Filter `complete_pairwise_v1` by exact manifest names for:

- anchors 72, 73, 74, 75, 76;
- adjacent directed pairs 72-73, 73-72, 73-74, 74-73, 74-75, 75-74, 75-76, 76-75.

Evaluate all four checkpoints through `evaluate_long_horizon_compositional.py` with seed 20260808, aligned timing, one deterministic rollout, direct actions, action repeat 1, and the same metric implementation. P1 uses Oracle MIDI to isolate controller retention from transcription. P2 uses correct canonical audio.

This produces 52 rollouts and supports direct per-task and aggregate deltas:

- original P1 800k versus refined P1 +500k;
- original P2 seed13 1M versus refined P2 +500k.

## Reusable existing evidence

- Checkpoint identities and hashes from the original and refined manifests are reusable.
- Original P1 and P2 results remain valid descriptive baselines.
- Refined Oracle P1 and correct-audio P2 rows for these 13 tasks are valid.
- None of those rows should be mixed into the primary exact-retention table because they were not all generated in the same invocation and evaluation implementation.

## Output and cost

New outputs go under `.../refined_evidence_repairs_v1/retention/{p1_original,p1_refined,pipeline2}/`. Estimated runtime is 5-8 minutes on one Hex GPU.
