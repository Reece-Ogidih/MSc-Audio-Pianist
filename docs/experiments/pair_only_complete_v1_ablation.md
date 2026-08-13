# Pair-Only Complete-v1 Ablation

## Question

Does complete coverage of local one- and two-event primitives improve acquisition and retention when long-horizon compositions are excluded from the +500k adaptation curriculum?

## Controlled comparison

The intended comparison is `BASE` versus `PAIR-ONLY` versus `PAIR + LONG-HORIZON` for each pipeline. Pair-only and combined adaptation branch independently from the same original model. They share seed 13, +500k environment steps, MIDI 72-76, aligned timing, direct 22D actions, no sustain, right hand, `transition_cleanup_sensitive_v1`, checkpoint schedule, and pipeline-specific optimizer settings.

The sole intended treatment difference is curriculum content:

- Pair-only: five anchors and all 25 ordered pairs, including five positive-gap repeated-note release/repress cases.
- Combined: the same anchors and pairs plus 36 compositions of length 3-20.

Pair-only sampling assigns 10% total mass to anchors and 90% to pairs: 2% per anchor and 3.6% per pair. Combined sampling assigned 10%, 50%, and 40% to anchors, pairs, and compositions respectively.

## Warm starts

- Pipeline 1 branches from the immutable canonical 800k symbolic controller. Only actor weights are loaded; critics, optimizers, replay, and RNG progression are fresh, with RNG initialized from seed 13.
- Pipeline 2 branches from the original successful seed13 1M full checkpoint. Networks, target critics, optimizers, and entropy/alpha state are loaded; replay is deliberately fresh and RNG is initialized from seed 13.

These semantics match the corresponding completed combined-refinement run. They differ between pipelines because the established adaptation mechanisms differ.

## Causal scope

Within each pipeline, differences between pair-only and combined adaptation can support a curriculum-content interpretation for this seed and budget. Better pair-only local performance would be consistent with long-horizon interference; better combined long-horizon performance would be consistent with useful compositional exposure.

The ablation cannot isolate pair completion from adaptation itself without a matched no-adaptation or baseline-continuation control. It cannot establish population effects from one seed, compare the two pipelines' warm-start mechanisms causally, or distinguish every possible source of stochastic training variation. A horizon-only cell is intentionally outside this experiment.
