# Internal Gemma–Indic bridge

## Decision

Normal users will **not** be connected yet. The first bridge is an offline/internal candidate system. Its purpose is to prove that Santali input can travel through an Indic mediator to a Gemma reasoning model and return through the reverse translation path without hiding uncertainty.

## Flow

```text
Santali input
    ↓
IndicTrans2 Santali → English
    ↓
Gemma reasoning interface
    ↓
IndicTrans2 English → Santali
    ↓
Candidate response for internal review only
```

The current English→Santali adapter is only one direction. A real two-way bridge therefore needs a reverse Santali→English adapter trained from the same parallel corpus, followed by validation on held-out examples.

## Release gates

The bridge remains blocked from public exposure until all of the following are true:

1. Model loading and inference are reproducible.
2. Both translation directions pass finite-loss and artifact checks.
3. BLEU and chrF are reported on a held-out set.
4. Ol Chiki validity and empty/repeated-output rates are acceptable.
5. Native Santali reviewers assess adequacy and fluency.
6. Gemma confidence and translation confidence thresholds are calibrated.
7. Medical, legal, financial, privacy, and unknown-question tests are reviewed.
8. The adapter, tokenizer, metrics, model card, and provenance manifest are archived.

## Confidence behavior

A low-confidence translation does not get silently sent to Gemma as fact. The bridge returns a review/clarification result. A low-confidence Gemma answer is also blocked from final translation. Every candidate carries a trace showing the route and why it remains internal.

## Model roles

IndicTrans2 is the language mediator and translation specialist. Gemma is the reasoning and conversation layer. The models are not merged immediately: keeping them separate makes failures attributable, allows independent upgrades, and prevents a large language model from silently overriding an uncertain translation.

## Implementation

The internal orchestration contract is in `src/santali_ai/bridge.py`. It has no public HTTP endpoint and refuses a public-release policy. Real model adapters will be plugged into its `translate` and `answer` interfaces only after the current training artifact and reverse-direction adapter are verified.
