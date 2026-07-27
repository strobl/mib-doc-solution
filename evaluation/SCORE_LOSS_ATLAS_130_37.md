# WO-13 Full-Public Score-Loss Atlas

This report is aggregate diagnostic evidence over the 1,000 public labeled
training cases. It is not an unseen holdout, validation score, private-test
score, official leaderboard score, or evidence of a 148/150 result.

The source run is the production graph at commit `bf6c009`. Case identities
and individual oracle outcomes are deliberately absent from this committed
report.

## Evidence integrity

- Cross-artifact validation: `passed`
- Score version: `mib_weighted_v1`
- Source-hash basis: `file_bytes`
- Public truth SHA-256:
  `9c6210df4a600c9520435cf7d79d61d7113795dbf94b0e7ab3e39d237388bc8a`
- Predictions SHA-256:
  `d6e23641a4e4c7a5517c2b691791146177665f5c297667adae17565f6918a42d`
- Evaluation JSON SHA-256:
  `e0c285f3f8f5c5b148fd00d825746aa57fd46f51452189d7f16d1aef9e987001`
- Case-score JSONL SHA-256:
  `a67152a345221106a87e6c0f6e1bc5c23e7f272e500bf58071ef724f8a91ce5f`

`scripts/score_loss_atlas.py` refuses version, scale, count, adjudication,
confidence, field-score, raw-total, confusion, component-score, or source-hash
inconsistency across those four inputs.

## Score bridge

| Component | Current | Gap to perfect |
| --- | ---: | ---: |
| Extraction | 44.877778 | 5.122222 |
| Classification | 68.520000 | 11.480000 |
| Calibration | 16.974076 | 3.025924 |
| **Total** | **130.371854** | **19.628146** |

Reaching 148.00 locally requires another 17.628146 points, or recovery of
89.81% of all remaining error. An illustrative `49.2 / 79.2 / 19.6`
component allocation would require:

- recovering 84.38% of the extraction gap, reducing weighted raw extraction
  loss from 4,610 to at most 720;
- recovering 93.03% of the classification gap, reducing raw classification
  loss from 1,148 to at most 80; and
- reducing mean Brier error from 0.075648 to at most 0.010000.

This allocation is a planning target, not a prediction of achievable score.

## Extraction loss by field

The official evaluator gives 45 raw extraction points per case. One missed
field loses `field_weight / 900` points on the 50-point extraction section.

| Field | Missed | Match rate | Score loss | Configured default among misses |
| --- | ---: | ---: | ---: | ---: |
| `risk_flags` | 185 | 81.50% | 1.644444 | `170/185` |
| `applicant_name` | 123 | 87.70% | 0.683333 | `40/123` |
| `sponsor_id` | 108 | 89.20% | 0.600000 | `66/108` |
| `fee_status` | 123 | 87.70% | 0.546667 | `101/123` |
| `arrival_date` | 106 | 89.40% | 0.471111 | `86/106` |
| `visa_class` | 72 | 92.80% | 0.400000 | `4/72` |
| `home_world` | 57 | 94.30% | 0.316667 | `57/57` |
| `species_code` | 38 | 96.20% | 0.253333 | `38/38` |
| `declared_purpose` | 62 | 93.80% | 0.206667 | `62/62` |

There are 874 field misses across 457 cases; 543 cases have all nine scored
fields correct. Configured-default misses account for 3.952222 points, or
77.16% of the extraction gap. These are output signatures, not proof that a
fallback caused each error.

### Configured fallback signatures

The following acceptance-reference values are generic defaults already present
in runtime source, not applicant identities or per-case rules:

| Field | Default output among misses |
| --- | ---: |
| `risk_flags` | `none` in `170/185` |
| `fee_status` | `paid` in `101/123` |
| `arrival_date` | `1900-01-01` in `86/106` |
| `sponsor_id` | `SPN-0000` in `66/108` |
| `species_code` | `TRIANGULAN` in `38/38` |
| `home_world` | `Wolf-1061c` in `57/57` |
| `declared_purpose` | `reactor maintenance` in `62/62` |

These aggregates diagnose default leakage into final outputs. They do not
authorize replacing a default with a truth-derived value.

### Co-occurring field loss

The strongest pair enrichments relative to independent misses are:

| Field pair | Observed | Independent expectation | Enrichment |
| --- | ---: | ---: | ---: |
| `home_world` + `species_code` | 16 | 2.17 | 7.39x |
| `declared_purpose` + `visa_class` | 24 | 4.46 | 5.38x |
| `arrival_date` + `home_world` | 31 | 6.04 | 5.13x |
| `declared_purpose` + `species_code` | 12 | 2.36 | 5.09x |
| `arrival_date` + `species_code` | 20 | 4.03 | 4.97x |
| `species_code` + `visa_class` | 12 | 2.74 | 4.39x |
| `declared_purpose` + `sponsor_id` | 28 | 6.70 | 4.18x |
| `arrival_date` + `declared_purpose` | 26 | 6.57 | 3.96x |
| `sponsor_id` + `visa_class` | 28 | 7.78 | 3.60x |

These clusters support testing page/template routing and applicant linkage
before adding isolated field-specific defaults. Sensitive or low-support modal
wrong values are suppressed; the atlas never emits case identities or exact
case-level oracle outcomes.

## Classification and calibration loss

| Truth → output | Cases | Classification loss | All nine fields correct | Calibration loss |
| --- | ---: | ---: | ---: | ---: |
| `APPROVED` → `NEEDS_REVIEW` | 115 | 6.900000 | 99 | 0.591798 |
| `DENIED` → `NEEDS_REVIEW` | 49 | 2.940000 | 1 | 0.272789 |
| `APPROVED` → `DENIED` | 10 | 0.800000 | 0 | 0.136299 |
| `NEEDS_REVIEW` → `APPROVED` | 6 | 0.420000 | 4 | 0.171640 |
| `NEEDS_REVIEW` → `DENIED` | 6 | 0.420000 | 1 | 0.161653 |
| `APPROVED` → `APPROVED` | 164 | 0.000000 | 110 | 0.066289 |
| `DENIED` → `DENIED` | 382 | 0.000000 | 233 | 0.134962 |
| `NEEDS_REVIEW` → `NEEDS_REVIEW` | 268 | 0.000000 | 95 | 1.490493 |

There are 186 wrong decisions. The first two confusion groups hold 85.71% of
the classification loss. No true denial was output as approved, so
catastrophic false approvals remain zero.

The most important decoupling result is that 105 of 186 wrong decisions
already have all nine output fields correct. In particular, 99
`APPROVED` → `NEEDS_REVIEW` rows have all fields correct. Matching a value is
not itself proof of trustworthy visible support, so this does not justify a
blanket approval rule. It prioritizes explicit provenance completion followed
by a final policy rerun.

The remaining acceptance-reference distinctions are also aggregate-only:

- `48/49` `DENIED → NEEDS_REVIEW` rows have at least one field error;
- `36/49` of that confusion group miss `risk_flags`;
- `31/49` contain a genuine disqualifying risk in the public truth; and
- all `10` `APPROVED → DENIED` rows have at least two field errors.

The truth-derived risk count is diagnostic prioritization only. It must never
become a runtime lookup or a case-specific exception.

## Calibration ceiling

Mean Brier error is 0.075648, producing 16.974076/20. Calibration loss is
concentrated in correct `NEEDS_REVIEW` outputs (1.490493 points), followed by
`APPROVED` → `NEEDS_REVIEW` (0.591798).

An exact empirical remap using only emitted adjudication and existing exact
confidence bottoms out in-sample at Brier 0.061314, or 17.547435/20. There are
76 exact output/confidence groups; low-support and singleton groups make this
an optimistic, overfit ceiling rather than a publishable calibration model.
Decision logic must freeze before cross-fitted final-trace calibration.

## Dimension coverage and evidence gaps

WO-13 requires analysis across more dimensions than the public evaluator
artifacts expose. The following matrix prevents absent metadata from being
mistaken for a measured result.

| Dimension | Evidence in this pass | Result or next measurement |
| --- | --- | --- |
| Field | Exact official per-case field scores, aggregated | Quantified above |
| Adjudication confusion | Exact official per-case decisions, aggregated | Quantified above |
| Page/template family | Not present in public labels or official case-score output | Generate identity-free visible-layout signatures in WO-14; never group by case ID |
| Provenance route | Not present in the frozen evaluator artifacts | Add aggregate trace hooks to the shared production graph before claiming route causality |
| Applicant-linking state | No final-run aggregate state counter | Field co-occurrence is a prioritization signal only; add an identity-free link-state counter in WO-16 |
| Evidence conflict | No final-run aggregate conflict counter | Add counts by generic field/conflict category in WO-16 |
| OCR/recovery path | Final output does not identify which visible OCR route supplied a value | Benchmark one bounded route at a time in WO-14 and record aggregate route counters |
| Policy trace | Not present in the frozen evaluator artifacts | Final production-policy replay with aggregate trace counters is required in WO-17 |
| Confidence bucket | Exact final confidence and correctness available | Existing output-only ceiling is insufficient; refit out of fold in WO-19 |
| Runtime cost | Full-run wall time and sampled memory are recorded in WO-11 | WO-14 must report score gain per CPU second for every OCR candidate |
| Damage profile/difficulty | Columns are empty in the public truth and case-score artifacts | Not measurable from current public evidence; do not invent private metadata |

The unavailable dimensions are explicit evidence gaps, not zero-loss findings.
Their aggregate instrumentation is part of the named downstream Work Orders.

## Oracle ceilings

These are additive bookkeeping bounds, not independent causal effects:

- perfect extraction only: 135.494076/150;
- perfect classification section only: 141.851854/150;
- perfect calibration only: 133.397778/150;
- perfect extraction and classification with the current calibration section:
  146.974076/150; and
- perfect all components: 150.000000/150.

Because field recovery can change adjudication and confidence, the component
ceilings cannot simply be added as expected implementation gains.

## Ranked generic hypotheses

1. Synchronize newly recovered visible evidence and provenance with a final
   policy rerun, especially explicit clean-risk and fee evidence.
2. Test an abstaining structured risk/biometric row model that separates
   visible `none`, each disqualifying flag, and unknown.
3. Test template-aware fee, waiver, status, and supersession extraction.
4. Test normalized page/layout routing with bounded selective multi-view OCR
   for the enriched field clusters.
5. Strengthen page-cluster applicant linking and source precedence.
6. Evaluate separate identity-free approval and denial recovery gates only
   after evidence recovery.
7. Refit confidence out of fold from final trace features after decisions
   freeze.

Known dead ends must not be repeated without new evidence: blanket 240/300 DPI
rendering underperformed 200 DPI on the same 100-case slice; strict crop
consensus had negligible coverage; broad OCR keyword denials produced severe
false positives; and prior review classifiers either regressed folds or
created catastrophic approvals.

## Reproduction and runtime boundary

Regenerate the aggregate report outside runtime artifacts:

```bash
python3 scripts/score_loss_atlas.py \
  --truth data/train_labels.csv \
  --submission /tmp/full1000-predictions.jsonl \
  --evaluation /tmp/full1000-evaluation.json \
  --case-scores /tmp/full1000-case-scores.jsonl \
  --output-json /tmp/full1000-score-loss-atlas.json \
  --output-markdown /tmp/full1000-score-loss-atlas.md \
  --target-score 148
```

`scripts/score_loss_atlas.py`, this report, truth labels, predictions, and
case-level evaluator output are development-only. The submitted Dockerfile
copies none of `evaluation/`, `devtools/`, `scripts/`, or `data/`. Runtime
code contains no lookup or import of this report.
