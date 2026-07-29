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

## Required dimension atlas

WO-13 requires ten dimensions. This pass measures field,
adjudication-confusion, and confidence-bucket loss on the exact frozen
1,000-case result. It also presents a clearly qualified auxiliary historical
page/template grouping and cites the frozen whole-run runtime. Five
production-trace dimensions and exact current-source page/runtime allocation
remain explicitly blocked; they are not reported as zero-loss findings.

| Dimension | Status |
| --- | --- |
| Field | Measured from official case scores |
| Adjudication confusion | Measured from official case scores |
| Page/template family | Auxiliary historical only: exact case set, unresolved source/input-tree binding |
| Provenance route | Blocked: current-source trace absent |
| Applicant-linking state | Blocked: current-source trace absent |
| Evidence conflict | Blocked: current-source trace absent |
| OCR/recovery path | Blocked: current-source trace absent |
| Policy trace | Blocked: current-source trace absent |
| Confidence bucket | Frozen-baseline measurement from final submitted confidence |
| Runtime cost | Frozen whole-run total cited; per-case loss allocation blocked |

### Page/template-family loss

The auxiliary page/template grouping comes from the label-blind
`page-count-plus-first-page-ink-v1` layout manifest. It covers exactly the same
1,000 case IDs as the frozen evaluator artifacts and is bound by SHA-256
`d7aac395c2d42dc42128ba3b4ce15fef6c42c37a6e247a066c267fba8a514b7c`.
The grouping uses only PDF page count and first-page rendered pixels; labels
are not construction inputs. However, that WO-12 manifest does not bind the
exact `bf6c009` source revision and input-tree bytes recorded by WO-11.
Therefore its status is `auxiliary_historical`, not `current_source`.

There are 24 raw layout groups. Thirteen groups below the `K=10` literal
support floor are combined into one 35-case aggregate suppression bucket,
leaving 12 reported rows. No emitted category has support below 10. Every loss
column sums back to the corresponding frozen component gap.

| Layout family | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `page-count-03__ink-bucket-00` | 239 | 190 | 81 | 1.227778 | 4.920000 | 1.070791 |
| `page-count-03__ink-bucket-01` | 113 | 124 | 49 | 0.757778 | 3.050000 | 0.607896 |
| `page-count-04__ink-bucket-00` | 137 | 140 | 18 | 0.808889 | 1.100000 | 0.354537 |
| `page-count-05__ink-bucket-00` | 178 | 128 | 9 | 0.704444 | 0.590000 | 0.311787 |
| `page-count-04__ink-bucket-01` | 57 | 37 | 11 | 0.227778 | 0.670000 | 0.149506 |
| `page-count-05__ink-bucket-01` | 68 | 44 | 4 | 0.243333 | 0.260000 | 0.128081 |
| `<suppressed_low_support>` | 35 | 55 | 3 | 0.304444 | 0.190000 | 0.110825 |
| `page-count-06__ink-bucket-00` | 88 | 71 | 2 | 0.375556 | 0.130000 | 0.066693 |
| `page-count-03__ink-bucket-02` | 16 | 32 | 5 | 0.174444 | 0.320000 | 0.053217 |
| `page-count-04__ink-bucket-02` | 22 | 29 | 2 | 0.153333 | 0.120000 | 0.043196 |
| `page-count-06__ink-bucket-01` | 31 | 6 | 2 | 0.032222 | 0.130000 | 0.096167 |
| `page-count-05__ink-bucket-02` | 16 | 18 | 0 | 0.112222 | 0.000000 | 0.033225 |

The first two three-page layout families alone account for 11.634243 of the
19.628146-point residual. This is a prioritization result, not proof that page
count or ink density causes the errors.

### Confidence-bucket loss

Fixed confidence deciles use final submitted confidence only. Exact confidence
values and case identities are not emitted. One under-`K` fixed decile and one
complementary decile are coarsened into a 21-case suppression pool. This keeps
the allocation complete without revealing either category's individual
support. No emitted category has support below 10.

| Confidence | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.5-0.6` | 97 | 146 | 43 | 0.948889 | 2.760000 | 0.960931 |
| `0.2-0.3` | 79 | 79 | 53 | 0.523333 | 3.180000 | 0.718878 |
| `0.9-1.0` | 661 | 493 | 5 | 2.715556 | 0.340000 | 0.210072 |
| `0.0-0.1` | 30 | 22 | 27 | 0.151111 | 1.620000 | 0.108546 |
| `0.3-0.4` | 23 | 37 | 13 | 0.221111 | 0.780000 | 0.236346 |
| `0.1-0.2` | 20 | 21 | 17 | 0.103333 | 1.020000 | 0.101172 |
| `0.4-0.5` | 27 | 20 | 13 | 0.141111 | 0.780000 | 0.268699 |
| `<suppressed_low_support>` | 21 | 37 | 8 | 0.213333 | 0.510000 | 0.189825 |
| `0.8-0.9` | 42 | 19 | 7 | 0.104444 | 0.490000 | 0.231455 |

### Runtime cost

The governed baseline ledger records 2,567.7 seconds for the 1,000-case run
(`experiment_ledger.jsonl` sequence 1, record hash
`769b6ef7c1297cbddc32c0440d8b601e58850fc0fdc84391a0355b1454d58881`),
or 2.5677 seconds per case on average. That is a measured whole-run cost.
Because the frozen run did not retain per-case timings, allocating score loss
to runtime buckets remains blocked rather than inferred from page family.

### Current-source trace blockers

The frozen evaluator artifacts and output rows do not contain:

- provenance route;
- applicant-linking state;
- evidence-conflict category;
- accepted OCR/recovery path; or
- final policy-trace category.

`scripts/score_loss_atlas.py` now accepts an optional development-only
`--trace-dimensions` JSON artifact. Its exact versioned schema carries source
revision and input-tree identifiers and binds the truth SHA-256, submission
SHA-256, record count, and rows. The CLI computes the supplied file's SHA-256
itself. Source revision and input tree remain self-declared until compared with
an authoritative frozen-run manifest, so any supplied trace stays
`auxiliary_historical`. Each row contains exactly `case_id`, the five trace
dimensions, and `runtime_seconds`. Every trace category must belong to that
dimension's versioned allowlist. The tool rejects missing or extra cases,
unexpected fields, mismatched truth/submission hashes, malformed source/input
identifiers, non-finite runtime, and duplicate case IDs.

The committed atlas remains aggregate-only. Literal categories require
`K=10`. Under-K categories may share a suppression bucket only when that
bucket itself has at least ten cases; otherwise the residual is omitted.
Neither case IDs nor trace rows are retained.

No exact-case trace from the frozen `bf6c009` source and its exact input tree
exists, so those five
measurements cannot be backfilled honestly. A fresh current-source capture is
required before WO-13 can claim those dimensions as measured. Downstream
32-case WO-14/WO-17 observations are not substituted because they use
different revisions and populations.

Damage profile and difficulty also remain unmeasurable: those columns are
empty in the public truth and case-score artifacts. No private metadata is
invented.

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
  --layout-manifest /tmp/full1000-layout-manifest.json \
  --output-json /tmp/full1000-score-loss-atlas.json \
  --output-markdown /tmp/full1000-score-loss-atlas.md \
  --target-score 148
```

This checked-in Markdown is a curated narrative over the validated aggregates;
the command produces the canonical machine-rendered tables used to verify it
and is not expected to reproduce this narrative byte-for-byte.

Add `--trace-dimensions /tmp/full1000-trace-dimensions.json` only when a
source-bound exact-case capture exists. Without it, the tool records the five
trace dimensions as blockers.

`scripts/score_loss_atlas.py`, this report, truth labels, predictions, and
case-level evaluator output are development-only. The submitted Dockerfile
copies none of `evaluation/`, `devtools/`, `scripts/`, or `data/`. Runtime
code contains no lookup or import of this report.
