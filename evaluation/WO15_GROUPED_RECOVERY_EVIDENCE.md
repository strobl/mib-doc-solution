# WO-15 grouped visible-evidence recovery

- Evidence class: `public_grouped_robustness_not_unseen`
- Status: **PASSED**
- Work Order acceptance: **PASS**
- Source revision: `de4f202317cb097b1d8f508d473237274345823e`
- Frozen layout manifest: `bf9e90f224945780d27f2cdb96030f7c222fa914f18dc16d15e4d6a81bc57bec`
- Records / layout groups: 32 / 9

## Paired score result

- Control: 132.315409307
- Candidate: 132.489020418
- Delta: +0.173611111

## Repeated grouped robustness

| Repeat | Weighted delta | Positive folds | Leave-best-fold-out delta |
| ---: | ---: | ---: | ---: |
| 1 | +0.173611111 | 1/5 | +0.000000000 |
| 2 | +0.173611111 | 1/5 | +0.000000000 |
| 3 | +0.173611111 | 1/5 | +0.000000000 |

## Safety and provenance

- Catastrophic false approvals: 0
- False-positive denial recovery delta: +0
- Missing / invalid / duplicate / extra records: 0 / 0 / 0 / 0
- Recovered fields with complete provenance: 8 / 8
- Serialization defaults used as evidence: 0

## Concentration diagnostics (non-hard)

| Diagnostic | Result |
| --- | :---: |
| `fold_majority_positive` | WARN |
| `leave_best_fold_out_positive` | WARN |

> **Concentration warning:** the positive score gain is concentrated in a minority of folds or becomes zero when the strongest fold is omitted. This is disclosed as a non-hard robustness warning; it does not change the Work Order acceptance result.

## Hard gates

| Gate | Result |
| --- | :---: |
| `public_exposed_evidence` | PASS |
| `manifest_frozen_before_scoring` | PASS |
| `group_exclusive` | PASS |
| `paired_fold_members` | PASS |
| `split_deterministic` | PASS |
| `run_deterministic` | PASS |
| `full_score_positive` | PASS |
| `repeat_weighted_deltas_positive` | PASS |
| `at_least_one_positive_fold_per_repeat` | PASS |
| `no_negative_folds` | PASS |
| `leave_best_fold_out_nonnegative` | PASS |
| `candidate_complete` | PASS |
| `control_complete` | PASS |
| `no_catastrophic_false_approvals` | PASS |
| `no_increased_false_positive_denial_recoveries` | PASS |
| `provenance_complete` | PASS |
| `no_serialization_default_as_evidence` | PASS |

> Public-label-exposed grouped robustness evidence; this is not an unseen holdout result.
