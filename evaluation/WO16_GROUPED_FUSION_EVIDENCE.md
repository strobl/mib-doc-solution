# WO-16 grouped applicant-aware evidence fusion

- Evidence class: `public_grouped_robustness_not_unseen`
- Status: **PASSED**
- Work Order acceptance: **PASS**
- Source revision: `c0ce148e85d2aae86da0e9afd589a6c92a1b2652`
- Frozen layout manifest: `bf9e90f224945780d27f2cdb96030f7c222fa914f18dc16d15e4d6a81bc57bec`
- Records / layout groups: 32 / 9
- Input PDFs / tree: 32 / `baf4a2ec982dcd578a10c8f58182310ab60148aeb308388e468d712b7f042a5f`
- Identity-free case cohort: `18d48664e5f5bd19f8d2738c329bce0f87296196ada4814376c564795d6ab8fb`
- Candidate observation binding: `56103551c02bf3ed2dbe7e19ae5b96641482b87c6771461f96a9d016b86dcd9e`
- Legacy-control observation binding: `3655851d848e24548bb0ac8e3fa96a8d6199bf8e2f728d6f31d9fa7700cec1d1`

## Legacy-control comparison

- Legacy-control revision: `de4f202317cb097b1d8f508d473237274345823e`
- Legacy control: 132.489020418
- Candidate: 136.186014331
- Delta: +3.696993913

## Repeated grouped robustness

| Repeat | Weighted delta | Positive folds | Leave-best-fold-out delta |
| ---: | ---: | ---: | ---: |
| 1 | +3.611516109 | 3/5 | +0.153579218 |
| 2 | +3.696993913 | 3/5 | +0.047370467 |
| 3 | +3.696993913 | 2/5 | +0.049124928 |

## Fusion safety and audit

- Catastrophic false-approval delta: +0
- False-positive denial delta: -1
- Newly introduced catastrophic false approvals: 0
- Newly introduced false-positive denials: 0
- Missing / invalid / duplicate / extra records: 0 / 0 / 0 / 0
- Changed fields with complete provenance: (resolver-level fusion vs legacy after current linkage; one accepted final resolver result per case): 1 / 1
- Correlated views collapsed: 10
- Independent-agreement resolutions: 71
- Same-rank contested outcomes: 1
- Cross-applicant candidates excluded: 71
- Clean higher-authority / binding-authority overrides: 0 / 0
- Text-layer winners: 0
- Serialization defaults used as evidence: 0

## Concentration diagnostics (non-hard)

| Diagnostic | Result |
| --- | :---: |
| `no_negative_folds` | WARN |
| `leave_best_fold_out_nonnegative` | PASS |
| `fold_majority_positive` | WARN |
| `leave_best_fold_out_positive` | PASS |

> **Robustness warning:** the aggregate repeated-CV gain is positive, but it is not uniform across layout folds and/or becomes non-positive when the strongest fold is omitted. This is disclosed as a non-hard diagnostic; the Work Order requires positive repeated grouped-CV deltas and no safety regression.

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
| `candidate_complete` | PASS |
| `legacy_control_complete` | PASS |
| `no_increased_catastrophic_false_approvals` | PASS |
| `no_increased_false_positive_denials` | PASS |
| `no_new_catastrophic_false_approvals` | PASS |
| `no_new_false_positive_denials` | PASS |
| `changed_fields_nonvacuous` | PASS |
| `changed_field_provenance_complete` | PASS |
| `no_clean_higher_authority_overrides` | PASS |
| `no_binding_authority_overrides` | PASS |
| `no_text_layer_winners` | PASS |
| `no_serialization_default_as_evidence` | PASS |

> Public-label-exposed grouped robustness evidence; this is not an unseen holdout result.
