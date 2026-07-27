# WO-17 late-recovery policy revalidation

- Evidence class: `public_grouped_robustness_not_unseen`
- Status: **PASSED**
- Source revision: `32ba65c934d1eb35c20f084276c1c9abbfcb586b`
- Legacy-control revision: `ae9ac224b5b9e0a7e25cd7a0d9197c573fdd5e0a`
- Frozen layout manifest: `bf9e90f224945780d27f2cdb96030f7c222fa914f18dc16d15e4d6a81bc57bec`
- Input PDFs / tree: 32 / `baf4a2ec982dcd578a10c8f58182310ab60148aeb308388e468d712b7f042a5f`
- Identity-free cohort binding: `18d48664e5f5bd19f8d2738c329bce0f87296196ada4814376c564795d6ab8fb`
- Candidate observations: `fa5db7edb3952d4933e2a67d8b3c24cf56e124c208a786cc1c59acebb2c5c3fe`
- Legacy observations: `764bc4a9cfbd95f6a2ff59f9f4e7c927f500b2cccdf3a09c10682ee746d36c55`
- Execution audits: `58690fa433ad38b957458898225c6c3907062236638930f792c333baaf9fd700`
- Contract fixtures: `c3ed71f632646aeb1151ae17d17cd8d1363b2673292284de843c6d9c000b3f37`

## Official evaluator

- Legacy control: 136.186014331
- Candidate: 136.297159217
- Delta: +0.111144885
- Extraction / classification / calibration deltas: +0.000000000 / +0.000000000 / +0.111144885

## Targeted confusion deltas

| Truth → prediction | Control | Candidate | Delta |
| --- | ---: | ---: | ---: |
| APPROVED → NEEDS_REVIEW | 3 | 3 | +0 |
| DENIED → NEEDS_REVIEW | 0 | 0 | +0 |

## Execution-order trace

| Aggregate trace | Frozen cohort | Contract probes |
| --- | ---: | ---: |
| Legacy synthetic decision before late recovery | — | 1 |
| Late recovery before revalidation | 10 | 1 |
| Revalidation after late recovery | 10 | 1 |
| Contradicted synthetic reasons removed | 1 | 1 |
| Independent denial reasons retained | 0 | 1 |
| Original review confidence restored | 1 | 1 |
| Signed late authority recovered | 1 | 3 |
| Late adjudication evidence preserved | 1 | 3 |
| Late biohazard evidence preserved | 0 | 1 |

> Cohort occurrence counters may be zero. The contract-probe column must be non-vacuous for every required branch.

## Safety

- Newly introduced catastrophic false approvals: 0
- Newly introduced false-positive denials: 0
- New missing / invalid records: 0 / 0
- Candidate missing / invalid / duplicate / extra records: 0 / 0 / 0 / 0
- Non-policy field changes: 0

## Hard gates

| Gate | Result |
| --- | :---: |
| `accepted_final_policy_results_complete` | PASS |
| `approved_to_review_non_regression` | PASS |
| `candidate_complete` | PASS |
| `cohort_contradictions_accounted` | PASS |
| `cohort_execution_order_accounted` | PASS |
| `cohort_unsafe_counters_zero` | PASS |
| `contract_candidate_order_exercised` | PASS |
| `contract_contradiction_removed` | PASS |
| `contract_execution_order_accounted` | PASS |
| `contract_independent_denial_retained` | PASS |
| `contract_late_biohazard_evidence_preserved` | PASS |
| `contract_late_decision_evidence_preserved` | PASS |
| `contract_legacy_order_exercised` | PASS |
| `contract_placeholder_guards_exercised` | PASS |
| `contract_policy_safety_guards_exercised` | PASS |
| `contract_review_confidence_restored` | PASS |
| `contract_signed_late_authority_recovered` | PASS |
| `contract_unsafe_counters_zero` | PASS |
| `denied_to_review_non_regression` | PASS |
| `execution_audits_deterministic` | PASS |
| `extraction_score_unchanged` | PASS |
| `legacy_control_complete` | PASS |
| `manifest_frozen_before_scoring` | PASS |
| `no_new_catastrophic_false_approvals` | PASS |
| `no_new_false_positive_denials` | PASS |
| `no_new_invalid_records` | PASS |
| `no_new_missing_records` | PASS |
| `non_policy_fields_unchanged` | PASS |
| `public_exposed_evidence` | PASS |
| `repeated_runs_deterministic` | PASS |
| `score_non_regression` | PASS |

> Public-label-exposed 32-case robustness evidence; this is not an unseen holdout result. Identity-bearing inputs and traces remain external.
