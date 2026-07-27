# WO-18 Identity-Free Decision Recovery

**Status: BLOCKED AT PRECONDITION — candidate disabled, no model promoted.**

This is public-exposed grouped robustness evidence, not an unseen holdout and not the full-public 1,000-case evaluation.

## Why promotion is blocked

1. **Protected-role coverage is impossible on the frozen cohort.** The `binding_authority` and `approval_guard` roles have zero eligible cases and zero eligible layout groups. Assigning either role would fabricate evidence, so the required five-role gate cannot be evaluated honestly. Consequently, the standard `DecisionRecoveryGate` was not evaluated.
2. **The gated hybrid adds no value.** Across all three grouped OOF repeats its official-evaluator score is byte-for-byte equivalent to the deterministic engine, producing a delta of `0.000000` in every repeat. The gate requires a strictly positive delta.

## Four-arm official-evaluator results

| Approach | Mean score | Minimum | Maximum |
|---|---:|---:|---:|
| `deterministic_engine` | 136.254041 | 136.254041 | 136.254041 |
| `evidence_completion_only` | 136.254041 | 136.254041 | 136.254041 |
| `compact_identity_free_model` | 116.958709 | 111.103960 | 120.662361 |
| `gated_hybrid` | 136.254041 | 136.254041 | 136.254041 |

## Verified controls

- Two production captures are byte-identical and fully bound.
- All four arms contain exact 3×5 grouped OOF manifests.
- Every repeat score was recomputed with the official evaluator.
- Two contract audits match the executable fixture deterministically.
- Contract, feature-schema, model-source, and runtime leakage scans are clean.
- The historical CV/audit field named `protected_role_manifest_sha256` is treated only as a legacy-misnamed coverage-gap binding; it is not evidence of a valid protected-role manifest.
- The evaluated production composition does not include the candidate adjudicator.
- Identity-bearing cases, paths, truth rows, predictions, and folds remain external.

Evaluated source: `e4330940414c382dcab89816110ab2be2ff238cf`.
