# WO-10 Full-Public Diagnostic at `95ceca6`

This is a retrospective diagnostic of the current Phase 5–8 production graph,
not a governed promotion result. The run started before its hypothesis and
primary variable were preregistered, and all 1,000 public labels were already
exposed. It is therefore not an unseen holdout, protected-fold result,
private-test score, leaderboard score, WO-10 milestone, or candidate-state
update.

## Exact source and population

| Item | Value |
| --- | --- |
| Repository commit | `95ceca6847b374fdcb74ae2ef4044bef088f689d` |
| Production entry point | `solution.py` |
| Evaluator | `mib_weighted_v1` |
| Public truth / submitted / scored rows | `1,000 / 1,000 / 1,000` |
| Missing / extra / duplicate rows | `0 / 0 / 0` |
| Invalid adjudication / confidence / fee rows | `0 / 0 / 0` |
| Catastrophic false approvals | `0` |

The exact predictions, case scores, and evaluator JSON remain outside the
repository. Only their aggregate facts and cryptographic bindings are retained
here.

## Aggregate result

| Component | `95ceca6` | Governed `bf6c009` baseline | Delta |
| --- | ---: | ---: | ---: |
| Extraction | `42.91888888888889` | `44.87777777777778` | `-1.9588888888888931` |
| Classification | `65.01` | `68.52000000000001` | `-3.510000000000005` |
| Calibration | `16.610224260718248` | `16.974076455665433` | `-0.3638521949471851` |
| **Total** | **`124.53911314960715`** | **`130.37185423344323`** | **`-5.832741083836083`** |

Mean confidence Brier error was `0.08474439348204374`.

### Confusion matrix

| Truth → output | Cases |
| --- | ---: |
| `APPROVED → APPROVED` | `119` |
| `APPROVED → DENIED` | `2` |
| `APPROVED → NEEDS_REVIEW` | `168` |
| `DENIED → DENIED` | `360` |
| `DENIED → NEEDS_REVIEW` | `71` |
| `NEEDS_REVIEW → APPROVED` | `1` |
| `NEEDS_REVIEW → DENIED` | `6` |
| `NEEDS_REVIEW → NEEDS_REVIEW` | `273` |

Relative to the governed baseline, the dominant classification regression is
the loss of 45 correct approvals together with 53 additional
`APPROVED → NEEDS_REVIEW` outcomes. This is consistent with the later WO-17
policy hardening disabling the guarded initial review-to-approval heads. The
orientation expansion introduced later in the branch is the leading
extraction-regression suspect. These are hypotheses for one-variable,
preregistered experiments; this report does not adopt either rollback.

## External artifact bindings

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| Predictions JSONL | `eed453743b7ffff47307797c6156c6acb6a4c515e5fb81ee3fc48f497dc15f86` | `320,948` |
| Case-score JSONL | `870cd12dbbc85fc7370e77c7313518b3a67048243925038eda1d20b47f983681` | `1,176,483` |
| Evaluation JSON | `d7c74069c54a25772e5f032b5ceb308b04ef2db0e8512c078ab3baf83eb523ca` | `1,369` |

## Runtime observation

| Measure | Seconds |
| --- | ---: |
| Wall | `1,361.29` |
| User CPU | `4,359.39` |
| System CPU | `318.93` |

The run attempted and answered all 1,000 records without a retry. Runtime was
observed on the local host and is not a clean-Docker WO-20 certification.

## Disposition

- The latest governed passing candidate remains `bf6c009` at
  `130.37185423344323 / 150`.
- `95ceca6` is rejected as a score candidate and retained only as diagnostic
  evidence.
- The next experiment must be preregistered, change one primary variable, use
  exact 3×5 layout-grouped public robustness evidence, preserve zero
  catastrophic/missing/invalid regressions, and remain non-promotional unless
  separately authorized protected evidence passes.
