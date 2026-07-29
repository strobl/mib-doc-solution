# WO-19 Final Confidence Refit

This is aggregate public-data robustness evidence, not an unseen, private, official-validation, or leaderboard score.

## Frozen graph

- Source revision: `93daa737ca50d560815884c01660c60c5048aaa1`
- Records: `1000`
- Label-blind layout groups: `24`
- Current mean Brier: `0.078607791354`
- Current calibration score: `16.855688345822/20`
- Tracked mean-Brier target: `0.01`

## Repeated grouped OOF comparison

| Family | OOF Brier | Improvement | Minimum fold | Positive folds | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `temperature` | 0.078516278752 | +0.000091512602 | -0.000040226670 | 10/15 | reject |
| `beta` | 0.077730958200 | +0.000876833155 | -0.003758485907 | 9/15 | reject |
| `hierarchical_shrunk` | 0.074850244658 | +0.003757546697 | -0.001321574317 | 12/15 | reject |
| `isotonic` | 0.077556443379 | +0.001051347976 | -0.006253808754 | 11/15 | reject |

Diagnostic best family: `hierarchical_shrunk`.
Decision: **evaluated_no_promotion**.

Every family improved aggregate OOF Brier, but a negative layout fold is a hard rejection. No confidence artifact is promoted.

## Hard gates

- `non_confidence_bytes_unchanged`: `true`
- `all_families_compared`: `true`
- `every_repeat_positive_for_selected`: `true`
- `no_negative_fold_for_selected`: `false`
- `mean_brier_target_met`: `false`

The shadow comparison changes confidence only; all non-confidence output bytes remain invariant.

## Reliability coverage

### Final class

| Slice | Support | Brier | ECE (10-bin) |
| --- | ---: | ---: | ---: |
| `APPROVED` | 573 | 0.014595273418 | 0.006168592507 |
| `DENIED` | 1179 | 0.030937404376 | 0.014629564981 |
| `NEEDS_REVIEW` | 1248 | 0.144000354604 | 0.090482143927 |

### Finalizer route

| Slice | Support | Brier | ECE (10-bin) |
| --- | ---: | ---: | ---: |
| `decision:APPROVED->NEEDS_REVIEW` | 105 | 0.153485134268 | 0.306928966040 |
| `decision:DENIED->NEEDS_REVIEW` | 21 | 0.184834190913 | 0.424447042384 |
| `decision:NEEDS_REVIEW->APPROVED` | 54 | 0.153833733383 | 0.151601480916 |
| `fields_only` | 411 | 0.093041817815 | 0.035802647944 |
| `unchanged` | 2409 | 0.065589899602 | 0.022124250937 |

## Limitations

- All 1,000 public labeled cases are exposed development data; grouped OOF is robustness evidence, not an unseen or private score.
- The final output does not expose internal policy/recovery trace features, so reliability is reported by the observable outer finalizer route.
- No calibrator may enter runtime unless every strict fold and repeat gate passes on the frozen decision graph.
