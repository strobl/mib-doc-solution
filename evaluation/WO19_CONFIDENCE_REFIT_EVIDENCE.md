# WO19 final-confidence refit evidence

- Status: `evaluated_no_promotion`
- Evidence scope: `public_grouped_robustness_not_unseen`
- S0 revision: `417550129f2884c1621e0a45a2cf629bc69130f5`
- Records / layout groups: 32 / 9
- Selected shadow family: `beta`
- Promotion recommendation: `false`
- Runtime action: `retained_current_s0_artifact`

## Exact runtime and byte gates

- Double truth-blind capture deterministic: `true`
- Normal solution.py/BatchRunner equals capture bytes: `true`
- All non-confidence bytes unchanged in shadow: `true`
- Refit and artifact bytes deterministic: `true`

## Official evaluator

| Arm | Total | Calibration | Mean Brier |
|---|---:|---:|---:|
| S0 baseline | 136.254041330120 | 17.816541330120 | 0.054586466747 |
| Selected shadow | 137.072877350672 | 18.635377350672 | 0.034115566233 |

The full-data shadow result above is an in-sample diagnostic. The required
group-exclusive out-of-fold result moved in the opposite direction:

- S0 OOF Brier: `0.054586466747`
- Selected beta OOF Brier: `0.057351715801`
- OOF Brier delta: `+0.002765249054`
- Repeat deltas: `+0.002758386515`, `+0.002739021134`,
  `+0.002798339512`
- Brier target `<= 0.01`: `not met`

The selected candidate failed all five CV promotion checks: mean Brier
improvement, every-repeat improvement, calibration-score improvement,
supported-slice non-regression, and leave-one-layout-group-out
non-regression.

## Decision

At least one promotion gate did not pass. The current S0 confidence artifact remains pinned and the shadow candidate remains outside the runtime composition.

The Brier target of 0.01 is tracked and reported, but it is not forced as a promotion gate. This is public-label-exposed grouped robustness evidence, not an unseen holdout result.
