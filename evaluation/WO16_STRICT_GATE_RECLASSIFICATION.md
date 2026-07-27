# WO-16 Strict-Gate Reclassification

This is an aggregate-only reclassification of the already published WO-16
evidence, not a new evaluation run. The original evidence bytes remain
unchanged and retain their historical `passed` result under the earlier gate.
They are bound here by SHA-256:

- Evidence:
  `evaluation/WO16_GROUPED_FUSION_EVIDENCE.json`
- Evidence SHA-256:
  `9526017cad8d59d340303a8edca151e34ea985bb917a812d41082743d4959717`
- Strict gate:
  `devtools/grouped_fusion_gate.py`
- Strict-gate source SHA-256:
  `7717c337f35fb0048b0d1ee0654d63c2c0c5ca819fd6b6f098f1b92772519ae9`

## Reclassification

| Aggregate fact | Value |
| --- | ---: |
| Repeats | `3` |
| Evaluated folds | `15` |
| Positive folds | `8` |
| Zero folds | `1` |
| Negative folds | `6` |
| Leave-best-fold-out positive | `true` |
| No negative folds | `false` |
| **Strict result** | **`BLOCKED`** |

The strengthened parent-program rule requires every fold to be non-negative
and every repeat to remain positive after its strongest fold is removed. The
historical candidate meets the second condition but fails the first in 6 of 15
folds. Its blocking reason is therefore exactly `no_negative_folds`.

## Authority boundary

- No historical evidence file or governed ledger is rewritten.
- No evaluation is presented as newly executed.
- No candidate is promoted and no candidate-state entry is created.
- The original 32-case result remains public-label-exposed robustness
  evidence, not an unseen or protected result.
- WO-16 remains in progress until a current-source, preregistered exact 3×5
  grouped evaluation passes the strict gate and the trusted Docker/adversarial
  confirmation is complete.
