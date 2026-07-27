# WO-12 repeated grouped-split demonstration

- Evidence class: `public_grouped_robustness_not_unseen`
- Status: **PASSED**
- Source revision: `cf65a79e1405b3171654369d858151569c687287`
- Evidence-tool source: `f90a8ed30baa1cc32e4b254a8530d86a863a1074975b6c6c5962d019c8d7a9d1`
- Canonical layout-manifest bytes: `d7aac395c2d42dc42128ba3b4ce15fef6c42c37a6e247a066c267fba8a514b7c`
- Input tree: `21e821aa3089b841683375da59cf961e679e10f7009e5332ea9e8582f00f4c8e`
- Taint registry / head: `c187216dd13db016e637396d545230a97feafaf2d8e857304a866bba7b6e556f` / `3721f84fe5e9b87547025f5f54eff77df09a7e70d0a92ebf9221bfcf881bf033`
- Records / layout groups: 1000 / 24
- Repeats / folds per repeat / total splits: 3 / 5 / 15

## Aggregate split counts

| Repeat | Fold | Tuning records | Tuning groups | Validation records | Validation groups |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 938 | 19 | 62 | 5 |
| 1 | 2 | 844 | 19 | 156 | 5 |
| 1 | 3 | 536 | 19 | 464 | 5 |
| 1 | 4 | 688 | 19 | 312 | 5 |
| 1 | 5 | 994 | 20 | 6 | 4 |
| 2 | 1 | 745 | 19 | 255 | 5 |
| 2 | 2 | 572 | 19 | 428 | 5 |
| 2 | 3 | 836 | 19 | 164 | 5 |
| 2 | 4 | 918 | 19 | 82 | 5 |
| 2 | 5 | 929 | 20 | 71 | 4 |
| 3 | 1 | 855 | 19 | 145 | 5 |
| 3 | 2 | 737 | 19 | 263 | 5 |
| 3 | 3 | 572 | 19 | 428 | 5 |
| 3 | 4 | 909 | 19 | 91 | 5 |
| 3 | 5 | 927 | 20 | 73 | 4 |

## Verified mechanics and evidence boundary

- Deterministic assignment: PASS
- Whole-group tuning/validation exclusivity: PASS
- Every manifest record and group appears in validation exactly once per repeat: PASS
- Synthetic exclusion-mechanics probe: PASS (1 record in 1 group; identities not emitted)
- Registry entries matching the layout-group namespace: 0 groups / 0 records
- Generic public-cohort taint token present: YES (1 matching event)
- Canonical manifest declares label-blind construction: PASS

> This demonstrates public grouped-robustness split mechanics only. The evaluated PDFs and labels are public, so no fold is represented as protected, private, pristine, or unseen.

> The generic public-cohort taint token is not bound to the layout-manifest or input-tree digest and does not attest that it identifies this exact population.

> This report makes no claim that the manifest was fixed before scoring. A future temporal claim requires the manifest hash and seed to be preregistered in a governed plan.
