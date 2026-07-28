# WO-13 Full-Public Score-Loss Atlas

This is aggregate diagnostic evidence over public labeled training data. It is not an unseen holdout, validation, private-test, or leaderboard score.

## Evidence integrity

- Cross-artifact validation: `passed`
- Score version: `mib_weighted_v1`
- Source-hash basis: `file_bytes`
- `truth` SHA-256: `9c6210df4a600c9520435cf7d79d61d7113795dbf94b0e7ab3e39d237388bc8a`
- `submission` SHA-256: `d6e23641a4e4c7a5517c2b691791146177665f5c297667adae17565f6918a42d`
- `evaluation` SHA-256: `e0c285f3f8f5c5b148fd00d825746aa57fd46f51452189d7f16d1aef9e987001`
- `case_scores` SHA-256: `a67152a345221106a87e6c0f6e1bc5c23e7f272e500bf58071ef724f8a91ce5f`
- Dimension evidence `frozen_baseline_manifest` SHA-256: `f2a202dcb795be07ac4caee3ba6920f1eff9cc517bd96a5121d8df408ea84d22`
- Dimension evidence `layout_manifest` SHA-256: `d7aac395c2d42dc42128ba3b4ce15fef6c42c37a6e247a066c267fba8a514b7c`
- Dimension evidence `trace_dimensions` SHA-256: `848e37a29be1266fd940d585270d7c2975dbcf9efe979ee0d60a5bee3303490b`

## Score bridge

| Component | Current | Gap to perfect |
| --- | ---: | ---: |
| Extraction | 44.877778 | 5.122222 |
| Classification | 68.520000 | 11.480000 |
| Calibration | 16.974076 | 3.025924 |
| **Total** | **130.371854** | **19.628146** |

Target `148.00` requires `+17.628146` points, or `89.81%` of all remaining error.

## Extraction loss by field

| Field | Missed | Match rate | Score loss | Default among misses | Modal wrong output |
| --- | ---: | ---: | ---: | ---: | --- |
| `risk_flags` | 185 | 81.50% | 1.644444 | 170/185 | `<configured_default>` (170) |
| `applicant_name` | 123 | 87.70% | 0.683333 | 40/123 | `<configured_default>` (40) |
| `sponsor_id` | 108 | 89.20% | 0.600000 | 66/108 | `<configured_default>` (66) |
| `fee_status` | 123 | 87.70% | 0.546667 | 101/123 | `<configured_default>` (101) |
| `arrival_date` | 106 | 89.40% | 0.471111 | 86/106 | `<configured_default>` (86) |
| `visa_class` | 72 | 92.80% | 0.400000 | 4/72 | `med-3` (44) |
| `home_world` | 57 | 94.30% | 0.316667 | 57/57 | `<configured_default>` (57) |
| `species_code` | 38 | 96.20% | 0.253333 | 38/38 | `<configured_default>` (38) |
| `declared_purpose` | 62 | 93.80% | 0.206667 | 62/62 | `<configured_default>` (62) |

## Field-miss concentration

- Total field misses: `874`
- Mean misses per case: `0.874000`
- Cases with no misses: `543`
- Cases with exactly one miss: `238`
- Cases with multiple misses: `219`

### Pairwise miss co-occurrence

| Field pair | Co-misses | Expected | Enrichment |
| --- | ---: | ---: | ---: |
| `applicant_name` + `risk_flags` | 47 | 22.755 | 2.065x |
| `sponsor_id` + `arrival_date` | 34 | 11.448 | 2.970x |
| `applicant_name` + `sponsor_id` | 34 | 13.284 | 2.559x |
| `applicant_name` + `arrival_date` | 32 | 13.038 | 2.454x |
| `home_world` + `arrival_date` | 31 | 6.042 | 5.131x |
| `sponsor_id` + `risk_flags` | 30 | 19.980 | 1.502x |
| `sponsor_id` + `declared_purpose` | 28 | 6.696 | 4.182x |
| `visa_class` + `sponsor_id` | 28 | 7.776 | 3.601x |
| `arrival_date` + `fee_status` | 27 | 13.038 | 2.071x |
| `applicant_name` + `fee_status` | 27 | 15.129 | 1.785x |
| `arrival_date` + `risk_flags` | 27 | 19.610 | 1.377x |
| `arrival_date` + `declared_purpose` | 26 | 6.572 | 3.956x |
| `risk_flags` + `fee_status` | 26 | 22.755 | 1.143x |
| `visa_class` + `declared_purpose` | 24 | 4.464 | 5.376x |
| `sponsor_id` + `fee_status` | 23 | 13.284 | 1.731x |
| `applicant_name` + `visa_class` | 22 | 8.856 | 2.484x |
| `species_code` + `arrival_date` | 20 | 4.028 | 4.965x |
| `applicant_name` + `home_world` | 20 | 7.011 | 2.853x |
| `visa_class` + `arrival_date` | 19 | 7.632 | 2.490x |
| `declared_purpose` + `risk_flags` | 19 | 11.470 | 1.656x |
| `applicant_name` + `declared_purpose` | 18 | 7.626 | 2.360x |
| `home_world` + `fee_status` | 17 | 7.011 | 2.425x |
| `species_code` + `home_world` | 16 | 2.166 | 7.387x |
| `visa_class` + `risk_flags` | 16 | 13.320 | 1.201x |
| `home_world` + `sponsor_id` | 15 | 6.156 | 2.437x |
| `declared_purpose` + `fee_status` | 15 | 7.626 | 1.967x |
| `visa_class` + `fee_status` | 15 | 8.856 | 1.694x |
| `home_world` + `visa_class` | 14 | 4.104 | 3.411x |
| `species_code` + `sponsor_id` | 14 | 4.104 | 3.411x |
| `applicant_name` + `species_code` | 14 | 4.674 | 2.995x |
| `species_code` + `risk_flags` | 13 | 7.030 | 1.849x |
| `species_code` + `declared_purpose` | 12 | 2.356 | 5.093x |
| `species_code` + `visa_class` | 12 | 2.736 | 4.386x |
| `home_world` + `declared_purpose` | 12 | 3.534 | 3.396x |
| `species_code` + `fee_status` | 12 | 4.674 | 2.567x |
| `home_world` + `risk_flags` | 12 | 10.545 | 1.138x |

## Wrong-decision / extraction decoupling

| Decision outcome | All fields correct | One or more field misses |
| --- | ---: | ---: |
| Correct | 438 | 376 |
| Wrong | 105 | 81 |

## Classification and calibration loss

| Truth → output | Cases | Class score loss | All fields correct | Mean Brier | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `APPROVED->APPROVED` | 164 | 0.000000 | 110 | 0.010105 | 0.066289 |
| `APPROVED->DENIED` | 10 | 0.800000 | 0 | 0.340749 | 0.136299 |
| `APPROVED->NEEDS_REVIEW` | 115 | 6.900000 | 99 | 0.128652 | 0.591798 |
| `DENIED->DENIED` | 382 | 0.000000 | 233 | 0.008833 | 0.134962 |
| `DENIED->NEEDS_REVIEW` | 49 | 2.940000 | 1 | 0.139178 | 0.272789 |
| `NEEDS_REVIEW->APPROVED` | 6 | 0.420000 | 4 | 0.715167 | 0.171640 |
| `NEEDS_REVIEW->DENIED` | 6 | 0.420000 | 1 | 0.673556 | 0.161653 |
| `NEEDS_REVIEW->NEEDS_REVIEW` | 268 | 0.000000 | 95 | 0.139039 | 1.490493 |

## Output + confidence empirical calibration ceiling

- Groups: `76` (`predicted_adjudication_x_exact_submitted_confidence`)
- Empirical-oracle mean Brier: `0.061314`
- Empirical-oracle calibration: `17.547435/20`
- Calibration gain ceiling: `+0.573359`
- Total-score ceiling from this remapping alone: `130.945213/150`

In-sample upper bound from replacing each exact output+confidence group with its public-label empirical correctness rate; singleton groups make this optimistic and it is not validation evidence.

## Required dimension coverage

| Dimension | Status |
| --- | --- |
| `field` | `frozen_baseline` |
| `adjudication_confusion` | `frozen_baseline` |
| `page_template_family` | `current_source` |
| `provenance_route` | `current_source` |
| `applicant_linking_state` | `current_source` |
| `evidence_conflict` | `current_source` |
| `ocr_recovery_path` | `current_source` |
| `policy_trace` | `current_source` |
| `confidence_bucket` | `frozen_baseline` |
| `runtime_cost` | `current_source` |

Field and adjudication-confusion loss are quantified above. The tables below allocate loss across emitted K-safe groups. A dimension is only a complete reconciliation when no under-K residual was omitted.

### Page Template Family

- Status: `current_source`

| Category | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
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

### Provenance Route

- Status: `current_source`

| Category | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `visible_ocr` | 596 | 365 | 163 | 2.322222 | 9.930000 | 2.350270 |
| `mixed_visible_sources` | 110 | 201 | 23 | 1.123333 | 1.550000 | 0.631780 |
| `authoritative_source` | 294 | 308 | 0 | 1.676667 | 0.000000 | 0.043873 |

### Applicant Linking State

- Status: `current_source`
- Complementary suppression: an additional category was coarsened into the K-safe suppression pool

| Category | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linked_unique` | 657 | 417 | 172 | 2.603333 | 10.540000 | 2.601282 |
| `<suppressed_low_support>` | 49 | 149 | 14 | 0.842222 | 0.940000 | 0.380769 |
| `authoritative_scope` | 294 | 308 | 0 | 1.676667 | 0.000000 | 0.043873 |

### Evidence Conflict

- Status: `current_source`
- Complementary suppression: an additional category was coarsened into the K-safe suppression pool

| Category | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 855 | 614 | 172 | 3.672222 | 10.540000 | 2.643103 |
| `<suppressed_low_support>` | 61 | 154 | 14 | 0.877778 | 0.940000 | 0.381476 |
| `authority_conflict` | 84 | 106 | 0 | 0.572222 | 0.000000 | 0.001344 |

### Ocr Recovery Path

- Status: `current_source`
- Complementary suppression: an additional category was coarsened into the K-safe suppression pool

| Category | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `targeted_rapidocr` | 445 | 615 | 73 | 3.430000 | 4.630000 | 1.406064 |
| `primary` | 468 | 124 | 106 | 0.951111 | 6.390000 | 1.346794 |
| `multiple_recovery_paths` | 65 | 132 | 7 | 0.721111 | 0.460000 | 0.272535 |
| `<suppressed_low_support>` | 22 | 3 | 0 | 0.020000 | 0.000000 | 0.000530 |

### Policy Trace

- Status: `current_source`

| Category | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deterministic_policy` | 475 | 212 | 111 | 1.424444 | 6.750000 | 1.458851 |
| `recovery_review` | 126 | 181 | 53 | 1.052222 | 3.180000 | 0.896693 |
| `binding_authority` | 294 | 308 | 0 | 1.676667 | 0.000000 | 0.043873 |
| `recovery_denial` | 27 | 70 | 9 | 0.395556 | 0.720000 | 0.215461 |
| `needs_review_conflict` | 29 | 71 | 8 | 0.411111 | 0.480000 | 0.237846 |
| `revalidated_policy` | 49 | 32 | 5 | 0.162222 | 0.350000 | 0.173200 |

### Confidence Bucket

- Status: `frozen_baseline`
- Complementary suppression: an additional category was coarsened into the K-safe suppression pool

| Category | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
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

### Runtime Cost

- Status: `current_source`

| Category | Cases | Field misses | Wrong decisions | Extraction loss | Classification loss | Calibration loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `8s_or_more` | 513 | 749 | 80 | 4.168889 | 5.090000 | 1.678647 |
| `2s_to_under_4s` | 241 | 72 | 99 | 0.602222 | 5.950000 | 1.067093 |
| `4s_to_under_8s` | 246 | 53 | 7 | 0.351111 | 0.440000 | 0.280184 |

### Remaining measurement blockers

- None.

## Oracle ceilings

- `perfect_extraction_only`: 135.494076/150
- `perfect_classification_only`: 141.851854/150
- `perfect_calibration_only`: 133.397778/150
- `perfect_extraction_and_classification`: 146.974076/150
- `perfect_all_components`: 150.000000/150

## Limitations

- All public labeled cases have already been evaluated; this is diagnostic evidence, not an unseen holdout.
- Oracle ceilings are additive or empirical in-sample bounds, not expected gains from a concrete implementation.
- Field corrections can change adjudication and confidence, so component effects are not causally independent.
- Public labels may omit private difficulty, damage-profile, trap, and unrecoverable-field metadata.
- Exact confidence values and low-support or sensitive modal outputs are suppressed.
- Trace dimensions are authoritative only when exact output, source, input, layout, archive, runtime, and tool bindings all pass.
- Summed per-case latency is not batch wall time; batch wall time is reported separately when the trace contract supplies it.
- The report contains aggregates only and must never be converted into runtime per-case rules.
