# Bounded OCR ablation report

This report records the WO-14 measurement at source revision
`67aaa95723ffa280b392435095e91f3d2190139a`. It is a public,
label-exposed development diagnostic, not an unseen/private evaluation and not
an official leaderboard result.

## Cohort and method

- The cohort contains 32 PDFs selected before reading labels as the 32 smallest
  SHA-256 digests of PDF bytes in the 1,000-case public training corpus.
- After selection, the aggregate truth distribution was 8 approved, 12 denied,
  and 12 needs-review cases.
- Input-tree SHA-256:
  `baf4a2ec982dcd578a10c8f58182310ab60148aeb308388e468d712b7f042a5f`.
- Truth-subset SHA-256:
  `595c00b5bf1eb2f4e79c4fa5c8e131d8d14f16704f4f5f9771a9ae9e221193db`.
- The baseline and all eight registered one-variable removal variants ran in
  two fresh processes with four workers: 18 runs, 576 attempted, 576 answered,
  and 0 omitted.
- Each pair produced byte-identical prediction JSONL and identical official
  evaluator results. Total observed wall time was 1,531.147 seconds and total
  process CPU time was 4,910.632 seconds.
- Environment: macOS 15.6.1 arm64, Python 3.12.13, Tesseract 5.5.0. Peak memory
  uses process `rusage`; it is not an exact constrained-Docker or cgroup peak.
- External aggregate JSON SHA-256:
  `9948f15962352bce61982959cf642f99fe20e80927ba2f8dbc11b519a02e530c`.
  Predictions, case-level scores, the truth subset, and the private selection
  manifest remain outside the repository.

## Baseline

- Total score: 135.560412/150.
- Extraction: 45.763889; classification: 73.750000; calibration: 16.046523.
- Mean Brier error: 0.098836916.
- Median CPU: 289.230 seconds; median wall: 91.647 seconds.
- Maximum sampled process RSS: 1,989.391 MiB.
- Catastrophic false approvals: 0; missing/invalid records: 0.

## Complete one-variable ledger

`Score contribution` is the score with the technique enabled minus the score
with it removed.

| Rank | Technique removed in variant | Score contribution | Gain / enabled-run CPU second | Incremental CPU | Deterministic | Safety |
|---:|---|---:|---:|---:|---|---|
| 1 | Targeted RapidOCR for unresolved fields | +2.967225 | 0.010259067 | +134.907 s | yes | pass |
| 2 | Bounded fee-row threshold consensus | +0.134910 | 0.000466446 | -18.073 s | yes | pass |
| — | Selective PSM 6 refinement | 0.000000 | 0.000000000 | +7.416 s | yes | pass |
| — | Cross-view consensus | 0.000000 | 0.000000000 | +15.168 s | yes | pass |
| — | Sparse intake-crop consensus | 0.000000 | 0.000000000 | +2.311 s | yes | pass |
| — | Trusted applicant-scope repair | 0.000000 | 0.000000000 | +2.293 s | yes | pass |
| — | Risk-row geometry retry | 0.000000 | 0.000000000 | -1.589 s | yes | pass |
| reject | Orientation retry | -0.553041 | -0.001912118 | +5.318 s | yes | pass |

The negative incremental CPU values for fee and risk are timing noise on this
small concurrent cohort, so no incremental-efficiency claim is made for them.
The stable full-run efficiency metric ranks Targeted RapidOCR first at
0.010259 score points per enabled-run CPU second and fee-row consensus second
at 0.000466.

The direct runtime observations for each technique-disabled variant were:

| Variant | Median CPU | Median wall | Maximum sampled RSS |
|---|---:|---:|---:|
| `without_psm6_refinement` | 281.814 s | 87.576 s | 2,239.641 MiB |
| `without_cross_view_consensus` | 274.061 s | 84.543 s | 2,031.719 MiB |
| `without_fee_threshold_consensus` | 307.303 s | 93.983 s | 2,663.078 MiB |
| `without_sparse_intake_crop_consensus` | 286.919 s | 89.050 s | 2,573.766 MiB |
| `without_orientation_retry` | 283.912 s | 86.757 s | 2,364.500 MiB |
| `without_trusted_scope_repair` | 286.937 s | 88.950 s | 2,475.016 MiB |
| `without_risk_geometry_retry` | 290.818 s | 92.053 s | 2,183.297 MiB |
| `without_targeted_rapidocr` | 154.322 s | 51.014 s | 343.203 MiB |

Here, `Safety: pass` means complete output plus no increase in catastrophic
false approvals. It is not a comprehensive adversarial or Docker safety
claim.

## Measured effects

Targeted RapidOCR is the clear winner on this 32-PDF diagnostic cohort.
Removing it reduced extraction by 1.076389 points, classification by 1.875000
points, and calibration by 0.015836 points. Its visible-field contribution
comprised 31 raw extraction points: risk flags 8, species 6, sponsor 5, visa 5,
arrival date 4, and declared purpose 3. It added no catastrophic false
approval.

Fee-row threshold consensus added four raw fee points, equivalent to
+0.138889 extraction points. A -0.003979 calibration offset left a net
+0.134910 total-score contribution, again with no safety regression.

Removing orientation retry left every scored non-confidence field and every
classification result unchanged but improved calibration by 0.553041. This
small-cohort result does not justify changing decision logic; it rejects
orientation retry as a WO-15 winner and flags its confidence interaction for
the post-decision WO-19 calibration work.

The five zero-delta routes were not exercised or did not change final outputs
on this bounded cohort. Zero here is not proof that they have no corpus-wide
value.

## Recommendation and gate

Carry the already bounded, uncertainty-routed Targeted RapidOCR path into
WO-15 first, preserve the fee-row consensus as the secondary technique, and
add complete physical-observation provenance plus explicit
resolved/unknown/contested semantics. Do not broaden RapidOCR to an
unconditional full-page pass.

This diagnostic is sufficient to select the implementation review order, but
not to promote a release candidate. WO-15 still requires repeated
group-exclusive public-data robustness evidence, linked score deltas, complete
provenance, and the normal zero-catastrophic/zero-missing/zero-invalid gates.

This first bounded report also does not independently isolate renderer deskew
or the combined checkbox/stamp/correction/strikethrough cue path. The measured
fee/sparse variants cover template-relative normalized crops, fee views cover
bounded threshold/contrast, and risk geometry covers the risk-cell route.
WO-14 remains in progress until the two remaining one-variable routes are
measured in the same governed format.
