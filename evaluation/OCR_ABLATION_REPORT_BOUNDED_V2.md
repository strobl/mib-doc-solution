# Bounded OCR ablation report V2

This report records the ten-route WO-14 V2 measurement at source revision
`c062c46d87112367de9a6dc7f8112ae63ab4dc4e`. Visible checkbox pixels remain
unmeasured in V2. This is a public, label-exposed development diagnostic, not
an unseen/private evaluation and not an official leaderboard result.

## Cohort and method

- The cohort contains 32 PDFs selected before reading labels as the 32 smallest
  SHA-256 digests of PDF bytes in the 1,000-case public training corpus.
- After selection, the aggregate truth distribution was 8 approved, 12 denied,
  and 12 needs-review cases.
- Input-tree SHA-256:
  `baf4a2ec982dcd578a10c8f58182310ab60148aeb308388e468d712b7f042a5f`.
- Truth-subset SHA-256:
  `595c00b5bf1eb2f4e79c4fa5c8e131d8d14f16704f4f5f9771a9ae9e221193db`.
- The baseline and all ten registered one-variable removal variants ran in two
  fresh processes with four workers: 22 runs, 704 attempted, 704 answered, and
  0 omitted.
- Every pair produced byte-identical prediction JSONL and identical official
  evaluator results. Total observed wall time was 1,864.045 seconds and total
  process CPU time was 6,000.404 seconds.
- Environment: macOS 15.6.1 arm64, Python 3.12.13, Tesseract 5.5.0. Peak memory
  uses process `rusage`; it is not an exact constrained-Docker or cgroup peak.
- External authoritative aggregate JSON SHA-256:
  `acbe1e1c83761d45ed7970c4a6ca2560ffed59c7d51c0e9d853413d86cbf25b2`.
  Predictions, case-level scores, the truth subset, and the private selection
  manifest remain outside the repository.

## Baseline

- Total score: 135.560412/150.
- Extraction: 45.763889; classification: 73.750000; calibration: 16.046523.
- Mean Brier error: 0.098836916.
- Median CPU: 296.104 seconds; median wall: 95.429 seconds.
- Maximum sampled process RSS: 2,143.656 MiB.
- Catastrophic false approvals: 0; missing/invalid records: 0.
- Both prediction files had SHA-256
  `6f8d619ccc3ba5ec4f37853213e782774ead4fe3c5a4e8c72f9270abdb38f8e3`,
  identical to the V1 baseline despite the harness-only source change.

## Recorded ten-variant one-variable ledger

`Score contribution` is the score with the technique enabled minus the score
with it removed. `Safety: pass` means complete output plus no increase in
catastrophic false approvals; it is not a comprehensive adversarial or Docker
safety claim.

| Rank | Technique removed in variant | Score contribution | Gain / enabled-run CPU second | Incremental CPU | Deterministic | Safety |
|---:|---|---:|---:|---:|---|---|
| 1 | Targeted RapidOCR for unresolved fields | +2.967225 | 0.010020873 | +142.194 s | yes | pass |
| 2 | Bounded render-time deskew | +2.624265 | 0.008862634 | +53.465 s | yes | pass |
| 3 | Bounded fee-row threshold consensus | +0.134910 | 0.000455616 | -14.296 s | yes | pass |
| — | Selective PSM 6 refinement | 0.000000 | 0.000000000 | +7.836 s | yes | pass |
| — | Cross-view consensus | 0.000000 | 0.000000000 | +19.773 s | yes | pass |
| — | Sparse intake-crop consensus | 0.000000 | 0.000000000 | +8.407 s | yes | pass |
| — | Trusted applicant-scope repair | 0.000000 | 0.000000000 | +8.841 s | yes | pass |
| — | Risk-row geometry retry | 0.000000 | 0.000000000 | +9.086 s | yes | pass |
| — | Stamp/correction/watermark/strikethrough cue interpretation | 0.000000 | 0.000000000 | +10.063 s | yes | pass |
| reject | Orientation retry | -0.553041 | -0.001867723 | +11.577 s | yes | pass |

The negative incremental CPU value for fee-row consensus is timing noise on
this small concurrent cohort, so no incremental-efficiency claim is made for
it. The stable full-run efficiency metric ranks Targeted RapidOCR first and
bounded deskew second.

The direct runtime observations for each technique-disabled variant were:

| Variant | Median CPU | Median wall | Maximum sampled RSS |
|---|---:|---:|---:|
| `without_psm6_refinement` | 288.268 s | 91.143 s | 2,736.719 MiB |
| `without_cross_view_consensus` | 276.331 s | 84.857 s | 2,340.078 MiB |
| `without_fee_threshold_consensus` | 310.401 s | 94.909 s | 2,531.000 MiB |
| `without_sparse_intake_crop_consensus` | 287.697 s | 89.235 s | 2,504.156 MiB |
| `without_orientation_retry` | 284.527 s | 87.010 s | 2,350.922 MiB |
| `without_trusted_scope_repair` | 287.264 s | 88.929 s | 2,503.625 MiB |
| `without_risk_geometry_retry` | 287.018 s | 88.684 s | 2,381.531 MiB |
| `without_renderer_deskew` | 242.639 s | 72.406 s | 2,315.812 MiB |
| `without_visible_cue_interpretation` | 286.041 s | 88.622 s | 2,485.375 MiB |
| `without_targeted_rapidocr` | 153.910 s | 50.799 s | 393.500 MiB |

## Measured effects

Targeted RapidOCR remains the strongest measured technique on this 32-PDF
diagnostic cohort. Its +2.967225 contribution comprises +1.076389 extraction,
+1.875000 classification, and +0.015836 calibration points. The 31 raw field
points were risk flags 8, species 6, sponsor 5, visa 5, arrival date 4, and
declared purpose 3.

Bounded render-time deskew is the new second-place finding at +2.624265:
+0.173611 extraction, +2.187500 classification, and +0.263154 calibration.
Its five raw extraction points were all sponsor-ID points. With deskew
disabled, one needs-review truth case became an incorrect approval; enabling
deskew restored the correct needs-review outcome. Neither side produced a
catastrophic false approval.

Fee-row threshold consensus added four raw fee points, equivalent to
+0.138889 extraction points. A -0.003979 calibration offset left a net
+0.134910 contribution.

Removing orientation retry left every scored non-confidence field and every
classification result unchanged but improved calibration by 0.553041. It is
therefore rejected as a WO-15 winner and its confidence interaction is left
for the post-decision WO-19 calibration work.

The six zero-delta routes were not exercised or did not change final outputs
on this bounded cohort. In particular, the combined
stamp/correction/watermark/strikethrough cue path is independently measured.
That route excludes checkbox pixels. Zero here is not proof that a route has
no corpus-wide or adversarial value.

## Recommendation and gate

Carry the already uncertainty-routed Targeted RapidOCR path into WO-15 first
and preserve bounded deskew and fee-row consensus as measured supporting
techniques. Add complete physical-observation provenance plus explicit
resolved/unknown/contested semantics before any promotion. Do not broaden
RapidOCR to an unconditional full-page pass.

All ten V2 routes are represented by an explicit one-variable measurement or a
measured zero/negative result. Visible checkbox pixels remain outstanding, so
WO-14 stays in progress until a separate deterministic checkbox ablation is
recorded. This report selects implementation-review order only. WO-15 still
requires repeated group-exclusive public-data robustness evidence, linked
score deltas, complete provenance, and the normal zero-catastrophic,
zero-missing, and zero-invalid gates.
