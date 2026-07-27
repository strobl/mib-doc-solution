# Bounded OCR ablation report V4

This is the scope-complete WO-14 aggregate. It combines the frozen ten-route
V2 removal diagnostic, the V3 visible-checkbox supplement, and the V4
template-registration and bounded-contrast supplement. It is a public,
label-exposed development diagnostic, not an unseen/private evaluation and not
an official leaderboard result. The supplements remain development-only and
are not integrated into the submitted production runtime.

## Cohort, bridge, and evidence totals

- The fixed cohort contains 32 PDFs selected before reading labels as the 32
  smallest SHA-256 digests of PDF bytes in the 1,000-case public training
  corpus, producing 146 rendered pages.
- Input-tree SHA-256:
  `baf4a2ec982dcd578a10c8f58182310ab60148aeb308388e468d712b7f042a5f`.
- Truth-subset SHA-256:
  `595c00b5bf1eb2f4e79c4fa5c8e131d8d14f16704f4f5f9771a9ae9e221193db`.
- The work spans source revisions `c062c46d87112367de9a6dc7f8112ae63ab4dc4e`,
  `a2c3852a90177319d9ac2eebca70135e0b2bf199`, and
  `9d0cfe958e611963b12469b17ef80ec10ff93330`.
- Every fresh baseline at all three revisions produced prediction SHA-256
  `6f8d619ccc3ba5ec4f37853213e782774ead4fe3c5a4e8c72f9270abdb38f8e3`
  and the same official evaluator score, 135.560412/150.
- Across 32 fresh runs there were 1,024 attempted cases, 1,024 answers, 0
  omissions, 16 deterministic configuration pairs, 8,949.925 CPU seconds,
  2,765.754 wall seconds, and a 2,736.719 MiB maximum peak-memory observation.

The byte-identical baseline bridge makes score effects comparable across
revisions. Runtime comparisons remain paired only within the same source
revision.

## Ranked one-variable results

`Score contribution` is the score with the technique enabled minus the score
with it disabled. `Gain/enabled CPU` is the ranking metric. Incremental
efficiency is omitted when the measured incremental CPU denominator is
non-positive. A safety pass means complete output and no increase in
catastrophic false approvals.

| Rank | Technique | Enabled CPU | Disabled CPU | Incremental CPU | Score contribution | Gain/enabled CPU | Gain/incremental CPU | Result |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Targeted RapidOCR for unresolved fields | 296.104 s | 153.910 s | +142.194 s | +2.967225 | 0.010020873 | 0.020867448 | carry into WO-15 review |
| 2 | Bounded render-time deskew | 296.104 s | 242.639 s | +53.465 s | +2.624265 | 0.008862634 | 0.049083598 | preserve |
| 3 | Bounded fee-row threshold consensus | 296.104 s | 310.401 s | -14.296 s | +0.134910 | 0.000455616 | — | preserve; runtime delta is noise |
| — | Selective PSM 6 refinement | 296.104 s | 288.268 s | +7.836 s | 0.000000 | 0.000000000 | 0.000000000 | no measured gain |
| — | Cross-view consensus | 296.104 s | 276.331 s | +19.773 s | 0.000000 | 0.000000000 | 0.000000000 | no measured gain |
| — | Sparse normalized intake crop | 296.104 s | 287.697 s | +8.407 s | 0.000000 | 0.000000000 | 0.000000000 | no measured gain |
| — | Trusted applicant-scope repair | 296.104 s | 287.264 s | +8.841 s | 0.000000 | 0.000000000 | 0.000000000 | no measured gain |
| — | Risk-row line/cell geometry | 296.104 s | 287.018 s | +9.086 s | 0.000000 | 0.000000000 | 0.000000000 | no measured gain |
| — | Stamp/correction/watermark/strikethrough cues | 296.104 s | 286.041 s | +10.063 s | 0.000000 | 0.000000000 | 0.000000000 | retain as safety path |
| — | Exact checked fee-option pixels | 287.220 s | 286.072 s | +1.148 s | 0.000000 | 0.000000000 | 0.000000000 | unobserved; do not promote |
| — | Bounded template registration | 292.905 s | 286.580 s | +6.326 s | 0.000000 | 0.000000000 | 0.000000000 | unobserved; do not promote |
| reject | Bounded orientation retry | 296.104 s | 284.527 s | +11.577 s | -0.553041 | -0.001867723 | -0.047769802 | reject as winner |
| reject | Bounded low-contrast autocontrast | 321.983 s | 286.580 s | +35.403 s | -0.448521 | -0.001392995 | -0.012668893 | reject as winner |

Every row has two byte-identical repetitions per configuration, complete
32/32 output, zero missing or invalid records, zero catastrophic false
approvals on both sides, and a safety pass. The JSON companion records the
source revision, target fields, enabled side, both CPU medians, both efficiency
definitions, activity counters, and safety counts for every variant.

## Final preprocessing supplement

The registration candidate is label-blind, translation-only, requires a ruled
template signature, preserves RGB input, never shifts PDF-point text geometry,
and changes at most two eligible pages per case. Both runs scanned 146 pages,
found zero eligible frames, registered zero pages, and abstained on all 146.
Predictions and score were identical to baseline. The route was measured but
unobserved; it is not recommended for promotion.

The contrast candidate applies RGB autocontrast to at most two pages per case
only after a label-blind low-contrast gate. Both runs scanned 146 pages,
enhanced the same 37, and abstained on 109. The two enabled outputs were
byte-identical, but arrival-date extraction lost four raw points and total
score fell by 0.448521. The candidate is rejected as a winner.

## Scope closure and recommendation

The final inventory separately measures template registration, normalized fee
and intake crops, PSM refinement, bounded orientation retry, active-applicant
scope repair, bounded deskew, bounded contrast, threshold views, targeted
RapidOCR, risk-row line/cell geometry, visible checkbox pixels, visible
stamp/correction/watermark/strikethrough cues, and cross-view consensus.
`scope_complete` is true and `unmeasured_routes` is empty.

Carry targeted RapidOCR into WO-15 implementation review first. Preserve
bounded renderer deskew and bounded fee-row threshold consensus. Keep the
visible-cue path as a safety mechanism without claiming a score gain. Do not
promote the unobserved checkbox or registration candidates, and reject
orientation retry and bounded contrast as winners.

The compact JSON companion is authoritative. Its three source aggregates have
SHA-256 digests
`acbe1e1c83761d45ed7970c4a6ca2560ffed59c7d51c0e9d853413d86cbf25b2`,
`a9afefa3243ede327536be530529ffcb361d23f7cfe1b83dd0bb47bb35e3ac6d`,
and
`479727441ee9287fd6d7fe8b9a5dd13ff5f84b13c2bad4cdf996e353f2872d7a`.
Predictions, case-level scores, the truth subset, and the private selection
manifest remain outside the repository.
