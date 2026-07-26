# Bounded OCR ablation report V3

This report extends the WO-14 measurement inventory by combining the ten-route
V2 diagnostic with a separate visible-checkbox supplement. Template
registration and bounded contrast remain unmeasured at this revision, so V3 is
not scope-complete. It is a public, label-exposed development diagnostic, not
an unseen/private evaluation and not an official leaderboard result. No
candidate from this report is integrated into the submitted production
runtime.

## Cohort and comparison bridge

- The fixed cohort contains 32 PDFs selected before reading labels as the 32
  smallest SHA-256 digests of PDF bytes in the 1,000-case public training
  corpus.
- Input-tree SHA-256:
  `baf4a2ec982dcd578a10c8f58182310ab60148aeb308388e468d712b7f042a5f`.
- Truth-subset SHA-256:
  `595c00b5bf1eb2f4e79c4fa5c8e131d8d14f16704f4f5f9771a9ae9e221193db`.
- V2 measured the baseline and ten removal variants at
  `c062c46d87112367de9a6dc7f8112ae63ab4dc4e`.
- The checkbox supplement measured a fresh baseline and the additive checkbox
  variant at `a2c3852a90177319d9ac2eebca70135e0b2bf199`.
- Both fresh supplement baselines, both checkbox runs, and the frozen V2
  baseline produced the same prediction SHA-256:
  `6f8d619ccc3ba5ec4f37853213e782774ead4fe3c5a4e8c72f9270abdb38f8e3`.
  Their official evaluator result was also identical at 135.560412/150.

The exact baseline bridge makes score effects comparable across the two source
revisions. Runtime deltas remain paired only within the same source revision.
Across V2 and the supplement there were 26 fresh runs, 832 attempted cases, 832
answers, and 0 omissions. Every repeated configuration produced
byte-identical predictions, identical evaluator results, and, where present,
identical route-activity counters.

## Ranked one-variable results

`Score contribution` is the score with the technique enabled minus the score
with it disabled. A safety pass means complete output and no increase in
catastrophic false approvals; it is not a Docker or adversarial-security claim.

| Rank | Technique | Score contribution | Median incremental CPU | Deterministic | Safety | Recommendation |
|---:|---|---:|---:|---|---|---|
| 1 | Targeted RapidOCR for unresolved fields | +2.967225 | +142.194 s | yes | pass | carry into WO-15 review |
| 2 | Bounded render-time deskew | +2.624265 | +53.465 s | yes | pass | preserve |
| 3 | Bounded fee-row threshold consensus | +0.134910 | -14.296 s | yes | pass | preserve; runtime delta is noise |
| — | Selective PSM 6 refinement | 0.000000 | +7.836 s | yes | pass | no measured gain |
| — | Cross-view consensus | 0.000000 | +19.773 s | yes | pass | no measured gain |
| — | Sparse intake-crop consensus | 0.000000 | +8.407 s | yes | pass | no measured gain |
| — | Trusted applicant-scope repair | 0.000000 | +8.841 s | yes | pass | no measured gain |
| — | Risk-row geometry retry | 0.000000 | +9.086 s | yes | pass | no measured gain |
| — | Stamp/correction/watermark/strikethrough cues | 0.000000 | +10.063 s | yes | pass | retain as a safety path |
| — | Exact checked fee-option pixels | 0.000000 | +1.148 s | yes | pass | do not promote; route unobserved |
| reject | Orientation retry | -0.553041 | +11.577 s | yes | pass | reject as a winner |

Only the three positive-gain techniques are ranked. The negative incremental
CPU value for fee consensus is timing noise on this small concurrent cohort,
so no incremental-efficiency claim is made for it.

## Visible-checkbox supplement

The development-only additive route requires all of the following before it
can emit one fee candidate:

- one exact case ID and one nearby `MIB Fee Receipt` / `Fee Status` anchor set;
- exactly one aligned `paid`, `unpaid`, `waived` option group in that order;
- one pixel-confirmed checked square and two pixel-confirmed empty squares;
- high-confidence OCR, clean untrusted-content checks, no correction,
  strikethrough, or sample watermark, no existing legible fee evidence, and no
  second qualifying page.

Each of the two runs scanned 146 rendered pages. Both recorded
`complete_groups=0`, `checked_groups=0`, `candidates_added=0`, and
`ambiguous_groups=0`. The predictions therefore remained byte-identical to the
baseline and the score contribution was 0.000000. This means the route was
honestly measured but unobserved in this bounded cohort; it does not prove
checkbox recovery has no value on another corpus.

The supplement baseline median was 286.072 CPU seconds and 88.769 wall
seconds. The enabled checkbox variant median was 287.220 CPU seconds and
88.935 wall seconds. Both sides were complete, deterministic, and had zero
catastrophic false approvals, missing records, and invalid records.

## Recommendation and gate

V3 has an explicit deterministic one-variable result for template-relative
crops, PSM refinement, bounded deskew, threshold views, targeted RapidOCR,
line/cell geometry, visible stamp/correction/strikethrough cues, visible
checkbox pixels, and cross-view consensus. Template registration and bounded
contrast remain for a later, separately paired supplement. Carry Targeted
RapidOCR into WO-15 first, while preserving bounded deskew and fee-row
consensus. Keep the checkbox candidate development-only because this cohort
contained no complete qualifying group.

WO-15 still requires group-exclusive robustness, complete physical-observation
provenance, explicit resolved/unknown/contested states, and the normal
zero-catastrophic, zero-missing, and zero-invalid gates before any production
promotion.

The compact JSON companion is authoritative. The frozen external V2 aggregate
has SHA-256
`acbe1e1c83761d45ed7970c4a6ca2560ffed59c7d51c0e9d853413d86cbf25b2`;
the external checkbox-supplement aggregate has SHA-256
`a9afefa3243ede327536be530529ffcb361d23f7cfe1b83dd0bb47bb35e3ac6d`.
Predictions, case-level scores, the truth subset, and the private selection
manifest remain outside the repository.
