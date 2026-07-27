# Work Order Reconciliation: WO-1 through WO-22

This is the repository-side reconciliation of all 22 Software Factory Work
Orders. It records the evidence state rather than treating `in_review` as
complete or treating a diagnostic score as a promoted candidate. The matrix
records the reconciled target state that the Factory is synchronized to after
this evidence is committed.

Snapshot date: 2026-07-27.

## Governing facts

- The latest governed passing full-public 1,000-case candidate remains
  `130.37185423344323 / 150`.
- The later source-bound full-public diagnostic at `95ceca6` scored
  `124.53911314960715 / 150` (extraction `42.91888888888889`,
  classification `65.01`, calibration `16.610224260718248`). It is a
  rejected, retrospective diagnostic, not a candidate-ledger promotion. Its
  `-5.832741083836083` delta from the frozen baseline is evidence that the
  current runtime lineage still needs governed revalidation.
- The later `132`–`137` totals are public-label-exposed 32-case diagnostics.
  They are not WO-10 milestones, protected promotions, official validation
  scores, private-test scores, or leaderboard results.
- The candidate-state ledger contains one passing entry: the frozen baseline.
- The historical WO-15 through WO-21 evidence was not preregistered. It is
  bound in `RETROSPECTIVE_RECONCILIATION_WO15_WO21.json` as retrospective,
  non-promotional evidence. Published ledgers were not backfilled.
- At the last external check during this reconciliation, official challenge
  PR #6 was open and unmerged. It remains organizer-controlled. No merge,
  resubmission, or official prediction regeneration is authorized by this
  reconciliation.

## 22/22 matrix

| WO | Reconciled target state | Reconciled outcome | Repository evidence / remaining gate |
| --- | --- | --- | --- |
| 1 | completed | Delivered | Docker/offline runtime scaffold was implemented and merged in `0d76c9b`; the current hardening is covered again by WO-20. |
| 2 | completed | Delivered | Batch runner and canonical twelve-field JSONL writer were implemented and merged in `da242e1`. |
| 3 | completed | Delivered | Render-first ingestion was implemented and merged in `26d342f`. |
| 4 | completed | Delivered | Visible OCR/CV extraction and untrusted-content filtering were implemented and merged in `df30f71`. |
| 5 | completed | Delivered | Evidence model, applicant scoping, and precedence resolver were implemented and merged in `c6deb7a`. |
| 6 | completed | Delivered | Deterministic adjudication and safety policy were implemented and merged in `d60c049`. |
| 7 | completed | Delivered | Pinned confidence calibration was implemented and merged in `2644110`. |
| 8 | completed | Delivered | Evaluation, runtime-artifact leakage, and catastrophic-false-approval gates were merged in `7ec3a46`; golden/adversarial regressions were subsequently upgraded from warnings to hard blocks in `cc269fa`. |
| 9 | completed | Delivered, external review pending | `origin/submission/strobl` contains the three-file challenge entry; commit `2244a8a` has 5,000 unique, schema-valid predictions with exact validation-manifest coverage. Form submission is a user attestation; at the last external check, official PR #6 was open and unmerged. |
| 10 | in_progress | Target not achieved | The latest passing candidate-ledger entry remains the `bf6c009` full-public baseline at `130.37185423344323`; no governed entry meets the 136/142/146/148 milestones. |
| 11 | completed | Delivered within its stated scope | `FULL_PUBLIC_1000_130_37.md` records a locally reproduced, source-pinned aggregate baseline, schema/coverage validity, and zero catastrophic false approvals on the exposed public labels. It discloses the 998+2 retry, externally retained raw artifacts, and missing clean-Docker certification; WO-11 requires Docker status to be recorded, not a Docker pass. |
| 12 | in_review | Implementation and real grouped-split demonstration ready for review | The experiment firewall now supports immutable two-stage plan/result events, strict adoption gates, protected-access binding, and durable retrospective reconciliation. `WO12_GROUPED_SPLIT_DEMONSTRATION.{json,md}` binds source `cf65a79`, all 1,000 public PDFs, 24 label-blind layout groups, and deterministic group-exclusive 3x5 mechanics. It explicitly claims public grouped robustness only—not protected, private, pristine, unseen, or pre-scoring evidence. `WO12_INPUT_TREE_DIGEST_RECONCILIATION.md` explains why its source-defined v2 tree digest is not directly comparable to WO-11's legacy digest and makes no unsupported population-equivalence claim. Published ledger heads remain unchanged. |
| 13 | in_review | Partial evidence ready; reviewer decision pending | `SCORE_LOSS_ATLAS_130_37.md` records the aggregate loss atlas and oracle ceilings without runtime lookup use, but explicitly lacks required page/template, provenance, linking, conflict, OCR-route, and policy-trace dimensions. |
| 14 | completed | Delivered within its bounded scope | The bounded OCR/layout/orientation ablation is scope-complete in `OCR_ABLATION_REPORT_BOUNDED_V4`; it recommends review order but does not promote a candidate. |
| 15 | blocked | Rejected by governing adoption gate | Historical WO-15 evidence passed its original gate with `+0.173611` on 32 public cases, but the governing program gate rejects it because only 3/15 folds are positive and the gain disappears after removing the strongest fold. This is `rejected_single_fold_concentration`, not a promotion. |
| 16 | in_progress | Historical evidence rejected; current-source revalidation pending | A narrowly scoped resolver change excludes cleanly separable foreign-case pages while mixed, unhinted, or conflicting scopes still fail closed. The historical 3x5 evidence is now rejected by the governing strict gate because 6/15 folds are negative (8 positive, 1 zero). A newly preregistered current-source 3x5 run plus clean trusted Docker/WO-21 confirmation is required; no waiver is used. |
| 17 | in_progress | Historical evidence recorded; governed current-source revalidation pending | The retrospective result was a `+0.111145` calibration-only 32-case diagnostic, while targeted confusion deltas remained zero. It is not a governed full-public milestone or promotion. The later full-public regression makes the guarded REVIEW-to-APPROVED suppression a leading classification hypothesis, so WO-17 cannot remain in review until an exact one-variable 3x5 experiment accepts or rejects that hypothesis. |
| 18 | blocked | No promotion | The exact four-arm comparison is `blocked_precondition`: two protected roles lack coverage and the standard promotion gate was not evaluated. No model entered runtime composition. |
| 19 | in_progress | Previous refit is stale; final decisions must freeze first | The selected beta refit worsened OOF Brier by `+0.002765249`; S0 remains pinned. The in-sample/shadow `137.072877` total is not promoted. Because the decision graph changed after the prior freeze/refit, confidence must be refit only after WO-16 and WO-17 are finally accepted or rejected. |
| 20 | in_progress | Local harness complete; exact-source Linux Docker evidence pending | The workflow binds source, input, image and model inventory, runs two independent constrained 5,000-case captures, and compares aggregate evidence fail-closed. A trusted run for `95ceca6` has passed preflight and is still running both captures, but the repository has since advanced to governance commit `cf65a79`; final acceptance requires a successful report whose provenance is bound to the release source selected for WO-22. |
| 21 | in_progress | Production-path host audit passes; exact-source trusted report pending | The real production processor passes all 15 scenarios twice locally with byte-identical captures, zero regressions, zero new approvals, and zero decoy adoption. The harness hard-gates all 13 categories and binds installed-model scanning plus the final report to the same-run WO-20 aggregate. Local model-root scanning and Docker/provenance attestations remain unavailable, so acceptance still requires the trusted workflow report bound to the release source. |
| 22 | backlog | Not startable | WO-15 and WO-18 are terminal rejected experiments, not blockers. WO-22 remains gated by WO-10, WO-11–14, WO-16–17, and WO-19–21. The program has not reached 148 and the current-source scoring, decision, confidence, Docker, and adversarial gates are unresolved, so no candidate freeze or resubmission package is authorized. |

## Current state totals

- `completed`: 11
- `in_review`: 2
- `in_progress`: 6
- `blocked`: 2
- `backlog`: 1
- total: 22

## Verification for this reconciliation change set

- Full local unit suite: 753 passed, 2 skipped.
- Focused experiment-control, strict fusion-gate, and grouped-split suite:
  159 passed.
- Real WO-12 public grouped-robustness demonstration: 1,000/1,000 PDFs,
  24 layout groups, 3 repeats x 5 folds, deterministic whole-group
  exclusivity, and exact once-per-repeat validation coverage. The external
  identity-bearing manifest was not committed.
- WO-11/WO-12 tree-digest framing is explicitly reconciled. The legacy WO-11
  hash has no recorded framing algorithm; WO-12 uses a source-defined
  length-prefixed basename plus raw file-digest stream. The two strings are
  not treated as directly comparable.
- Independent reviews of the governance tooling and final 22/22 matrix found
  no remaining P0-P2 issue after the wording corrections recorded here.
- Real local WO-21 production-path audit: 15/15 scenarios passed twice,
  byte-identical, with zero regressions, new approvals, or decoy adoption.
  Installed-model scanning and Docker/provenance gates remain external.
- Focused resolver/fusion/adversarial suite: 148 passed.
- The WO-12 JSON and Markdown evidence hashes are
  `296c587596d95343e2a54ce7ef28972f3b48b757a4e68ed8ed0284cd00a97dea`
  and
  `e537ed9eaf0e9b6fbe7a73b1d20911682fb3842dfc2dd55f53e4fc9e05b0d29f`.
- Shell syntax, workflow YAML parsing, Python compilation, and repository diff
  checks pass locally.
- Docker is unavailable on the local host. No WO-20 or WO-21 Docker acceptance
  claim is made until the trusted Linux workflow completes.
