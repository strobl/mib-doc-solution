# Work Order Reconciliation: WO-1 through WO-22

This is the repository-side reconciliation of all 22 Software Factory Work
Orders. It records the evidence state rather than treating `in_review` as
complete or treating a diagnostic score as a promoted candidate.

Snapshot date: 2026-07-27.

## Governing facts

- The latest governed passing full-public 1,000-case candidate remains
  `130.37185423344323 / 150`.
- The later `132`–`137` totals are public-label-exposed 32-case diagnostics.
  They are not WO-10 milestones, protected promotions, official validation
  scores, private-test scores, or leaderboard results.
- The candidate-state ledger contains one passing entry: the frozen baseline.
- The historical WO-15 through WO-21 evidence was not preregistered. It is
  bound in `RETROSPECTIVE_RECONCILIATION_WO15_WO21.json` as retrospective,
  non-promotional evidence. Published ledgers were not backfilled.
- The official challenge PR #6 remains organizer-controlled. No merge,
  resubmission, or official prediction regeneration is authorized by this
  reconciliation.

## 22/22 matrix

| WO | Factory state | Reconciled outcome | Repository evidence / remaining gate |
| --- | --- | --- | --- |
| 1 | completed | Delivered | Docker/offline runtime scaffold was implemented and merged; the current hardening is covered again by WO-20. |
| 2 | completed | Delivered | Batch runner and canonical twelve-field JSONL writer were implemented and merged. |
| 3 | completed | Delivered | Render-first ingestion was implemented and merged. |
| 4 | completed | Delivered | Visible OCR/CV extraction and untrusted-content filtering were implemented and merged. |
| 5 | completed | Delivered | Evidence model, applicant scoping, and precedence resolver were implemented and merged. |
| 6 | completed | Delivered | Deterministic adjudication and safety policy were implemented and merged. |
| 7 | completed | Delivered | Pinned confidence calibration was implemented and merged. |
| 8 | completed | Delivered | Evaluation, regression, leakage, and false-approval gates were implemented and merged. |
| 9 | completed | Delivered, external review pending | The three-file challenge entry and 5,000-row validation were completed; the user attested that the form was submitted. Official PR #6 is still open/unmerged. |
| 10 | in_progress | Target not achieved | The governed score is still `130.371854`; M1 `>=136`, M2 `>=142`, M3 `>=146`, and final `>=148` are not claimed. |
| 11 | in_review | Evidence ready | `FULL_PUBLIC_1000_130_37.md` freezes the reproducible governed baseline and its hashes, validity, safety, and disclosed retry limitation. |
| 12 | in_progress | Implementation ready for review | The experiment firewall now supports immutable two-stage plan/result events, strict adoption gates, protected-access binding, and durable retrospective reconciliation. Published ledger heads remain unchanged. |
| 13 | in_review | Evidence ready | `SCORE_LOSS_ATLAS_130_37.md` records the aggregate loss atlas and oracle ceilings without runtime lookup use. |
| 14 | in_review | Evidence ready | The bounded OCR/layout/orientation ablation is scope-complete in `OCR_ABLATION_REPORT_BOUNDED_V4`; it recommends review order but does not promote a candidate. |
| 15 | blocked | Rejected by parent hard gate | The standalone harness reports `+0.173611` on 32 public cases, but only 3/15 folds are positive and the gain disappears when the best fold is removed. This is `rejected_single_fold_concentration`, not a promotion. |
| 16 | in_progress | Regression fixed locally; independent confirmation pending | A narrowly scoped resolver change excludes cleanly separable foreign-case pages while mixed, unhinted, or conflicting scopes still fail closed. Targeted resolver/fusion/adversarial tests pass; the clean Docker WO-21 rerun remains required. |
| 17 | in_review | Evidence ready; no program promotion | Policy revalidation produced a `+0.111145` calibration-only 32-case diagnostic with zero safety/validity regressions. It is not a governed full-public milestone. |
| 18 | blocked | No promotion | The exact four-arm comparison is `blocked_precondition`: two protected roles lack coverage and the standard promotion gate was not evaluated. No model entered runtime composition. |
| 19 | in_review | Evaluated, no promotion | The selected beta refit worsened OOF Brier by `+0.002765249`; S0 remains pinned. The in-sample/shadow `137.072877` total is not promoted. |
| 20 | in_progress | Local harness complete; real Linux Docker evidence pending | The workflow now binds source, input, image and model inventory, runs two independent constrained 5,000-case captures, compares aggregate evidence fail-closed, and records sampled resource maxima. Acceptance requires the trusted workflow run to pass. |
| 21 | in_progress | Local adversarial audit complete; trusted Docker attestation pending | The audit covers 13 categories / 15 scenarios twice, makes category completeness a hard gate, scans installed ONNX, Tesseract, and runtime JSON artifacts, and binds its report to the same-run WO-20 aggregate. Acceptance requires the trusted workflow run to pass. |
| 22 | backlog | Not startable | Its own precondition is that all prior gates pass. WO-10 has not reached 148 and WO-15/18 are blocked, so no final 1,000-case rerun, candidate freeze, or resubmission package is authorized. |

## Current state totals

- `completed`: 9
- `in_review`: 5
- `in_progress`: 5
- `blocked`: 2
- `backlog`: 1
- total: 22

## Verification for this reconciliation change set

- Full local unit suite: 658 passed, 2 skipped.
- Focused resolver/fusion/adversarial suite: 148 passed.
- Focused WO-12/WO-20/WO-21 governance and runtime suite: 79 passed.
- Shell syntax, workflow YAML parsing, Python compilation, and repository diff
  checks pass locally.
- Docker is unavailable on the local host. No WO-20 or WO-21 Docker acceptance
  claim is made until the trusted Linux workflow completes.
