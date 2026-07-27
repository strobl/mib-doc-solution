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
- At the last external check during this reconciliation, official challenge
  PR #6 was open and unmerged. It remains organizer-controlled. No merge,
  resubmission, or official prediction regeneration is authorized by this
  reconciliation.

## 22/22 matrix

| WO | Factory state | Reconciled outcome | Repository evidence / remaining gate |
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
| 11 | in_review | Evidence ready within its stated scope | `FULL_PUBLIC_1000_130_37.md` records a locally reproduced, source-pinned aggregate baseline, schema/coverage validity, and zero catastrophic false approvals on the exposed public labels. It discloses the 998+2 retry, externally retained raw artifacts, and missing clean-Docker certification. |
| 12 | in_review | Implementation ready for review | The experiment firewall now supports immutable two-stage plan/result events, strict adoption gates, protected-access binding, and durable retrospective reconciliation. Published ledger heads remain unchanged. |
| 13 | in_review | Evidence ready | `SCORE_LOSS_ATLAS_130_37.md` records the aggregate loss atlas and oracle ceilings without runtime lookup use. |
| 14 | in_review | Evidence ready | The bounded OCR/layout/orientation ablation is scope-complete in `OCR_ABLATION_REPORT_BOUNDED_V4`; it recommends review order but does not promote a candidate. |
| 15 | blocked | Rejected by governing adoption gate | Historical WO-15 evidence passed its original gate with `+0.173611` on 32 public cases, but the governing program gate rejects it because only 3/15 folds are positive and the gain disappears after removing the strongest fold. This is `rejected_single_fold_concentration`, not a promotion. |
| 16 | in_progress | Regression fixed locally; independent confirmation pending | A narrowly scoped resolver change excludes cleanly separable foreign-case pages while mixed, unhinted, or conflicting scopes still fail closed. Targeted resolver/fusion/adversarial tests pass. A clean Docker WO-21 rerun and a grouped-score revalidation bound to the fixed source are still required. |
| 17 | in_review | Historical evidence recorded; governed revalidation pending | The retrospective result was a `+0.111145` calibration-only 32-case diagnostic, while targeted confusion deltas remained zero. It is not a governed full-public milestone or promotion and still requires revalidation under the current evidence contract. |
| 18 | blocked | No promotion | The exact four-arm comparison is `blocked_precondition`: two protected roles lack coverage and the standard promotion gate was not evaluated. No model entered runtime composition. |
| 19 | in_review | Evaluated, no promotion | The selected beta refit worsened OOF Brier by `+0.002765249`; S0 remains pinned. The in-sample/shadow `137.072877` total is not promoted. |
| 20 | in_progress | Local harness complete; real Linux Docker evidence pending | The workflow binds source, input, image and model inventory, runs two independent constrained 5,000-case captures, and compares aggregate evidence fail-closed. A trusted smoke run exposed missing `docker stats` samples; the measurement now conservatively takes the maximum of in-container cgroup readings and Docker stats while still blocking if both are unavailable. Acceptance requires the new exact-source workflow run to pass. |
| 21 | in_progress | Production-path host audit passes; trusted report pending | The real production processor now passes all 15 scenarios twice locally with byte-identical captures, zero regressions, zero new approvals, and zero decoy adoption. The harness hard-gates all 13 categories and binds installed-model scanning plus the final report to the same-run WO-20 aggregate. Local model-root scanning and Docker/provenance attestations remain unavailable, so acceptance still requires the trusted workflow run. |
| 22 | backlog | Not startable | Its own precondition is that all prior gates pass. WO-10 has not reached 148 and WO-15/18 are blocked, so no final 1,000-case rerun, candidate freeze, or resubmission package is authorized. |

## Current state totals

- `completed`: 9
- `in_review`: 6
- `in_progress`: 4
- `blocked`: 2
- `backlog`: 1
- total: 22

## Verification for this reconciliation change set

- Full local unit suite: 665 passed, 2 skipped.
- Real local WO-21 production-path audit: 15/15 scenarios passed twice,
  byte-identical, with zero regressions, new approvals, or decoy adoption.
  Installed-model scanning and Docker/provenance gates remain external.
- Focused resolver/fusion/adversarial suite: 148 passed.
- Focused WO-12/WO-20/WO-21 governance and runtime suite: 79 passed.
- Shell syntax, workflow YAML parsing, Python compilation, and repository diff
  checks pass locally.
- Docker is unavailable on the local host. No WO-20 or WO-21 Docker acceptance
  claim is made until the trusted Linux workflow completes.
