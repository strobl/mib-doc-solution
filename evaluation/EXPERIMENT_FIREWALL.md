# WO-12 Experiment Firewall

This is development-only control infrastructure. None of the ledgers, split
manifests, labels, evaluator reports, or case-level diagnostics described here
are copied into the submitted runtime image.

## Evidence classification

All 1,000 public labeled cases have already been evaluated and some have been
inspected during diagnosis. No subset selected from those cases is described
as pristine, unseen, private, or an honest untouched holdout.

Repeated grouped folds provide only:

```text
public_grouped_robustness_not_unseen
```

The organizer's private evaluation remains the only proof of the official
score. The 5,000 public validation PDFs are unlabeled and may be used only for
format, runtime, determinism, and missing-row checks. Leaderboard movement,
manual validation inspection, and social feedback are not tuning signals.

## Controls

`devtools/experiment_control.py` implements the following fail-closed
contracts:

| Control | Enforced property |
| --- | --- |
| `CanonicalHashChainStore` | Canonical JSONL, sequence and previous-head hash, locked append, compare-and-swap, and retained-head truncation detection |
| `ExperimentLedger` | Unique experiment IDs and strict allowlist-based aggregate-only evidence; case IDs, PDF filenames, row/sample/outcome collections, predictions, and case-score payloads are rejected |
| `FrozenBaselineManifest` | Create-once path, byte-size, and SHA-256 pins; changed artifacts or a changed requested manifest fail verification |
| `TaintRegistry` | Append-only exposure events; tainted groups cannot be untainted |
| `RepeatedGroupedSplitManager` | Deterministic repeated K-fold assignment with whole layout/template groups kept together and tainted groups excluded |
| `ProtectedAccessBudget` | An immutable finite aggregate-only access budget; one exclusive lock covers duplicate detection, limit enforcement, and append, so concurrent writers cannot overspend |
| `RuntimeLeakageScanner` | Static Python/JSON scan for MIB case IDs, PDF filenames, case/label lookup maps, filename-to-digest tables, and per-file digest keys; allowlists are exact-path and exact-code only |
| `CandidatePromotionGate` | The only authority that may persist `PASSED`; it evaluates every hard gate and derives protected authorization from the access ledger |
| `CandidateStateStore` | Every pass/block decision is hash chained; direct pass injection fails and a blocked candidate never replaces the latest passing candidate |

The hash chain detects edited, reordered, malformed, non-canonical, or
partially written records. A valid prefix cannot reveal that later records
were removed by itself, so each phase-exit evidence package must retain and
publish the expected ledger head and record count.

## Frozen baseline demonstration

`evaluation/program/` contains the committed aggregate-only demonstration on
the WO-11 baseline:

| Artifact | Purpose |
| --- | --- |
| `frozen_baseline_manifest.json` | Pins the external predictions/evaluator artifacts plus public truth, evaluator source, and all three runtime artifacts by path, byte size, and SHA-256 |
| `experiment_ledger.jsonl` | Records the governed baseline contract and aggregate score/runtime facts |
| `taint_registry.jsonl` | Permanently marks the evaluated public labeled cohort as exposed |
| `protected_access_ledger.jsonl` | Freezes a five-access budget and consumes one aggregate-only access to establish the baseline |
| `candidate_state_ledger.jsonl` | Establishes the byte-reproduced baseline as the initial passing candidate through `CandidatePromotionGate` |
| `integrity_heads.json` | Publishes each ledger's expected head, length, and file hash so valid-prefix truncation is detectable |

For the initial baseline only, `fold_consistent=true` means a zero-delta
self-comparison: the baseline defines the reference rather than asserting that
new repeated-fold evidence already exists. Every optimization candidate must
replace that self-comparison with measured repeated group-exclusive fold
results before promotion.

The committed ledgers contain no case ID, filename, prediction row, case-score
row, or per-case outcome. The external files named by the manifest are not
committed and never cross into runtime.

## Experiment contract

Before executing a candidate, record one hypothesis and one primary variable
unless the experiment is explicitly factorial. The aggregate ledger evidence
must contain:

- experiment ID and hypothesis;
- parent commit SHA and exact changed files;
- evidence class and split-manifest hash;
- baseline and candidate artifact hashes;
- total, extraction, classification, and calibration scores;
- catastrophic false approvals, missing rows, and invalid rows;
- aggregate per-field and adjudication deltas;
- golden/adversarial regression counts;
- wall time, process CPU time, peak RSS, `/tmp`, model, image, and output sizes;
- decision-freeze result for a declared confidence-only change;
- runtime leakage-scan result;
- protected-access ID, when applicable; and
- adopt, reject, or rollback decision with rationale.

Individual protected-case outcomes must not enter the experiment ledger.
Diagnosis that reveals a protected case or group appends a taint event before
that evidence can be used for tuning.

## Grouped robustness protocol

Layout/template group IDs must be generated without labels or case identity,
from visible layout properties such as page count, normalized page geometry,
and coarse structure. Groups may not be based on filenames, PDF hashes,
answers, adjudication, error status, or case IDs.

For each frozen candidate:

1. derive the group manifest before reading candidate scores;
2. generate at least three repeats of five group-exclusive folds;
3. run baseline and candidate on identical fold members;
4. retain individual rows outside the repository and broker;
5. publish only aggregate component, safety, validity, and runtime metrics;
6. reject gains that depend on one fold or one layout family; and
7. label the result as public-data robustness evidence.

The case-level split manifest is required to prove separation but stays in the
external evaluation directory. The committed phase evidence contains only its
hash, group/fold counts, aggregate score distribution, and taint snapshot hash.

## Protected access budget

The Phase 5–8 program has five planned protected aggregate accesses:

1. frozen baseline;
2. first candidate at or above 136;
3. first candidate at or above 142;
4. first candidate at or above 146; and
5. the frozen 148 release candidate.

A retry with identical bytes and the same access ID is free. A changed
candidate or changed aggregate result requires a new access. Debugging from
individual protected errors is prohibited; diagnosis first taints and removes
the relevant group from that protected role.

## Runtime leakage boundary

The scan targets are:

```text
solution.py
mib_pipeline/**/*.py
mib_pipeline/artifacts/*.json
```

The scanner is deliberately separate from the older artifact-only check. It
examines both executable source and pinned JSON artifacts. Current production
source and all three pinned runtime artifacts scan with zero findings and no
allowlist. Development tools, evaluator files, public labels, reports, and
examples are not runtime inputs and are not copied by the root Dockerfile.

## Promotion and rollback

The candidate is blocked and the latest passing candidate remains active if
any hard gate fails:

- catastrophic false approvals increase above zero;
- missing or invalid rows increase above zero;
- a runtime identity/leakage finding exists;
- a confidence-only change alters any non-confidence byte;
- a protected case is exposed without being tainted;
- a newly regressed golden/adversarial case lacks an explicit human waiver;
- deterministic output differs;
- a gain exists only on tuning data or a single fold; or
- an official Docker/runtime limit fails.

Failures remain in the append-only ledger. They are never erased or silently
promoted.

## Verification

Run the focused contract suite:

```bash
python3 -m unittest tests.test_experiment_control -v
```

It covers hash tampering, partial and retained-prefix truncation, stale
compare-and-swap writers, duplicate experiments, nested identity leakage,
permanent taint, manifest mutation, deterministic group isolation, access
retry/exhaustion, concurrent budget enforcement, source and artifact leakage,
filename/digest maps, narrow allowlists, non-bypassable promotion, and
candidate rollback.

Run the current production leakage scan:

```bash
python3 - <<'PY'
from pathlib import Path
from devtools.experiment_control import RuntimeLeakageScanner

root = Path(".")
RuntimeLeakageScanner().require_clean(
    [root / "solution.py", root / "mib_pipeline"]
)
print("runtime leakage findings: 0")
PY
```
