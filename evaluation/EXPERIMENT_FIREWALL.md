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
| `CanonicalHashChainStore` | Canonical JSONL, sequence and previous-head hash, locked append, and compare-and-swap |
| `ProgramIntegrityCheckpoint` | Exact head, length, and file hash for all four governed ledgers plus the immutable protected-population tuple, bound to a digest supplied by Git/Factory rather than discovered beside the mutable ledgers; every supported governed mutation consumes the current checkpoint and produces an unpublished successor |
| `CheckpointAuthorityResolver` | The only supported bridge from an authenticated Git/Factory “current checkpoint” lookup into mutation authority; a locally recomputed digest can be inspected but cannot authorize mutation |
| `ExperimentLedger` | Immutable `experiment_plan` preregistration followed by at most one plan-bound `experiment_result`; unique IDs and strict allowlist-based aggregate-only results reject case IDs, PDF filenames, row/sample/outcome collections, predictions, and case-score payloads |
| `FrozenBaselineManifest` | Create-once path, byte-size, and SHA-256 pins; changed artifacts or a changed requested manifest fail verification |
| `TaintRegistry` | Supported exposure-event API is append-only and checkpoint-gated; tainted groups cannot be untainted |
| `RepeatedGroupedSplitManager` | Deterministic repeated K-fold assignment with whole layout/template groups kept together and tainted groups excluded |
| `ProtectedAccessBudget` | Finite aggregate-only access budget; every supported access mutation exact-verifies the current four-ledger checkpoint, and one exclusive lock covers duplicate detection, limit enforcement, and append so concurrent writers cannot overspend |
| `RuntimeLeakageScanner` | Static Python/JSON scan for MIB case IDs, PDF filenames, case/label lookup maps, filename-to-digest tables, and per-file digest keys; allowlists are exact-path and exact-code only |
| `CandidatePromotionGate` | The supported authority for persisting `PASSED`; it selects one unique hash-bound experiment result, revalidates its earlier plan and exact protected aggregate, derives every gate from that record, and enforces strict score improvement plus the next unskipped milestone |
| `CandidateStateStore` | Supported pass/block decisions are hash chained and checkpoint-gated; its public assessment API rejects direct `PASSED`, and a blocked candidate never replaces the latest passing candidate |

The hash chain detects edited, reordered, malformed, non-canonical, or
partially written records. A hash chain alone cannot reveal removal of a valid
suffix. Every supported governed mutation must therefore exact-verify the
single externally published current checkpoint before its compare-and-swap
append. It returns a successor checkpoint, which remains non-authoritative
until Git/Factory publishes its digest. The old checkpoint becomes stale after
the append, so a publication failure quarantines further mutation rather than
permitting silent truncation. Direct access to the underlying storage object is
outside this supported trust boundary and remains subject to code review.

Every supported mutation holds the same deterministic
`evaluation/program/.program-integrity.lock` from checkpoint verification
through all reads, the one append, and successor construction. The lock is
opened without following symlinks and is inode-checked before and after the
critical section. Candidate-state readers must also resolve the externally
published successor before treating a new `PASSED` record as finalized.

Production integration must construct mutation checkpoints through
`CheckpointAuthorityResolver`. Its callback must resolve the single current
checkpoint path and digest from authenticated Git/Factory state; it must never
call `build_program_integrity_checkpoint` over caller-mutable ledgers. Before
publishing a successor, the authority verifies that
`previous_checkpoint_sha256` equals its current digest. Local Git state is a
useful transport and audit trail, but is not by itself an absolute trust root.

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
| `integrity_heads.json` | Historical baseline checkpoint. Its Git/Factory-pinned SHA-256 binds exact heads, lengths, and file hashes for all four ledgers; later checkpoints are versioned instead of overwriting it |

For the initial baseline only, `fold_consistent=true` means a zero-delta
self-comparison: the baseline defines the reference rather than asserting that
new repeated-fold evidence already exists. Every optimization candidate must
replace that self-comparison with measured repeated group-exclusive fold
results before promotion.

The committed ledgers contain no case ID, filename, prediction row, case-score
row, or per-case outcome. The external files named by the manifest are not
committed and never cross into runtime.

## Experiment contract

Before executing a candidate, call `ExperimentLedger.preregister` to append
SHA-256 commitments to one externally retained hypothesis and one primary
variable, the parent commit, a non-empty exact changed-file scope, evidence
class, expected record count, evaluator, input tree, pre-run runtime contract,
frozen split manifest, and truth artifact. Hash commitments keep free-form plan
text and any accidental identity outside the ledger. The preregistration must
consume the currently published four-ledger checkpoint. After execution,
`ExperimentLedger.record_result` appends at most one aggregate-only result bound
to the exact plan record hash and separately binds the produced runtime-evidence
artifact. A result without a plan, a conflicting retry of a plan, a changed
population/runtime-contract binding, or any duplicate result fails closed.

The previously published one-stage `event=experiment` records remain readable
and exactly retryable for backward compatibility, but the legacy API cannot
create a new one-stage record and old records are not treated as
preregistrations. Historical work that was not preregistered must be documented
in a separate `retrospective_not_preregistered` reconciliation artifact; its
verifier recomputes every evidence and ledger digest, ledger head/length, and
latest passing-candidate binding. Such an artifact cannot mutate published
ledgers, consume protected access, or authorize candidate promotion.
`evaluation/program/RETROSPECTIVE_RECONCILIATION_WO15_WO21.json` is the
durable WO15–WO21 use-site. It explicitly records
`preregistered=false`, `promotion_authority=false`,
`protected_access_consumed=false`, and unchanged baseline/candidate state.
It is an auditable historical snapshot, not a backfilled governance record.

Together the immutable plan and aggregate result capture the full experiment
contract. The result is automatically bound to the experiment ID and plan hash,
and records an adopt, reject, or rollback decision plus a non-identifying
rationale token. Its evidence must contain:

- baseline and candidate artifact hashes;
- total, extraction, classification, and calibration scores;
- catastrophic false approvals, missing rows, and invalid rows;
- aggregate per-field and adjudication deltas;
- golden/adversarial regression counts;
- fold-consistency status and, for both public-grouped and protected adoption,
  exactly three repeats of five group-exclusive folds, positive integer fold
  weights covering the expected population, non-negative fold deltas, and
  weighted per-repeat means where every repeat still shows a positive gain
  after any one fold is removed;
- non-vacuous wall time, process CPU time, peak RSS, peak container memory,
  `/tmp`, total model, largest model artifact, image, and output sizes;
- decision-freeze result for a declared confidence-only change;
- runtime leakage-scan result;
- the matching protected-access record hash, when applicable.

An adopted result is rejected above 4 GiB image size, 1 GiB total model size,
250 MiB for any one model artifact, 25 MiB output, 8 GiB peak RSS or container
memory, 2 GiB `/tmp`, 30,000 seconds total wall time, 6 seconds per record, or
120,000 process-CPU seconds. Image bytes, output bytes, wall time, process CPU,
peak RSS, and peak container memory must also be positive so an all-zero
attestation cannot satisfy the gate.

For a protected result, `record_result` also verifies that hash against the
actual configured protected-access ledger, requires the same candidate digest,
requires the protected access to bind the earlier plan and the experiment-ledger
head observed before measurement, requires the access snapshot to precede the
result, and requires every protected-derived aggregate to match the recorded
access exactly at canonical-JSON byte semantics. An `adopt` result fails when
any supplied boolean check is false, any fold is negative, any repeat depends on
one fold, the result population differs from the plan, or the declared resource
limits are exceeded.
Recording an experiment result never authorizes promotion.
`CandidatePromotionGate` remains the only passing authority. Its strict API
does not accept caller-supplied scores, safety counts, fold flags, access
authorization, waivers, or free-form promotion evidence. It requires the full
hash of one `experiment_result`, revalidates that result and its earlier plan,
requires `evidence_label=protected`, rechecks exact equality with the bound
protected-access aggregate, and derives the baseline from the latest passing
candidate digest. The candidate must strictly improve that score. The protected
access purpose must equal the next persisted, unskipped milestone:
`milestone-136`, `milestone-142`, `milestone-146`, or `milestone-148`.
The evaluator, input tree, runtime contract, split manifest, truth, and expected
record count must also equal both the externally checkpointed cutover population
and the latest governed passing population. A candidate digest must differ from
its baseline and may not be reused by a later passing milestone. A non-zero
checkpointed runtime-leakage finding count is an independent hard failure even
when a result claims that its local leakage check passed.

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
2. generate exactly three repeats of five group-exclusive folds;
3. run baseline and candidate on identical fold members;
4. retain individual rows outside the repository and broker;
5. publish only aggregate component, safety, validity, and runtime metrics;
6. reject gains that depend on one fold or one layout family; and
7. label the result as public-data robustness evidence.

The case-level split manifest is required to prove separation but stays in the
external evaluation directory. `devtools/layout_manifest_freezer.py` creates it
from page count and first-page rendered-pixel density only; it reads no labels
or PDF text. Its `mib-wo12-layout-groups/v2` schema attests label-blind
construction only. It deliberately makes no historical
`frozen_before_scoring` claim for the already exposed public baseline. The
committed WO-12 demonstration therefore proves deterministic 3×5 mechanics,
population coverage, and group exclusion, not unseen status or temporal
freeze.

For a future experiment, temporal ordering is established separately: the
exact v2 manifest hash, input-tree hash, and split seed commitment must be
included in the immutable experiment plan and externally checkpointed before
the candidate is executed. The historical
`mib-wo15-layout-groups/v1` schema remains readable only for its already
published retrospective evidence; its caller-authored timing flag is not
promotion authority. Committed phase evidence contains only manifest and
producer hashes, group/fold counts, aggregate score distribution, and explicit
public/tainted evidence labels.

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
the relevant group from that protected role. A protected-access append is not
accepted from a stale checkpoint, so truncating a prior access cannot reset the
budget. Each successful access yields a successor checkpoint that must be
published before the next governed mutation.

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
- any golden/adversarial regression count is nonzero;
- deterministic output differs;
- a gain exists only on tuning data or a single fold; or
- an official Docker/runtime limit fails.

For a new `PASSED` candidate, the candidate-state record also retains the
experiment-plan, experiment-result, and protected-access record hashes, both
ledger heads, the enforced milestone threshold, the complete normalized result,
and every derived gate outcome. Existing historical candidate records remain
readable; they are not rewritten by the strict cutover.

Under the externally pinned checkpoint workflow, failures and consumed accesses
cannot be erased by replacing a ledger with a valid older prefix. The supported
APIs never silently promote a failed result.

## Verification

Run the focused contract suite:

```bash
python3 -m unittest tests.test_experiment_control -v
```

It covers hash tampering, partial and valid-prefix truncation across all four
ledgers, externally pinned checkpoint digests, stale checkpoints,
compare-and-swap writers, duplicate experiments, nested identity leakage,
permanent taint, manifest mutation, exact 3x5 group isolation, access
retry/exhaustion, concurrent budget enforcement, source and artifact leakage,
filename/digest maps, narrow allowlists, supported promotion authority, strict
milestone sequencing, score non-regression, and candidate rollback.

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
