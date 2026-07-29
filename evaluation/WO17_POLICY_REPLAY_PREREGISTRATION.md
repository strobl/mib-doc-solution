# WO17 Final-Review Ordinary-Policy Replay — Preregistration

Status: **preregistered; candidate corpus execution has not started**.

This is an aggregate-only, public-data robustness experiment. The 1,000
labeled cases are already exposed development evidence and are not described
as an unseen holdout. The organizer's private evaluation remains the only
official score.

## Immutable experiment identity

- Experiment ID:
  `wo17-final-review-policy-replay-b02d73c`
- Runtime parent:
  `b02d73c97f3b4e5320e2c38b7a74981de62e35e8`
- Evidence label:
  `public_grouped_robustness_not_unseen`
- Expected records: `1000`
- Baseline score: `130.37185423344323`
- Baseline predictions SHA-256:
  `d6e23641a4e4c7a5517c2b691791146177665f5c297667adae17565f6918a42d`

The experiment may change exactly these candidate paths:

```text
mib_pipeline/rapid_recovery.py
solution.py
tests/test_rapid_recovery.py
```

## Frozen hypothesis

Exact UTF-8 text, hashed without a trailing newline:

```text
Routing only final clean complete primary reviews through one additional truth-blind RapidOCR pass and replaying the unchanged ordinary adjudication engine over conflict-free combined visible evidence will recover correct APPROVED or DENIED decisions while preserving case identity and all nine extraction fields, without changing existing non-review outcomes or treating signed authority, serialization defaults, ambiguous linkage, contested evidence, or policy exceptions as replay inputs.
```

SHA-256:
`747a3f1866be87b73d72c6cfdfaf351f6f8742927cad54aac4a30ec5f214dbf5`

## Frozen primary variable

Exact UTF-8 text, hashed without a trailing newline:

```text
final-review ordinary-policy replay: after all existing heads, route only remaining NEEDS_REVIEW rows with clean visible primary winners for all nine output fields through exactly one policy-only RapidOCR pass; do not overlay fields or re-run existing semantic/recovery heads; combine primary and Rapid candidates through the existing linker/resolver; accept only row/trace-consistent APPROVED or DENIED when case, applicant, and all nine outputs match byte-for-byte and no authority, unresolved linkage, contested evidence, unsafe cue, policy exception, or execution exception is present.
```

SHA-256:
`7795acb3140e4d739eddf42493e928aac98d6efa6d9bb0e23b57a9e5c5b58241`

## Frozen evidence bindings

- Evaluator SHA-256:
  `20515df0d93d6ac73f1f989dd230e4bb1ff6e295e118302114ddd5b4b753c2cf`
- Truth SHA-256:
  `9c6210df4a600c9520435cf7d79d61d7113795dbf94b0e7ab3e39d237388bc8a`
- Input-tree SHA-256:
  `21e821aa3089b841683375da59cf961e679e10f7009e5332ea9e8582f00f4c8e`
- Runtime-contract SHA-256:
  `29de57cbe6500d7754c847b264a3cfad7bec2f4b896e20048388bea6a5e9907f`
- Grouped split-manifest SHA-256:
  `d7aac395c2d42dc42128ba3b4ce15fef6c42c37a6e247a066c267fba8a514b7c`
- Protected-access binding: `null`

## Aggregate diagnostic basis

The current-source WO13 atlas found that classification is the largest score
gap. Its K-safe aggregates include 80 `APPROVED`-truth rows emitted as
`NEEDS_REVIEW` while all nine output fields were already correct in the
primary/deterministic route, plus 19 such rows in the Rapid/recovery route.
The same atlas found no wrong decision in the authority-conflict cohort.

Those aggregates motivate a policy replay; they do not authorize case lookup,
identity-specific handling, label-derived runtime rules, or field mutation.

## Predeclared pass and rejection gates

The candidate is rejected unless all of the following are true:

1. exactly 1,000 valid rows are produced with no missing, extra, duplicate, or
   invalid records;
2. case identity and every one of the nine extracted output fields are
   byte-identical to the frozen baseline for every row;
3. every existing `APPROVED` or `DENIED` decision is unchanged, and the only
   permitted decision transitions are `NEEDS_REVIEW` to `APPROVED` or
   `NEEDS_REVIEW` to `DENIED`;
4. there are zero catastrophic false approvals and zero new false-positive
   denials;
5. extraction score is exactly unchanged and total score is strictly greater
   than `130.37185423344323`;
6. all 15 frozen grouped-fold deltas are non-negative, with every repeat and
   every leave-one-fold-out aggregate strictly positive;
7. focused, full, leakage, and adversarial tests pass;
8. runtime source is clean and the four-hour runtime limit is met; and
9. no candidate is promoted before trusted Docker evidence verifies the image,
   model, memory, temporary-storage, output-size, runtime, and deterministic
   repeat gates.

The first local full-corpus result is therefore diagnostic. A score gain does
not itself authorize adoption or submission.
