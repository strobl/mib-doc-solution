# WO-11 Governed Full-Public 1,000-Case Baseline

This artifact freezes the aggregate full-public diagnostic baseline used by the
Phase 5 score-improvement program. All 1,000 public labels had already been
evaluated when this report was created. The result is therefore **not** an
unseen holdout, unlabeled-validation, private-test, official-leaderboard, or
final-test score.

The historical frozen 150-case release result remains separately preserved in
`evaluation/RELEASE_EVIDENCE.md`; this report does not replace or reinterpret
that 100.11 result.

## Frozen source

| Item | Value |
| --- | --- |
| Repository commit | `bf6c009cb5a636b00536c62230544a1855c0b757` |
| Commit subject | `Raise offline evaluation score above 130` |
| Commit timestamp | `2026-07-22T07:06:03+02:00` |
| Production entry point | `solution.py` |
| Evaluator version | `mib_weighted_v1` |
| Public truth rows | `1,000` |
| Public PDF count | `1,000` |
| Public PDF tree SHA-256 | `9031e646542f4892a36e07ea42dc34d4a335699a8ced543e1f8038735de40884` |

The production process was launched while `HEAD` was the commit above and the
worktree was clean. Subsequent Phase 5 development changes were made only after
that process had loaded the frozen production graph; none of the pinned runtime
artifacts changed.

## Environment

| Item | Value |
| --- | --- |
| Host | `macOS 15.6.1 (24G90), arm64` |
| Python | `3.12.13` |
| Python dependencies | Direct versions pinned by `requirements.lock` |
| Tesseract | Host `5.5.0`, Leptonica `1.85.0` |
| Worker limit | `4` |
| BLAS/OpenMP thread limits | `4` |
| Docker | Unavailable on this host; no local image-build or container-runtime claim is made |

The host Tesseract version differs from the Dockerfile's pinned image package.
Clean container build and constrained-container runtime remain a separate
WO-20 gate.

## Reproduction

The full production pass was run with the pinned dependency directory:

```bash
env \
  PYTHONPATH=/private/tmp/mib-wo11-py312 \
  MIB_MAX_WORKERS=4 \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  /usr/bin/time -p \
  /Users/cgstrobl/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  solution.py \
  data/train \
  /private/tmp/mib-wo11-full1000-reproduced.jsonl
```

The first pass attempted all 1,000 PDFs and emitted 998 rows. The 998 emitted
rows were byte-for-byte identical to the corresponding rows in the reference
artifact. The two omitted cases both completed successfully when retried
through the same frozen production entry point with one worker. Their
case-level identities and outputs remain in the external run directory and are
not committed here.

The retry set can be derived without embedding case identities in the
repository:

```bash
retry_dir="$(mktemp -d /private/tmp/mib-wo11-retry.XXXXXX)"
while IFS= read -r case_id; do
  ln -s "$PWD/data/train/${case_id}.pdf" "${retry_dir}/"
done < <(
  comm -23 \
    <(tail -n +2 data/train_labels.csv | cut -d, -f1 | LC_ALL=C sort) \
    <(jq -r '.case_id' /private/tmp/mib-wo11-full1000-reproduced.jsonl | LC_ALL=C sort)
)

env \
  PYTHONPATH=/private/tmp/mib-wo11-py312 \
  MIB_MAX_WORKERS=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  /usr/bin/time -p \
  /Users/cgstrobl/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  solution.py \
  "${retry_dir}" \
  /private/tmp/mib-wo11-two-retry.jsonl

LC_ALL=C sort \
  /private/tmp/mib-wo11-full1000-reproduced.jsonl \
  /private/tmp/mib-wo11-two-retry.jsonl \
  -o /private/tmp/mib-wo11-full1000-recovered.jsonl
```

Validation and official evaluator replay:

```bash
python3 scripts/validate_submission.py \
  --submission /private/tmp/mib-wo11-full1000-recovered.jsonl \
  --manifest data/train_labels.csv \
  --require-complete

python3 scripts/evaluate.py \
  --truth data/train_labels.csv \
  --submission /private/tmp/mib-wo11-full1000-recovered.jsonl \
  --output-json /private/tmp/mib-wo11-full1000-evaluation.json \
  --case-scores-jsonl /private/tmp/mib-wo11-full1000-case-scores.jsonl
```

This recovery is a disclosed deterministic retry of transient omissions, not a
prediction patch. No case-level row, answer table, or lookup is committed or
available to the submitted runtime.

## Aggregate result

| Component | Score |
| --- | ---: |
| Extraction | `44.87777777777778 / 50` |
| Classification | `68.52000000000001 / 80` |
| Calibration | `16.974076455665433 / 20` |
| Missing-case penalty | `0.0 / 10` |
| **Total** | **`130.37185423344323 / 150`** |

| Quality measure | Result |
| --- | ---: |
| Submitted/scored records | `1,000 / 1,000` |
| Missing cases | `0` |
| Extra cases | `0` |
| Duplicate case IDs | `0` |
| Invalid adjudication records | `0` |
| Invalid confidence records | `0` |
| Invalid fee-status records | `0` |
| Catastrophic false approvals | `0` |
| Mean confidence Brier error | `0.07564808860836421` |

### Confusion matrix

| Truth → output | Cases |
| --- | ---: |
| `APPROVED → APPROVED` | `164` |
| `APPROVED → DENIED` | `10` |
| `APPROVED → NEEDS_REVIEW` | `115` |
| `DENIED → DENIED` | `382` |
| `DENIED → NEEDS_REVIEW` | `49` |
| `NEEDS_REVIEW → APPROVED` | `6` |
| `NEEDS_REVIEW → DENIED` | `6` |
| `NEEDS_REVIEW → NEEDS_REVIEW` | `268` |

## Runtime and memory

| Measurement | Result |
| --- | ---: |
| First-pass wall time | `2,560.60 s` |
| First-pass user CPU | `8,639.56 s` |
| First-pass system CPU | `887.42 s` |
| Two-case retry wall time | `7.10 s` |
| End-to-end production time including retry | `2,567.70 s` |
| Highest periodically sampled process RSS | `1,645,056 KiB` (`1,606.5 MiB`) |

The RSS value is the largest periodic observation collected during the run,
not an exact operating-system peak. The exact peak is therefore recorded as
unavailable rather than inferred. WO-20 owns authoritative container runtime
and memory measurements under the official resource profile.

## Artifact integrity

| Artifact | SHA-256 |
| --- | --- |
| Complete predictions JSONL | `d6e23641a4e4c7a5517c2b691791146177665f5c297667adae17565f6918a42d` |
| Official evaluation JSON | `e0c285f3f8f5c5b148fd00d825746aa57fd46f51452189d7f16d1aef9e987001` |
| Official case-score JSONL | `a67152a345221106a87e6c0f6e1bc5c23e7f272e500bf58071ef724f8a91ce5f` |
| First-pass 998-row JSONL | `f450e6a266503facea4d05b5e1163f9d183eed21ac5528f17cfddbb902c6a575` |
| Two-row retry JSONL | `00cbeeca596ddf6c91f16a9969bdd99d0de2f42e470ff97c87c3e7ce5434278b` |
| Public labels CSV | `9c6210df4a600c9520435cf7d79d61d7113795dbf94b0e7ab3e39d237388bc8a` |
| Official evaluator source | `20515df0d93d6ac73f1f989dd230e4bb1ff6e295e118302114ddd5b4b753c2cf` |
| Submission validator source | `206d89d37f4ea38ed72af53fa6afd3b7d529b337b90bf606a884a65e5db50160` |
| `requirements.lock` | `fbacf7de9d18359f2b3ff24509e9214be4ff2aa59e4bf9b619a06e1c04892a67` |
| `Dockerfile` | `950813595c877a5dda6904d4073db8e4dfa797c6f05c7c5749b5c94a2d5cf1fd` |
| `run.sh` | `2ab1b5f52ca40cbb9f8bf9517b366ce2ab17543681d10374ee25066416eb4382` |
| Inner calibration artifact | `848306a09afdbe0075ab4ef8059e33cea6dd63793dae46b8d4301cc267cec738` |
| Policy-exception artifact | `4fbf2c14a7e1016dc1b3ff10ada8b4a55b58ecd52a542ebfd0d8a1ab46c866eb` |
| Final-output recalibration artifact | `cccfd667988f653c6569ce447a8c5751ba7a6bb90d3b79e8b5e8b893c3524762` |

The complete predictions, evaluation JSON, and case-score JSONL are
byte-identical to the recorded reference hashes. This reproduces the governed
130.371854 reference result despite the disclosed transient first-pass
omissions.

## Verification status

- Clean baseline suite: `Ran 232 tests`; `OK (skipped=2)`, independently
  reproduced with Python 3.12.13.
- Complete-output validator: passed with 1,000 valid records and 0 missing
  expected IDs.
- Official evaluator replay: passed and reproduced all three reference artifact
  hashes exactly.
- Historical `evaluation/RELEASE_EVIDENCE.md`: unchanged from the frozen source
  commit.
- Local Docker build/run: not performed because Docker is unavailable on this
  host; no Docker claim is made.

## Governed interpretation

`130.371854` is the Phase 5 full-public diagnostic baseline. It is appropriate
for aggregate score-loss analysis and for comparing leakage-resistant,
group-exclusive public-data robustness experiments. It must never be described
as an untouched holdout or as evidence of the organizer's private score.
