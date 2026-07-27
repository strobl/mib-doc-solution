# WO13 Exact-Baseline Trace Capture — Execution Template

Status: **tooling only; no full capture has been executed**.

The raw trace and predictions are external development evidence. Only the
K-safe aggregate atlas may enter Git. This procedure cannot run
authoritatively from a dirty checkout or before these WO13 tools and the atlas
are committed.

## Frozen baseline and preregistration

- Baseline production revision:
  `bf6c009cb5a636b00536c62230544a1855c0b757`
- Expected prediction bytes:
  `d6e23641a4e4c7a5517c2b691791146177665f5c297667adae17565f6918a42d`
- Input-tree SHA-256:
  `21e821aa3089b841683375da59cf961e679e10f7009e5332ea9e8582f00f4c8e`
- Canonical layout-manifest SHA-256:
  `d7aac395c2d42dc42128ba3b4ce15fef6c42c37a6e247a066c267fba8a514b7c`
- Frozen evaluator manifest:
  `evaluation/program/frozen_baseline_manifest.json`
- Frozen baseline predictions:
  `/private/tmp/mib-wo11-full1000-recovered-v2.jsonl`

The capture source revision is deliberately different from the baseline
production revision: it is the future clean commit containing these reviewed
WO13 tools. The authority check requires every production path to remain
byte-identical to the baseline revision while requiring the capture, contract,
atlas, and WO12 freezer tools to be committed and clean at that exact HEAD.

The older `/private/tmp/mib-wo17-private/runtime_contract.json` is not valid for
this run: it declares a 30,000-second 5,000-scale limit. Create and review a
new exact 1,000-case contract with the existing v1 schema, all Dockerfile
environment values, exactly four workers, and
`container_limits.runtime_seconds = 14400`.

The reviewed contract content is:

```json
{
  "capture": {
    "arm_repeat_count": 2,
    "execution": "sequential",
    "max_workers": 4,
    "metrics_source": "fresh_process_rusage_self_plus_waited_children_and_monotonic_wall",
    "required_byte_determinism": true
  },
  "container_limits": {
    "image_bytes": 4294967296,
    "max_model_artifact_bytes": 262144000,
    "model_bytes": 1073741824,
    "network": "none",
    "output_bytes": 26214400,
    "peak_memory_bytes": 8589934592,
    "per_record_runtime_seconds": 6,
    "runtime_seconds": 14400,
    "tmp_bytes": 2147483648
  },
  "environment": {
    "BLIS_NUM_THREADS": "4",
    "HOME": "/tmp",
    "MALLOC_ARENA_MAX": "4",
    "MIB_MAX_WORKERS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OC_DISABLE_DOT_ACCESS_WARNING": "1",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "TMPDIR": "/tmp",
    "TOKENIZERS_PARALLELISM": "false",
    "VECLIB_MAXIMUM_THREADS": "4"
  },
  "evaluation": {
    "evaluator_sha256": "20515df0d93d6ac73f1f989dd230e4bb1ff6e295e118302114ddd5b4b753c2cf",
    "evidence_label": "public_grouped_robustness_not_unseen",
    "expected_record_count": 1000,
    "input_tree_sha256": "21e821aa3089b841683375da59cf961e679e10f7009e5332ea9e8582f00f4c8e",
    "layout_manifest_sha256": "d7aac395c2d42dc42128ba3b4ce15fef6c42c37a6e247a066c267fba8a514b7c",
    "truth_sha256": "9c6210df4a600c9520435cf7d79d61d7113795dbf94b0e7ab3e39d237388bc8a"
  },
  "interface": {
    "entrypoint": "solution.py",
    "input": "directory containing canonical PDF cases",
    "output": "canonical twelve-field JSONL",
    "runner": "run.sh"
  },
  "schema_version": "mib-wo17-runtime-contract/v1"
}
```

An aggregate CI zip is not a dataset archive. The approved source archive is
`/private/tmp/mib-wo13-approved-input-1000.zip`. The tool does not trust that
filename: it streams the ZIP and requires exactly 1,000 root
`MIB-NNNNNN.pdf` members with no directories, links, encryption, duplicates,
traversal, or extras. It recomputes the canonical name/content input-tree
SHA-256 directly from member bytes and requires `21e821…`. Stored and deflated
members are supported within fixed archive and uncompressed-size bounds.

## Exact execution environment

Both preregistration and capture must use CPython with `-I -B`, the exact
installed dependency closure, and these Dockerfile values:

```bash
env \
  BLIS_NUM_THREADS=4 \
  HOME=/tmp \
  MALLOC_ARENA_MAX=4 \
  MIB_MAX_WORKERS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  OC_DISABLE_DOT_ACCESS_WARNING=1 \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  TMPDIR=/tmp \
  TOKENIZERS_PARALLELISM=false \
  VECLIB_MAXIMUM_THREADS=4 \
  /ABSOLUTE/PINNED/PYTHON3 -I -B \
  devtools/wo13_trace_capture.py prepare-authority \
  --authority-output /private/tmp/wo13-authority.json \
  --input-dir /ABSOLUTE/EXACT_1000_PDF_DIRECTORY \
  --layout-manifest /private/tmp/mib-wo12-full1000.yICKRb/layout_manifest.json \
  --dataset-archive /private/tmp/mib-wo13-approved-input-1000.zip \
  --runtime-contract /private/tmp/mib-wo13-runtime-contract.json \
  --frozen-baseline-manifest \
    /ABSOLUTE/REPOSITORY/evaluation/program/frozen_baseline_manifest.json \
  --baseline-predictions /private/tmp/mib-wo11-full1000-recovered-v2.jsonl \
  --retry-missing-attempts 1
```

The authority output is canonical, read-only, and create-once. Review and
record its SHA-256 before running the capture. Do not regenerate it to make a
failed check pass.

## External authoritative capture

Run with the same interpreter, flags, environment, checkout, and dependency
installation:

```bash
env \
  BLIS_NUM_THREADS=4 \
  HOME=/tmp \
  MALLOC_ARENA_MAX=4 \
  MIB_MAX_WORKERS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  OC_DISABLE_DOT_ACCESS_WARNING=1 \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  TMPDIR=/tmp \
  TOKENIZERS_PARALLELISM=false \
  VECLIB_MAXIMUM_THREADS=4 \
  /ABSOLUTE/PINNED/PYTHON3 -I -B \
  devtools/wo13_trace_capture.py capture \
  --authority-manifest /private/tmp/wo13-authority.json \
  --predictions-output /private/tmp/wo13-predictions.jsonl \
  --trace-output /private/tmp/wo13-trace.json
```

The runner copies PDFs, every consumed production/tool source, and the three
policy artifacts into a private external snapshot before processing. It
recomputes the canonical WO12 layout groups from those PDF bytes. It then
rechecks both the private snapshot and every origin authority before
finalization. Caller-supplied processors, bindings, probes, workers, retry
policies, paths, or hashes cannot create `authoritative_production`; injected
captures are always `capture_mode = test`.

The trace finalizes only when all 1,000 rows are present, canonical prediction
bytes equal the frozen `d6e236…` artifact, every authority is unchanged, the
runtime identity matches exactly, and all raw outputs remain outside Git.

## Independent aggregate-only atlas

The atlas independently reloads the authority, rechecks clean exact-source
Git state and every authority byte, and recomputes the layout groups from the
bound PDFs:

```bash
/ABSOLUTE/PINNED/PYTHON3 -I -B scripts/score_loss_atlas.py \
  --truth /ABSOLUTE/EXACT_TRAIN_LABELS_CSV \
  --submission /private/tmp/wo13-predictions.jsonl \
  --evaluation /private/tmp/mib-wo11-full1000-recovered-v2-evaluation.json \
  --case-scores /private/tmp/mib-wo11-full1000-recovered-v2-case-scores.jsonl \
  --layout-manifest /private/tmp/mib-wo12-full1000.yICKRb/layout_manifest.json \
  --trace-dimensions /private/tmp/wo13-trace.json \
  --frozen-baseline-manifest \
    /ABSOLUTE/REPOSITORY/evaluation/program/frozen_baseline_manifest.json \
  --runtime-contract /private/tmp/mib-wo13-runtime-contract.json \
  --dataset-archive /private/tmp/mib-wo13-approved-input-1000.zip \
  --baseline-predictions /private/tmp/mib-wo11-full1000-recovered-v2.jsonl \
  --authority-manifest /private/tmp/wo13-authority.json \
  --output-json /private/tmp/WO13_SCORE_LOSS_ATLAS_CURRENT_SOURCE.json \
  --output-markdown /private/tmp/WO13_SCORE_LOSS_ATLAS_CURRENT_SOURCE.md
```

Use the same exact environment prefix for the atlas command. Copy only the
reviewed aggregate JSON and Markdown into `evaluation/` after all dimensions
report `current_source`, `dimension_measurement_blockers` is empty, and a
case-ID/raw-row scan is clean.
