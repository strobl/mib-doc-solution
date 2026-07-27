# WO-21 adversarial audit harness

`devtools/wo21_adversarial_audit.py` generates a deterministic synthetic
document corpus in an external working directory and exercises every attack
and metamorphic category named by WO-21.

The harness runs the production processor twice. Decoy attacks must preserve
all non-confidence output, moderate readable transforms must be
non-confidence invariant, and destructive transforms must never produce a new
approval, adopt decoy content, omit a record, or emit an invalid record.

Corpus documents, prediction rows, and failure details are never written to
the repository. The committable evidence schema accepts only aggregate
counts, booleans, timings, and whole-artifact hashes. It rejects case
identifiers, document filenames, paths, per-case sequences, prediction rows,
and any nonzero regression-waiver count.

After an observed regression has been filed against its owning work order,
pass its aggregate count with `--regression-filed-count`. The harness rejects
a filed count greater than the number of observed regressions.

A host-oracle or identity-scan failure takes precedence as
`blocked_regression`; unavailable or incomplete Docker verification remains
visible as a separate blocked hard gate. When the host audit is clean but
Docker is unavailable or its clean constrained evidence has not been
provided, the top-level outcome is `blocked_environment`. Host-side results
cannot by themselves satisfy the Docker reproducibility or runtime gates.
Every required attack and metamorphic category must have at least one
scenario; complete category coverage is a hard host gate, even if every
scenario that did run passed its oracle.

The leakage scan uses the same fixed installed scope as WO20: `/app`, `/opt`,
the Python 3.12 site-packages tree, and the Tesseract data tree. It scans all
WO20 model extensions (including `.onnx` and `.traineddata`) plus runtime
model/config JSON under `/app/mib_pipeline/artifacts`. Missing model/config
inventory, unreadable artifacts, case identifiers, or PDF filenames fail
closed.

The committed WO20/WO21 workflow closes the environment gate only after both
independent 5,000-case Docker captures and their fail-closed comparator pass.
It then runs this audit inside a fresh image with no network, four CPUs, 8 GiB
RAM, a read-only root/repository, bounded PIDs, and tmpfs. The workflow verifies
the clean checkout before supplying its exact revision and Docker attestations
to the minimal image, which deliberately does not contain Git. Those external
attestations remain explicit booleans in the aggregate evidence.

A passing WO21 artifact is additionally bound to the exact aggregate WO20
artifact downloaded from the same GitHub Actions run. The committed evidence
records the repository, workflow run ID and attempt, and the actual SHA-256 of
that downloaded aggregate; its Markdown renders a direct workflow-run link.
The harness rejects Docker reproducibility/runtime attestations without this
complete provenance. The committed workflow is the trusted producer: it
downloads and validates the prerequisite before computing the recorded hash.
Reviewers can follow the run link and verify that hash against the WO20
artifact. Invoking the host audit alone or supplying only a source revision
cannot produce a passing result.
