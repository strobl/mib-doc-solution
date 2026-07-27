#!/usr/bin/env python3
"""Run production WO-17 auditing and emit a path-free aggregate artifact.

The exact production composition root is observed once per accepted final
result.  The generated prediction bytes must match an already recorded
candidate run, so transient resolver/recovery attempts cannot enter the
artifact.  Rare execution-order branches are measured independently by the
executable contract probes; no probe count is fabricated from cohort absence.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.experiment_control import (  # noqa: E402
    ExperimentControlError,
    canonical_json,
)
from devtools.fusion_audit_run import (  # noqa: E402
    case_id_set_sha256,
    cohort_tree_sha256,
)
from devtools.policy_revalidation_audit_contract import (  # noqa: E402
    CONTRACT_AUDIT_COUNTS,
    COHORT_AUDIT_COUNTS,
    POLICY_REVALIDATION_AUDIT_SCHEMA,
)
from devtools.policy_revalidation_contract_probe import (  # noqa: E402
    PolicyContractProbeError,
    contract_fixture_sha256,
    run_contract_probes,
)
from mib_pipeline import BatchRunner, build_production_processor  # noqa: E402
from mib_pipeline.batch import discover_case_pdfs  # noqa: E402
from mib_pipeline.decision_recovery import (  # noqa: E402
    POLICY_AUDIT_COUNT_NAMES,
)


_SOURCE_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REQUIRED_CORE_COUNTS = frozenset(
    {
        "late_recovery_before_revalidation_count",
        "contradicted_synthetic_reason_removed_count",
        "independent_denial_reason_retained_count",
        "review_confidence_restored_count",
        "normal_policy_rerun_count",
        "signed_late_authority_recovery_count",
        "late_adjudication_evidence_preserved_count",
        "late_biohazard_evidence_preserved_count",
        "forced_approval_count",
        "serialization_default_used_as_policy_evidence_count",
        "sentinel_value_used_as_policy_evidence_count",
        "placeholder_value_used_as_evidence_count",
        "stale_threshold_mismatch_count",
        "contradicted_synthetic_reason_left_active_count",
    }
)


class PolicyRevalidationAuditRunError(ValueError):
    """The accepted-final policy audit could not be proven complete."""


def _validate_source_revision(value: str) -> str:
    normalized = str(value).strip().casefold()
    if not _SOURCE_REVISION_RE.fullmatch(normalized):
        raise PolicyRevalidationAuditRunError(
            "source revision must be a full Git commit SHA"
        )
    return normalized


def _validate_core_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise PolicyRevalidationAuditRunError(
            "accepted final result did not expose policy audit counters"
        )
    if (
        set(value) != set(POLICY_AUDIT_COUNT_NAMES)
        or set(value) != _REQUIRED_CORE_COUNTS
    ):
        raise PolicyRevalidationAuditRunError(
            "accepted final policy audit does not match the complete "
            "frozen counter contract"
        )
    validated: dict[str, int] = {}
    for name in sorted(_REQUIRED_CORE_COUNTS):
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PolicyRevalidationAuditRunError(
                f"policy audit counter {name} must be non-negative"
            )
        validated[name] = count
    return validated


class _AcceptedFinalPolicyCollector:
    """Thread-safe collector for exactly one accepted result per PDF."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepted: list[dict[str, int]] = []
        self._errors: list[str] = []

    def record(self, counts: Any) -> None:
        try:
            validated = _validate_core_counts(counts)
        except PolicyRevalidationAuditRunError as exc:
            with self._lock:
                self._errors.append(str(exc))
            return
        with self._lock:
            self._accepted.append(validated)

    def aggregate(
        self, *, expected_count: int
    ) -> tuple[dict[str, int], int]:
        with self._lock:
            errors = tuple(self._errors)
            accepted = tuple(dict(value) for value in self._accepted)
        if errors:
            raise PolicyRevalidationAuditRunError(errors[0])
        if len(accepted) != expected_count:
            raise PolicyRevalidationAuditRunError(
                "accepted final policy result count does not match "
                "the output cohort"
            )
        aggregate = {
            name: sum(value[name] for value in accepted)
            for name in sorted(_REQUIRED_CORE_COUNTS)
        }
        return aggregate, len(accepted)


def _instrument_production_processor(
    processor_factory: Callable[[], Any],
) -> tuple[Any, object, bool, _AcceptedFinalPolicyCollector]:
    production = processor_factory()
    inner = getattr(production, "processor", None)
    if inner is None or not callable(getattr(inner, "process_case", None)):
        raise PolicyRevalidationAuditRunError(
            "production composition root does not expose its inner processor"
        )
    collector = _AcceptedFinalPolicyCollector()
    observer_existed = hasattr(inner, "_policy_audit_observer")
    original_observer = getattr(inner, "_policy_audit_observer", None)
    try:
        setattr(inner, "_policy_audit_observer", collector.record)
    except (AttributeError, TypeError) as exc:
        raise PolicyRevalidationAuditRunError(
            "production final policy result cannot be instrumented"
        ) from exc
    if getattr(inner, "_policy_audit_observer", None) != collector.record:
        raise PolicyRevalidationAuditRunError(
            "production final policy instrumentation was not retained"
        )
    return production, original_observer, observer_existed, collector


def _normalize_cohort_counts(
    core: Mapping[str, int], *, accepted_count: int
) -> dict[str, int]:
    """Map the production contract to the stable public evidence vocabulary."""

    removed = core["contradicted_synthetic_reason_removed_count"]
    remaining = core[
        "contradicted_synthetic_reason_left_active_count"
    ]
    sentinel = core["sentinel_value_used_as_policy_evidence_count"]
    normalized = {
        "accepted_final_policy_result_count": accepted_count,
        "late_recovery_before_revalidation_count": core[
            "late_recovery_before_revalidation_count"
        ],
        "revalidation_after_late_recovery_count": core[
            "normal_policy_rerun_count"
        ],
        "contradicted_synthetic_reason_before_count": (
            removed + remaining
        ),
        "contradicted_synthetic_reason_removed_count": removed,
        "contradicted_synthetic_reason_remaining_count": remaining,
        "independent_denial_reason_retained_count": core[
            "independent_denial_reason_retained_count"
        ],
        "review_confidence_restored_count": core[
            "review_confidence_restored_count"
        ],
        "normal_policy_rerun_count": core["normal_policy_rerun_count"],
        "signed_late_authority_recovery_count": core[
            "signed_late_authority_recovery_count"
        ],
        "late_adjudication_evidence_preserved_count": core[
            "late_adjudication_evidence_preserved_count"
        ],
        "late_biohazard_evidence_preserved_count": core[
            "late_biohazard_evidence_preserved_count"
        ],
        "forced_approval_count": core["forced_approval_count"],
        "sentinel_value_used_as_evidence_count": sentinel,
        "placeholder_value_used_as_evidence_count": core[
            "placeholder_value_used_as_evidence_count"
        ],
        "serialization_default_used_as_evidence_count": core[
            "serialization_default_used_as_policy_evidence_count"
        ],
        "stale_threshold_mismatch_count": core[
            "stale_threshold_mismatch_count"
        ],
    }
    if set(normalized) != set(COHORT_AUDIT_COUNTS):
        raise PolicyRevalidationAuditRunError(
            "normalized cohort audit contract is incomplete"
        )
    return normalized


def _validate_contract_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(
        CONTRACT_AUDIT_COUNTS
    ):
        raise PolicyRevalidationAuditRunError(
            "contract probes did not return the complete counter contract"
        )
    validated: dict[str, int] = {}
    for name in CONTRACT_AUDIT_COUNTS:
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PolicyRevalidationAuditRunError(
                f"contract probe counter {name} must be non-negative"
            )
        validated[name] = count
    return validated


def run_policy_revalidation_audit(
    *,
    input_dir: Path | str,
    predictions_path: Path | str,
    source_revision: str,
    repeat_index: int,
    max_workers: int = 4,
    processor_factory: Callable[[], Any] | None = None,
    contract_probe: Callable[[], Mapping[str, int]] | None = None,
    fixture_digest_provider: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Run production and probes, returning a candidate-byte-bound audit."""

    source_digest = _validate_source_revision(source_revision)
    if isinstance(repeat_index, bool) or repeat_index not in {1, 2}:
        raise PolicyRevalidationAuditRunError(
            "repeat_index must be 1 or 2"
        )
    if not 1 <= max_workers <= 4:
        raise PolicyRevalidationAuditRunError(
            "max_workers must be between 1 and 4"
        )
    directory = Path(input_dir)
    expected_predictions = Path(predictions_path)
    if not directory.is_dir():
        raise PolicyRevalidationAuditRunError(
            "input directory must exist"
        )
    if not expected_predictions.is_file():
        raise PolicyRevalidationAuditRunError(
            "bound candidate predictions must exist"
        )
    pdf_paths = discover_case_pdfs(directory)
    if not pdf_paths:
        raise PolicyRevalidationAuditRunError(
            "input directory contains no PDF cases"
        )

    production, original_observer, observer_existed, collector = (
        _instrument_production_processor(
            processor_factory or build_production_processor
        )
    )
    inner = production.processor
    try:
        with tempfile.TemporaryDirectory(
            prefix="mib-wo17-policy-audit-"
        ) as temporary_directory:
            generated_predictions = (
                Path(temporary_directory) / "predictions.jsonl"
            )
            report = BatchRunner(
                production, max_workers=max_workers
            ).run(directory, generated_predictions)
            generated_sha256 = hashlib.sha256(
                generated_predictions.read_bytes()
            ).hexdigest()
    finally:
        if observer_existed:
            setattr(inner, "_policy_audit_observer", original_observer)
        else:
            try:
                delattr(inner, "_policy_audit_observer")
            except AttributeError:
                pass

    expected_predictions_sha256 = hashlib.sha256(
        expected_predictions.read_bytes()
    ).hexdigest()
    if generated_sha256 != expected_predictions_sha256:
        raise PolicyRevalidationAuditRunError(
            "audited production predictions do not match the bound "
            "candidate prediction bytes"
        )
    if (
        report.attempted != len(pdf_paths)
        or report.answered != len(pdf_paths)
        or report.omitted != 0
        or report.failures
    ):
        raise PolicyRevalidationAuditRunError(
            "production audit output is incomplete"
        )

    core_counts, accepted_count = collector.aggregate(
        expected_count=len(pdf_paths)
    )
    cohort_counts = _normalize_cohort_counts(
        core_counts, accepted_count=accepted_count
    )
    try:
        contract_counts = _validate_contract_counts(
            (contract_probe or run_contract_probes)()
        )
    except PolicyContractProbeError as exc:
        raise PolicyRevalidationAuditRunError(
            f"policy contract probe failed: {exc}"
        ) from exc
    fixture_sha256 = str(
        (fixture_digest_provider or contract_fixture_sha256)()
    ).strip().casefold()
    if not _SHA256_RE.fullmatch(fixture_sha256):
        raise PolicyRevalidationAuditRunError(
            "contract fixture provider must return a full SHA-256"
        )

    return {
        "schema_version": POLICY_REVALIDATION_AUDIT_SCHEMA,
        "source_revision_sha": source_digest,
        "input_tree_sha256": cohort_tree_sha256(pdf_paths),
        "input_pdf_count": len(pdf_paths),
        "case_id_set_sha256": case_id_set_sha256(
            tuple(path.stem for path in pdf_paths)
        ),
        "repeat_index": repeat_index,
        "predictions_sha256": expected_predictions_sha256,
        "contract_fixture_sha256": fixture_sha256,
        "cohort_counts": cohort_counts,
        "contract_counts": contract_counts,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run accepted-final aggregate WO-17 policy auditing."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_policy_revalidation_audit(
            input_dir=args.input_dir,
            predictions_path=args.predictions,
            source_revision=args.source_revision,
            repeat_index=args.repeat_index,
            max_workers=args.max_workers,
        )
        _atomic_write(Path(args.output), canonical_json(payload) + "\n")
    except (
        OSError,
        ExperimentControlError,
        PolicyRevalidationAuditRunError,
    ) as exc:
        print(f"policy revalidation audit error: {exc}", file=sys.stderr)
        return 1
    print("WO-17 accepted-final policy audit complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
