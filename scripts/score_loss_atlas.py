#!/usr/bin/env python3
"""Build an aggregate-only score-loss atlas from official evaluator artifacts.

The atlas deliberately revalidates the four source artifacts against one
another before calculating diagnostics.  A report is therefore rejected when
the evaluator summary, per-case scores, labels, or submission have been mixed
or edited independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCORE_VERSION = "mib_weighted_v1"
FIELD_NAMES = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
)
FIELD_WEIGHTS = {
    "applicant_name": 5,
    "species_code": 6,
    "home_world": 5,
    "visa_class": 5,
    "sponsor_id": 5,
    "arrival_date": 4,
    "declared_purpose": 3,
    "risk_flags": 8,
    "fee_status": 4,
}
SCORE_SCALE = {
    "extraction_points": 50.0,
    "classification_points": 80.0,
    "calibration_points": 20.0,
    "max_score": 150.0,
    "missing_penalty_cap": 10.0,
}
ADJUDICATION_VALUES = {"APPROVED", "DENIED", "NEEDS_REVIEW"}
FEE_VALUES = {"paid", "waived", "unpaid", "unknown"}
CLASSIFICATION_MAX_RAW = 8.0
OUTPUT_DEFAULTS = {
    "applicant_name": "unknown",
    "species_code": "TRIANGULAN",
    "home_world": "Wolf-1061c",
    "visa_class": "XW-1",
    "sponsor_id": "SPN-0000",
    "arrival_date": "1900-01-01",
    "declared_purpose": "reactor maintenance",
    "risk_flags": "none",
    "fee_status": "paid",
}
SENSITIVE_MODAL_FIELDS = {"applicant_name", "sponsor_id", "arrival_date"}
SAFE_LITERAL_MODAL_MIN_SUPPORT = 10
FLOAT_TOLERANCE = 1e-9
DIMENSION_MIN_SUPPORT = 10
TRACE_DIMENSIONS = (
    "provenance_route",
    "applicant_linking_state",
    "evidence_conflict",
    "ocr_recovery_path",
    "policy_trace",
)
REQUIRED_ATLAS_DIMENSIONS = (
    "field",
    "adjudication_confusion",
    "page_template_family",
    *TRACE_DIMENSIONS,
    "confidence_bucket",
    "runtime_cost",
)
_DIMENSION_CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,95}$")
_LAYOUT_CATEGORY_RE = re.compile(
    r"^page-count-[0-9]{2}__ink-bucket-[0-9]{2}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
TRACE_CATEGORY_ENUMS = {
    "provenance_route": frozenset(
        {
            "visible_ocr",
            "authoritative_source",
            "mixed_visible_sources",
            "no_accepted_provenance",
            "unknown",
        }
    ),
    "applicant_linking_state": frozenset(
        {
            "linked_unique",
            "linked_ambiguous",
            "unlinked",
            "authoritative_scope",
            "unknown",
        }
    ),
    "evidence_conflict": frozenset(
        {
            "none",
            "field_conflict",
            "identity_conflict",
            "authority_conflict",
            "multiple_conflicts",
            "unknown",
        }
    ),
    "ocr_recovery_path": frozenset(
        {
            "primary",
            "consensus_retry",
            "sparse_intake_retry",
            "orientation_retry",
            "risk_flag_retry",
            "targeted_rapidocr",
            "multiple_recovery_paths",
            "no_visible_ocr",
            "unknown",
        }
    ),
    "policy_trace": frozenset(
        {
            "binding_authority",
            "deterministic_policy",
            "revalidated_policy",
            "recovery_review",
            "recovery_denial",
            "needs_review_conflict",
            "unknown",
        }
    ),
}


class AtlasInputError(ValueError):
    """The supplied evaluator artifacts cannot form a trustworthy atlas."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prediction_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".csv":
        return [dict(row) for row in _read_csv(path)]
    return _read_jsonl(path)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _index_unique(
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise AtlasInputError(f"{label} contains an empty case_id")
        if case_id in indexed:
            raise AtlasInputError(f"{label} contains duplicate case_id values")
        indexed[case_id] = row
    return indexed


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def _normalize_flags(value: Any) -> str:
    raw = _normalize(value)
    if raw in {"", "none", "null", "unknown"}:
        return "none"
    return "|".join(sorted(part.strip() for part in raw.split("|") if part.strip()))


def _split_pipe(value: Any) -> set[str]:
    raw = _normalize(value)
    if raw in {"", "none", "null"}:
        return set()
    return {part.strip() for part in raw.split("|") if part.strip()}


def _field_matches(
    field_name: str,
    truth: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> bool:
    if field_name == "risk_flags":
        return _normalize_flags(truth.get(field_name)) == _normalize_flags(
            prediction.get(field_name)
        )
    return _normalize(truth.get(field_name)) == _normalize(
        prediction.get(field_name)
    )


def _parse_confidence(prediction: Mapping[str, Any]) -> tuple[float, bool]:
    try:
        value = float(prediction.get("confidence", 0))
    except (TypeError, ValueError):
        return 0.0, False
    if not 0.0 <= value <= 1.0:
        return 0.0, False
    return value, True


def _classification_points(truth_adj: str, pred_adj: str) -> tuple[float, str, bool]:
    if pred_adj not in ADJUDICATION_VALUES:
        return 0.0, "invalid_or_blank", False
    if truth_adj == pred_adj:
        return CLASSIFICATION_MAX_RAW, "correct", False
    if truth_adj == "DENIED" and pred_adj == "APPROVED":
        return -4.0, "catastrophic_false_approval", True
    if pred_adj == "NEEDS_REVIEW":
        return 2.0, "conservative_review", False
    if truth_adj == "NEEDS_REVIEW":
        return 1.0, "missed_review", False
    return 0.0, "wrong_decision", False


def _assert_close(label: str, actual: Any, expected: float) -> None:
    try:
        actual_float = float(actual)
    except (TypeError, ValueError) as exc:
        raise AtlasInputError(f"{label} is not numeric") from exc
    if not math.isclose(
        actual_float,
        float(expected),
        rel_tol=FLOAT_TOLERANCE,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise AtlasInputError(
            f"{label} mismatch: artifact={actual_float!r}, recomputed={expected!r}"
        )


def _assert_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AtlasInputError(
            f"{label} mismatch: artifact={actual!r}, recomputed={expected!r}"
        )


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise AtlasInputError(f"evaluation.{key} must be an object")
    return value


def _component_scores(evaluation: Mapping[str, Any]) -> dict[str, float]:
    scores = _require_mapping(evaluation, "scores")
    scale = _require_mapping(evaluation, "score_scale")
    try:
        return {
            "extraction": float(scores["extraction_score"]),
            "classification": float(scores["classification_score"]),
            "calibration": float(scores["calibration_score"]),
            "total": float(scores["total_score"]),
            "extraction_max": float(scale["extraction_points"]),
            "classification_max": float(scale["classification_points"]),
            "calibration_max": float(scale["calibration_points"]),
            "total_max": float(scale["max_score"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AtlasInputError("evaluation score fields are incomplete or invalid") from exc


def _validate_version_and_scale(evaluation: Mapping[str, Any]) -> None:
    _assert_equal("evaluation.score_version", evaluation.get("score_version"), SCORE_VERSION)
    scale = _require_mapping(evaluation, "score_scale")
    for name, expected in SCORE_SCALE.items():
        if name not in scale:
            raise AtlasInputError(f"evaluation.score_scale.{name} is missing")
        _assert_close(f"evaluation.score_scale.{name}", scale[name], expected)


def _validate_hashes(
    source_sha256: Mapping[str, str] | None,
    *,
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    case_scores: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], str]:
    expected_keys = ("truth", "submission", "evaluation", "case_scores")
    if source_sha256 is None:
        return (
            {
                "truth": _canonical_sha256(list(truth_rows)),
                "submission": _canonical_sha256(list(prediction_rows)),
                "evaluation": _canonical_sha256(evaluation),
                "case_scores": _canonical_sha256(list(case_scores)),
            },
            "canonical_json",
        )

    hashes = {key: str(source_sha256.get(key, "")).casefold() for key in expected_keys}
    for key, value in hashes.items():
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise AtlasInputError(f"source SHA-256 for {key} is missing or invalid")
    return hashes, "file_bytes"


def _validate_artifacts(
    *,
    truth: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    scored: Mapping[str, Mapping[str, Any]],
    prediction_row_count: int,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check every per-case and aggregate evaluator fact."""

    _validate_version_and_scale(evaluation)
    counts = _require_mapping(evaluation, "counts")
    raw = _require_mapping(evaluation, "raw")
    scores = _require_mapping(evaluation, "scores")

    totals = Counter()
    brier_sum = 0.0
    confusion = Counter()
    recomputed: dict[str, Any] = {}

    for case_id, truth_row in truth.items():
        prediction = predictions[case_id]
        case = scored[case_id]
        prefix = f"case_scores[{case_id}]"

        _assert_equal(f"{prefix}.present", case.get("present"), True)
        truth_adj = str(truth_row.get("adjudication", "")).strip().upper()
        pred_adj = str(prediction.get("adjudication", "")).strip().upper()
        _assert_equal(
            f"{prefix}.truth_adjudication",
            case.get("truth_adjudication"),
            truth_adj,
        )
        _assert_equal(
            f"{prefix}.pred_adjudication",
            case.get("pred_adjudication"),
            pred_adj,
        )

        field_results = case.get("field_results")
        if not isinstance(field_results, Mapping):
            raise AtlasInputError(f"{prefix}.field_results must be an object")
        _assert_equal(
            f"{prefix}.field_results keys",
            set(field_results),
            set(FIELD_NAMES),
        )
        unrecoverable = _split_pipe(truth_row.get("unrecoverable_fields", ""))
        case_extraction_raw = 0.0
        case_extraction_max_raw = 0.0
        for field_name in FIELD_NAMES:
            result = field_results[field_name]
            if not isinstance(result, Mapping):
                raise AtlasInputError(
                    f"{prefix}.field_results.{field_name} must be an object"
                )
            if field_name in unrecoverable:
                expected_status = "not_scorable_unrecoverable"
                expected_points = 0.0
                expected_max = 0.0
            else:
                expected_max = float(FIELD_WEIGHTS[field_name])
                expected_status = (
                    "matched"
                    if _field_matches(field_name, truth_row, prediction)
                    else "missed"
                )
                expected_points = expected_max if expected_status == "matched" else 0.0
                case_extraction_raw += expected_points
                case_extraction_max_raw += expected_max
            _assert_equal(
                f"{prefix}.field_results.{field_name}.status",
                result.get("status"),
                expected_status,
            )
            _assert_close(
                f"{prefix}.field_results.{field_name}.points",
                result.get("points"),
                expected_points,
            )
            _assert_close(
                f"{prefix}.field_results.{field_name}.max_points",
                result.get("max_points"),
                expected_max,
            )

        _assert_close(
            f"{prefix}.extraction_raw",
            case.get("extraction_raw"),
            case_extraction_raw,
        )
        _assert_close(
            f"{prefix}.extraction_max_raw",
            case.get("extraction_max_raw"),
            case_extraction_max_raw,
        )

        classification_raw, reason, catastrophic = _classification_points(
            truth_adj, pred_adj
        )
        _assert_close(
            f"{prefix}.classification_raw",
            case.get("classification_raw"),
            classification_raw,
        )
        _assert_close(
            f"{prefix}.classification_max_raw",
            case.get("classification_max_raw"),
            CLASSIFICATION_MAX_RAW,
        )
        _assert_equal(
            f"{prefix}.classification_reason",
            case.get("classification_reason"),
            reason,
        )
        _assert_equal(
            f"{prefix}.catastrophic_false_approval",
            case.get("catastrophic_false_approval"),
            catastrophic,
        )

        confidence, confidence_valid = _parse_confidence(prediction)
        _assert_close(f"{prefix}.confidence", case.get("confidence"), confidence)
        _assert_equal(
            f"{prefix}.confidence_valid",
            case.get("confidence_valid"),
            confidence_valid,
        )
        correct = truth_adj == pred_adj
        expected_brier = (
            (confidence - (1.0 if correct else 0.0)) ** 2
            if confidence_valid
            else 1.0
        )
        _assert_close(
            f"{prefix}.confidence_brier",
            case.get("confidence_brier"),
            expected_brier,
        )

        adjudication_valid = pred_adj in ADJUDICATION_VALUES
        fee_status_valid = str(prediction.get("fee_status", "")).strip() in FEE_VALUES
        _assert_equal(
            f"{prefix}.adjudication_valid",
            case.get("adjudication_valid"),
            adjudication_valid,
        )
        _assert_equal(
            f"{prefix}.fee_status_valid",
            case.get("fee_status_valid"),
            fee_status_valid,
        )
        _assert_close(
            f"{prefix}.missing_penalty_score",
            case.get("missing_penalty_score"),
            0.0,
        )

        totals["extraction_raw"] += case_extraction_raw
        totals["extraction_max_raw"] += case_extraction_max_raw
        totals["classification_raw"] += classification_raw
        totals["classification_max_raw"] += CLASSIFICATION_MAX_RAW
        totals["catastrophic_false_approvals"] += int(catastrophic)
        totals["invalid_adjudication_records"] += int(not adjudication_valid)
        totals["invalid_confidence_records"] += int(not confidence_valid)
        totals["invalid_fee_status_records"] += int(not fee_status_valid)
        brier_sum += expected_brier
        confusion[f"{truth_adj}->{pred_adj}"] += 1
        recomputed[case_id] = {
            "missed_fields": {
                field_name
                for field_name in FIELD_NAMES
                if field_results[field_name]["status"] == "missed"
            },
            "scorable_fields": {
                field_name
                for field_name in FIELD_NAMES
                if field_results[field_name]["status"] != "not_scorable_unrecoverable"
            },
            "decision_correct": correct,
            "confidence": confidence,
            "confidence_valid": confidence_valid,
            "confidence_brier": expected_brier,
        }

    case_count = len(truth)
    expected_counts = {
        "truth_cases": case_count,
        "submitted_records": prediction_row_count,
        "scored_predictions": case_count,
        "missing_cases": 0,
        "extra_cases": 0,
        "duplicate_case_ids": 0,
        "blank_case_rows": 0,
        "invalid_adjudication_records": totals["invalid_adjudication_records"],
        "invalid_confidence_records": totals["invalid_confidence_records"],
        "invalid_fee_status_records": totals["invalid_fee_status_records"],
    }
    for name, expected in expected_counts.items():
        _assert_equal(f"evaluation.counts.{name}", counts.get(name), expected)

    mean_brier = brier_sum / case_count if case_count else None
    for name in (
        "extraction_raw",
        "extraction_max_raw",
        "classification_raw",
        "classification_max_raw",
        "catastrophic_false_approvals",
    ):
        _assert_close(f"evaluation.raw.{name}", raw.get(name), totals[name])
    if mean_brier is None:
        _assert_equal(
            "evaluation.raw.mean_confidence_brier",
            raw.get("mean_confidence_brier"),
            None,
        )
    else:
        _assert_close(
            "evaluation.raw.mean_confidence_brier",
            raw.get("mean_confidence_brier"),
            mean_brier,
        )

    expected_confusion = dict(sorted(confusion.items()))
    _assert_equal(
        "evaluation.confusion",
        dict(evaluation.get("confusion", {})),
        expected_confusion,
    )

    extraction_score = SCORE_SCALE["extraction_points"] * (
        totals["extraction_raw"] / totals["extraction_max_raw"]
        if totals["extraction_max_raw"]
        else 0.0
    )
    classification_score = SCORE_SCALE["classification_points"] * (
        totals["classification_raw"] / totals["classification_max_raw"]
        if totals["classification_max_raw"]
        else 0.0
    )
    calibration_score = (
        SCORE_SCALE["calibration_points"] * max(0.0, 1.0 - 2.0 * mean_brier)
        if mean_brier is not None
        else 0.0
    )
    missing_penalty = 0.0
    total_score = (
        extraction_score + classification_score + calibration_score - missing_penalty
    )
    for name, expected in {
        "extraction_score": extraction_score,
        "classification_score": classification_score,
        "calibration_score": calibration_score,
        "missing_penalty": missing_penalty,
        "total_score": total_score,
    }.items():
        _assert_close(f"evaluation.scores.{name}", scores.get(name), expected)

    return {
        "per_case": recomputed,
        "totals": totals,
        "mean_brier": mean_brier,
    }


def _safe_wrong_output_modes(
    field_name: str,
    wrong_values: Sequence[Any],
) -> dict[str, Any]:
    normalized_values = [
        _normalize_flags(value) if field_name == "risk_flags" else _normalize(value)
        for value in wrong_values
    ]
    raw_counts = Counter(normalized_values)
    default_value = (
        _normalize_flags(OUTPUT_DEFAULTS[field_name])
        if field_name == "risk_flags"
        else _normalize(OUTPUT_DEFAULTS[field_name])
    )
    if not raw_counts:
        return {
            "value": None,
            "count": 0,
            "share_of_misses": 0.0,
            "privacy": "no_misses",
        }
    raw_value, count = sorted(
        raw_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    if not raw_value:
        value = "<blank>"
        privacy = "safe_bucket"
    elif raw_value == default_value:
        value = "<configured_default>"
        privacy = "safe_bucket"
    elif field_name in SENSITIVE_MODAL_FIELDS:
        value = "<other_nonblank_redacted>"
        privacy = "sensitive_value_redacted"
    elif count < SAFE_LITERAL_MODAL_MIN_SUPPORT:
        value = "<suppressed_low_support>"
        privacy = "low_support_value_suppressed"
    else:
        value = raw_value
        privacy = "literal_aggregate"
    return {
        "value": value,
        "count": count,
        "share_of_misses": count / len(normalized_values),
        "privacy": privacy,
    }


def _pairwise_miss_diagnostics(
    per_case: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for field_a, field_b in itertools.combinations(FIELD_NAMES, 2):
        eligible = [
            case
            for case in per_case.values()
            if field_a in case["scorable_fields"]
            and field_b in case["scorable_fields"]
        ]
        a_misses = sum(field_a in case["missed_fields"] for case in eligible)
        b_misses = sum(field_b in case["missed_fields"] for case in eligible)
        co_misses = sum(
            field_a in case["missed_fields"] and field_b in case["missed_fields"]
            for case in eligible
        )
        expected = (
            a_misses * b_misses / len(eligible)
            if eligible
            else 0.0
        )
        pairs.append(
            {
                "field_a": field_a,
                "field_b": field_b,
                "eligible_cases": len(eligible),
                "field_a_misses": a_misses,
                "field_b_misses": b_misses,
                "co_missed_cases": co_misses,
                "expected_co_misses_under_independence": expected,
                "enrichment_lift": co_misses / expected if expected else None,
                "excess_co_misses": co_misses - expected,
            }
        )
    return sorted(
        pairs,
        key=lambda item: (
            -item["co_missed_cases"],
            -(item["enrichment_lift"] or 0.0),
            item["field_a"],
            item["field_b"],
        ),
    )


def _decision_field_decoupling(
    per_case: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cells = Counter()
    for case in per_case.values():
        decision = "correct_decision" if case["decision_correct"] else "wrong_decision"
        fields = (
            "all_fields_correct"
            if not case["missed_fields"]
            else "one_or_more_field_misses"
        )
        cells[f"{decision}_{fields}"] += 1
    wrong_cases = (
        cells["wrong_decision_all_fields_correct"]
        + cells["wrong_decision_one_or_more_field_misses"]
    )
    return {
        "correct_decision_all_fields_correct": cells[
            "correct_decision_all_fields_correct"
        ],
        "correct_decision_one_or_more_field_misses": cells[
            "correct_decision_one_or_more_field_misses"
        ],
        "wrong_decision_all_fields_correct": cells[
            "wrong_decision_all_fields_correct"
        ],
        "wrong_decision_one_or_more_field_misses": cells[
            "wrong_decision_one_or_more_field_misses"
        ],
        "wrong_decision_cases": wrong_cases,
        "wrong_decision_all_fields_correct_share": (
            cells["wrong_decision_all_fields_correct"] / wrong_cases
            if wrong_cases
            else 0.0
        ),
    }


def _empirical_calibration_ceiling(
    *,
    scored: Mapping[str, Mapping[str, Any]],
    per_case: Mapping[str, Mapping[str, Any]],
    current_calibration_score: float,
    current_total_score: float,
) -> dict[str, Any]:
    """Return the in-sample oracle for exact output+confidence groups.

    Exact confidence values are never emitted.  Only group counts and the
    aggregate optimum are retained, keeping the atlas identity-free.
    """

    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for case_id, case in scored.items():
        confidence = per_case[case_id]["confidence"]
        confidence_key = (
            f"valid:{float(confidence).hex()}"
            if per_case[case_id]["confidence_valid"]
            else "invalid"
        )
        groups[(str(case["pred_adjudication"]), confidence_key)].append(
            bool(per_case[case_id]["decision_correct"])
        )
    oracle_brier_sum = 0.0
    support_histogram: Counter[str] = Counter()
    for outcomes in groups.values():
        support = len(outcomes)
        successes = sum(outcomes)
        empirical_accuracy = successes / support
        oracle_brier_sum += support * empirical_accuracy * (1.0 - empirical_accuracy)
        if support == 1:
            support_histogram["1"] += 1
        elif support <= 4:
            support_histogram["2-4"] += 1
        elif support <= 9:
            support_histogram["5-9"] += 1
        else:
            support_histogram["10+"] += 1

    case_count = len(per_case)
    oracle_mean_brier = oracle_brier_sum / case_count if case_count else 0.0
    oracle_calibration_score = SCORE_SCALE["calibration_points"] * max(
        0.0, 1.0 - 2.0 * oracle_mean_brier
    )
    gain = max(0.0, oracle_calibration_score - current_calibration_score)
    return {
        "grouping": "predicted_adjudication_x_exact_submitted_confidence",
        "group_count": len(groups),
        "support_histogram": dict(sorted(support_histogram.items())),
        "exact_confidence_values_emitted": False,
        "empirical_oracle_mean_brier": oracle_mean_brier,
        "empirical_oracle_calibration_score": oracle_calibration_score,
        "empirical_calibration_gain_ceiling": gain,
        "empirical_total_score_ceiling": current_total_score + gain,
        "interpretation": (
            "In-sample upper bound from replacing each exact output+confidence "
            "group with its public-label empirical correctness rate; singleton "
            "groups make this optimistic and it is not validation evidence."
        ),
    }


def _dimension_category(
    value: Any,
    *,
    dimension: str,
    allowed: frozenset[str] | None = None,
) -> str:
    category = str(value or "").strip().casefold()
    if not _DIMENSION_CATEGORY_RE.fullmatch(category):
        raise AtlasInputError(
            f"{dimension} category must be a generic lower-case token"
        )
    if re.fullmatch(r"mib-[0-9]{6}", category):
        raise AtlasInputError(f"{dimension} category resembles a case identifier")
    if allowed is not None and category not in allowed:
        raise AtlasInputError(
            f"{dimension} category is not in the versioned allowlist"
        )
    return category


def _dimension_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_ids: set[str],
    dimension: str,
    value_key: str,
    exact_keys: set[str],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        if set(row) != exact_keys:
            raise AtlasInputError(
                f"{dimension} rows must contain exactly {sorted(exact_keys)}"
            )
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise AtlasInputError(f"{dimension} contains an empty case_id")
        if case_id in labels:
            raise AtlasInputError(f"{dimension} contains duplicate case_id values")
        labels[case_id] = _dimension_category(
            row.get(value_key),
            dimension=dimension,
        )
    if set(labels) != case_ids:
        raise AtlasInputError(
            f"{dimension} must cover exactly the evaluated case set"
        )
    return labels


def _loss_by_dimension(
    *,
    labels: Mapping[str, str],
    scored: Mapping[str, Mapping[str, Any]],
    per_case: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, float],
    suppress_low_support: bool = True,
) -> dict[str, Any]:
    raw_support = Counter(labels.values())
    low_support_categories = {
        category
        for category, support in raw_support.items()
        if suppress_low_support and support < DIMENSION_MIN_SUPPORT
    }
    raw_low_support = sum(
        support
        for category, support in raw_support.items()
        if category in low_support_categories
    )
    suppressed_categories = set(low_support_categories)
    complementary_category: str | None = None
    if low_support_categories and raw_low_support < DIMENSION_MIN_SUPPORT:
        candidates = [
            (support, category)
            for category, support in raw_support.items()
            if category not in low_support_categories
        ]
        if candidates:
            _, complementary_category = min(candidates)
            suppressed_categories.add(complementary_category)
    suppression_pool_support = sum(
        support
        for category, support in raw_support.items()
        if category in suppressed_categories
    )
    emit_low_support_pool = (
        bool(suppressed_categories)
        and suppression_pool_support >= DIMENSION_MIN_SUPPORT
    )
    extraction_max_raw = sum(
        float(case["extraction_max_raw"]) for case in scored.values()
    )
    classification_max_raw = sum(
        float(case["classification_max_raw"]) for case in scored.values()
    )
    calibration_gap = components["calibration_max"] - components["calibration"]
    total_brier_sum = sum(
        float(case["confidence_brier"]) for case in scored.values()
    )
    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id, category in labels.items():
        if category in suppressed_categories:
            if not emit_low_support_pool:
                continue
            safe_category = "<suppressed_low_support>"
        else:
            safe_category = category
        grouped[safe_category].append(case_id)

    groups: list[dict[str, Any]] = []
    for category, case_ids in grouped.items():
        extraction_raw_loss = 0.0
        classification_raw_loss = 0.0
        brier_sum = 0.0
        field_misses = 0
        wrong_decisions = 0
        for case_id in case_ids:
            case = scored[case_id]
            extraction_raw_loss += (
                float(case["extraction_max_raw"])
                - float(case["extraction_raw"])
            )
            classification_raw_loss += (
                float(case["classification_max_raw"])
                - float(case["classification_raw"])
            )
            brier_sum += float(case["confidence_brier"])
            field_misses += len(per_case[case_id]["missed_fields"])
            wrong_decisions += int(not per_case[case_id]["decision_correct"])
        extraction_loss = (
            extraction_raw_loss
            / extraction_max_raw
            * components["extraction_max"]
            if extraction_max_raw
            else 0.0
        )
        classification_loss = (
            classification_raw_loss
            / classification_max_raw
            * components["classification_max"]
            if classification_max_raw
            else 0.0
        )
        calibration_loss = (
            calibration_gap * brier_sum / total_brier_sum
            if total_brier_sum
            else 0.0
        )
        groups.append(
            {
                "category": category,
                "case_count": len(case_ids),
                "case_share": len(case_ids) / len(scored) if scored else 0.0,
                "field_misses": field_misses,
                "wrong_decision_cases": wrong_decisions,
                "extraction_score_loss": extraction_loss,
                "classification_score_loss": classification_loss,
                "calibration_score_loss_contribution": calibration_loss,
                "accounted_score_loss": (
                    extraction_loss + classification_loss + calibration_loss
                ),
            }
        )
    groups.sort(
        key=lambda item: (
            -item["accounted_score_loss"],
            -item["case_count"],
            item["category"],
        )
    )
    return {
        "case_count": len(scored),
        "emitted_case_count": sum(group["case_count"] for group in groups),
        "emitted_category_count": len(groups),
        "low_support_case_count": (
            raw_low_support
            if emit_low_support_pool and complementary_category is None
            else None
        ),
        "suppression_pool_case_count": (
            suppression_pool_support if emit_low_support_pool else None
        ),
        "low_support_pool_emitted": emit_low_support_pool,
        "complementary_suppression_applied": complementary_category is not None,
        "under_k_residual_omitted": bool(
            low_support_categories and not emit_low_support_pool
        ),
        "minimum_literal_support": (
            DIMENSION_MIN_SUPPORT if suppress_low_support else 1
        ),
        "groups": groups,
    }


def _confidence_bucket(confidence: float, valid: bool) -> str:
    if not valid:
        return "invalid"
    bucket = min(9, int(confidence * 10.0))
    return f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}"


def _runtime_bucket(runtime_seconds: float) -> str:
    if runtime_seconds < 1.0:
        return "under_1s"
    if runtime_seconds < 2.0:
        return "1s_to_under_2s"
    if runtime_seconds < 4.0:
        return "2s_to_under_4s"
    if runtime_seconds < 8.0:
        return "4s_to_under_8s"
    return "8s_or_more"


def _blocked_dimension(reason: str, required_input: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "required_input": required_input,
    }


def _layout_evidence(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise AtlasInputError("layout manifest must be an object")
    if payload.get("schema") != "mib-wo12-layout-groups/v2":
        raise AtlasInputError("layout manifest schema is unsupported")
    if payload.get("label_blind_construction") is not True:
        raise AtlasInputError("layout manifest must attest label-blind construction")
    layout_signature = payload.get("layout_signature")
    if not isinstance(layout_signature, Mapping):
        raise AtlasInputError("layout manifest layout_signature must be an object")
    if layout_signature.get("version") != "page-count-plus-first-page-ink-v1":
        raise AtlasInputError("layout signature version is unsupported")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise AtlasInputError("layout manifest cases must be an array")
    rows: list[dict[str, Any]] = []
    for item in cases:
        if not isinstance(item, Mapping) or set(item) != {
            "case_id",
            "layout_group",
        }:
            raise AtlasInputError(
                "layout manifest cases must contain only case_id and layout_group"
            )
        layout_group = item["layout_group"]
        if (
            not isinstance(layout_group, str)
            or not _LAYOUT_CATEGORY_RE.fullmatch(layout_group)
        ):
            raise AtlasInputError(
                "layout_group must match the canonical "
                "page-count-NN__ink-bucket-NN schema"
            )
        rows.append(dict(item))
    return rows, {
        "file_sha256": _sha256_path(path),
        "schema": payload["schema"],
        "layout_signature_version": layout_signature["version"],
        "label_blind_construction": True,
    }


def _trace_evidence(
    path: Path,
    *,
    source_sha256: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    expected_keys = {
        "schema_version",
        "source_revision_sha",
        "input_tree_sha256",
        "truth_sha256",
        "submission_sha256",
        "case_count",
        "rows",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise AtlasInputError(
            "trace evidence must use the exact versioned top-level schema"
        )
    if payload["schema_version"] != "mib-score-loss-dimension-trace/v1":
        raise AtlasInputError("trace evidence schema_version is unsupported")
    source_revision = str(payload["source_revision_sha"]).casefold()
    input_tree_sha256 = str(payload["input_tree_sha256"]).casefold()
    truth_sha256 = str(payload["truth_sha256"]).casefold()
    submission_sha256 = str(payload["submission_sha256"]).casefold()
    if not _REVISION_RE.fullmatch(source_revision):
        raise AtlasInputError("trace source_revision_sha is invalid")
    for label, digest in (
        ("input_tree_sha256", input_tree_sha256),
        ("truth_sha256", truth_sha256),
        ("submission_sha256", submission_sha256),
    ):
        if not _SHA256_RE.fullmatch(digest):
            raise AtlasInputError(f"trace {label} is invalid")
    if truth_sha256 != source_sha256["truth"]:
        raise AtlasInputError("trace truth hash does not match the atlas source")
    if submission_sha256 != source_sha256["submission"]:
        raise AtlasInputError("trace submission hash does not match the atlas source")
    rows = payload["rows"]
    if not isinstance(rows, list) or payload["case_count"] != len(rows):
        raise AtlasInputError("trace case_count does not match rows")
    if not isinstance(payload["case_count"], int) or payload["case_count"] < 1:
        raise AtlasInputError("trace case_count must be a positive integer")
    if any(not isinstance(row, Mapping) for row in rows):
        raise AtlasInputError("trace rows must be objects")
    return [dict(row) for row in rows], {
        "file_sha256": _sha256_path(path),
        "schema_version": payload["schema_version"],
        "source_revision_sha": source_revision,
        "input_tree_sha256": input_tree_sha256,
        "truth_sha256": truth_sha256,
        "submission_sha256": submission_sha256,
    }


def _build_dimension_atlas(
    *,
    predictions: Mapping[str, Mapping[str, Any]],
    scored: Mapping[str, Mapping[str, Any]],
    per_case: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, float],
    source_sha256: Mapping[str, str],
    layout_manifest_path: Path | None,
    trace_dimensions_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, str]]:
    case_ids = set(scored)
    evidence_hashes: dict[str, str] = {}

    confidence_labels = {
        case_id: _confidence_bucket(
            per_case[case_id]["confidence"],
            per_case[case_id]["confidence_valid"],
        )
        for case_id in sorted(case_ids)
    }
    dimensions: dict[str, Any] = {
        "confidence_bucket": {
            **_loss_by_dimension(
                labels=confidence_labels,
                scored=scored,
                per_case=per_case,
                components=components,
            ),
            "status": "frozen_baseline",
            "source": "final_submitted_confidence",
            "bucket_definition": "fixed_deciles_left_closed_right_open_except_1.0",
        }
    }

    if layout_manifest_path is None:
        dimensions["page_template_family"] = _blocked_dimension(
            "No label-blind layout manifest was supplied.",
            "--layout-manifest",
        )
    else:
        layout_rows, layout_binding = _layout_evidence(layout_manifest_path)
        evidence_hashes["layout_manifest"] = layout_binding["file_sha256"]
        layout_labels = _dimension_labels(
            layout_rows,
            case_ids=case_ids,
            dimension="page_template_family",
            value_key="layout_group",
            exact_keys={"case_id", "layout_group"},
        )
        dimensions["page_template_family"] = {
            **_loss_by_dimension(
                labels=layout_labels,
                scored=scored,
                per_case=per_case,
                components=components,
            ),
            "status": "auxiliary_historical",
            "source": "label_blind_layout_manifest",
            "binding": {
                **layout_binding,
                "exact_case_set_verified": True,
                "exact_source_revision_verified": False,
                "input_tree_sha256_verified": False,
            },
            "qualification": (
                "Auxiliary only: the manifest has exact case coverage but does "
                "not bind the frozen baseline source revision and input tree."
            ),
        }

    trace_by_case: dict[str, Mapping[str, Any]] | None = None
    expected_trace_keys = {"case_id", *TRACE_DIMENSIONS, "runtime_seconds"}
    trace_binding: dict[str, Any] | None = None
    if trace_dimensions_path is not None:
        trace_rows, trace_binding = _trace_evidence(
            trace_dimensions_path,
            source_sha256=source_sha256,
        )
        evidence_hashes["trace_dimensions"] = trace_binding["file_sha256"]
        trace_by_case = {}
        for row in trace_rows:
            if set(row) != expected_trace_keys:
                raise AtlasInputError(
                    "trace rows must contain exactly "
                    f"{sorted(expected_trace_keys)}"
                )
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                raise AtlasInputError("trace rows contain an empty case_id")
            if case_id in trace_by_case:
                raise AtlasInputError("trace rows contain duplicate case_id values")
            try:
                runtime_seconds = float(row["runtime_seconds"])
            except (TypeError, ValueError) as exc:
                raise AtlasInputError(
                    "trace runtime_seconds must be numeric"
                ) from exc
            if not math.isfinite(runtime_seconds) or runtime_seconds < 0.0:
                raise AtlasInputError(
                    "trace runtime_seconds must be finite and non-negative"
                )
            normalized = {
                "case_id": case_id,
                "runtime_seconds": runtime_seconds,
            }
            for dimension in TRACE_DIMENSIONS:
                normalized[dimension] = _dimension_category(
                    row[dimension],
                    dimension=dimension,
                    allowed=TRACE_CATEGORY_ENUMS[dimension],
                )
            trace_by_case[case_id] = normalized
        if set(trace_by_case) != case_ids:
            raise AtlasInputError(
                "trace dimensions must cover exactly the evaluated case set"
            )

    for dimension in TRACE_DIMENSIONS:
        if trace_by_case is None:
            dimensions[dimension] = _blocked_dimension(
                "The frozen evaluator artifacts do not contain this production trace.",
                "--trace-dimensions",
            )
            continue
        labels = {
            case_id: str(row[dimension])
            for case_id, row in trace_by_case.items()
        }
        dimensions[dimension] = {
            **_loss_by_dimension(
                labels=labels,
                scored=scored,
                per_case=per_case,
                components=components,
            ),
            "status": "auxiliary_historical",
            "source": "development_only_trace",
            "binding": {
                **dict(trace_binding or {}),
                "source_revision_verified_against_authority": False,
                "input_tree_verified_against_authority": False,
            },
            "qualification": (
                "Auxiliary only: the file, schema, truth, and evaluated output "
                "are bound. Source revision and input tree are self-declared "
                "metadata without an authoritative frozen-baseline comparison."
            ),
        }

    if trace_by_case is not None:
        runtime_labels = {
            case_id: _runtime_bucket(float(row["runtime_seconds"]))
            for case_id, row in trace_by_case.items()
        }
        total_runtime = sum(
            float(row["runtime_seconds"]) for row in trace_by_case.values()
        )
        dimensions["runtime_cost"] = {
            **_loss_by_dimension(
                labels=runtime_labels,
                scored=scored,
                per_case=per_case,
                components=components,
            ),
            "status": "auxiliary_historical",
            "source": "development_only_trace",
            "binding": {
                **dict(trace_binding or {}),
                "source_revision_verified_against_authority": False,
                "input_tree_verified_against_authority": False,
            },
            "total_runtime_seconds": total_runtime,
            "mean_runtime_seconds_per_case": (
                total_runtime / len(trace_by_case) if trace_by_case else 0.0
            ),
            "bucket_definition": (
                "under_1s,1s_to_under_2s,2s_to_under_4s,"
                "4s_to_under_8s,8s_or_more"
            ),
        }
    else:
        dimensions["runtime_cost"] = _blocked_dimension(
            "No source-bound per-case runtime trace was supplied; the committed "
            "report separately cites the frozen whole-run ledger total.",
            "--trace-dimensions",
        )

    coverage = {
        "field": "frozen_baseline",
        "adjudication_confusion": "frozen_baseline",
        **{
            dimension: details["status"]
            for dimension, details in dimensions.items()
        },
    }
    if set(coverage) != set(REQUIRED_ATLAS_DIMENSIONS):
        raise AssertionError("required dimension coverage is incomplete")
    blockers: list[dict[str, str]] = []
    for dimension in REQUIRED_ATLAS_DIMENSIONS:
        status = coverage[dimension]
        if status in {"frozen_baseline", "current_source"}:
            continue
        if status == "auxiliary_historical":
            blockers.append(
                {
                    "dimension": dimension,
                    "status": status,
                    "reason": (
                        "Auxiliary evidence is not exact current-source evidence "
                        "for the frozen baseline."
                    ),
                }
            )
            continue
        details = dimensions[dimension]
        blockers.append(
            {
                "dimension": dimension,
                "status": status,
                "reason": str(details["reason"]),
            }
        )
    return dimensions, blockers, evidence_hashes


def build_atlas(
    *,
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    case_scores: Sequence[Mapping[str, Any]],
    target_score: float = 148.0,
    source_sha256: Mapping[str, str] | None = None,
    layout_manifest_path: Path | None = None,
    trace_dimensions_path: Path | None = None,
) -> dict[str, Any]:
    """Return aggregate score losses after cross-artifact validation."""

    truth = _index_unique(truth_rows, label="truth")
    predictions = _index_unique(prediction_rows, label="submission")
    scored = _index_unique(case_scores, label="case scores")
    if set(truth) != set(predictions) or set(truth) != set(scored):
        raise AtlasInputError(
            "truth, submission, and case-score artifacts must have identical coverage"
        )

    validated = _validate_artifacts(
        truth=truth,
        predictions=predictions,
        scored=scored,
        prediction_row_count=len(prediction_rows),
        evaluation=evaluation,
    )
    hashes, hash_basis = _validate_hashes(
        source_sha256,
        truth_rows=truth_rows,
        prediction_rows=prediction_rows,
        evaluation=evaluation,
        case_scores=case_scores,
    )
    components = _component_scores(evaluation)
    residual_loss = components["total_max"] - components["total"]
    required_gain = max(0.0, target_score - components["total"])
    required_recovery_fraction = (
        required_gain / residual_loss if residual_loss else 0.0
    )
    extraction_max_raw = float(evaluation["raw"]["extraction_max_raw"])
    classification_max_raw = float(evaluation["raw"]["classification_max_raw"])
    case_count = len(scored)

    fields: dict[str, dict[str, Any]] = {}
    for field_name in FIELD_NAMES:
        missed = 0
        scorable = 0
        raw_loss = 0.0
        default_on_miss = 0
        wrong_values: list[Any] = []
        for case_id, case in scored.items():
            result = case["field_results"][field_name]
            if result["status"] == "not_scorable_unrecoverable":
                continue
            scorable += 1
            lost = float(result["max_points"]) - float(result["points"])
            raw_loss += lost
            if result["status"] != "matched":
                missed += 1
                predicted = predictions[case_id].get(field_name, "")
                wrong_values.append(predicted)
                pred_normalized = (
                    _normalize_flags(predicted)
                    if field_name == "risk_flags"
                    else _normalize(predicted)
                )
                default_normalized = (
                    _normalize_flags(OUTPUT_DEFAULTS[field_name])
                    if field_name == "risk_flags"
                    else _normalize(OUTPUT_DEFAULTS[field_name])
                )
                if pred_normalized == default_normalized:
                    default_on_miss += 1
        fields[field_name] = {
            "scorable": scorable,
            "matched": scorable - missed,
            "missed": missed,
            "match_rate": (scorable - missed) / scorable if scorable else 0.0,
            "raw_weighted_loss": raw_loss,
            "normalized_score_loss": (
                raw_loss / extraction_max_raw * components["extraction_max"]
                if extraction_max_raw
                else 0.0
            ),
            "configured_output_default": OUTPUT_DEFAULTS[field_name],
            "default_on_missed_cases": default_on_miss,
            "default_share_of_misses": default_on_miss / missed if missed else 0.0,
            "modal_wrong_output": _safe_wrong_output_modes(
                field_name, wrong_values
            ),
        }

    calibration_gap = components["calibration_max"] - components["calibration"]
    total_brier_sum = sum(
        float(case["confidence_brier"]) for case in scored.values()
    )
    confusion_groups: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for case_id, case in scored.items():
        key = f"{case['truth_adjudication']}->{case['pred_adjudication']}"
        grouped[key].append((case_id, case))

    for key, members in sorted(grouped.items()):
        raw_loss = sum(
            float(case["classification_max_raw"]) - float(case["classification_raw"])
            for _, case in members
        )
        brier_sum = sum(float(case["confidence_brier"]) for _, case in members)
        field_misses = Counter()
        all_fields_correct = 0
        for case_id, _ in members:
            missed_fields = validated["per_case"][case_id]["missed_fields"]
            field_misses.update(missed_fields)
            if not missed_fields:
                all_fields_correct += 1
        confusion_groups[key] = {
            "cases": len(members),
            "classification_raw_loss": raw_loss,
            "classification_score_loss": (
                raw_loss / classification_max_raw
                * components["classification_max"]
                if classification_max_raw
                else 0.0
            ),
            "all_fields_correct_cases": all_fields_correct,
            "field_miss_counts": dict(sorted(field_misses.items())),
            "mean_brier": brier_sum / len(members),
            "calibration_score_loss_contribution": (
                calibration_gap * brier_sum / total_brier_sum
                if total_brier_sum
                else 0.0
            ),
        }

    miss_histogram = Counter(
        len(case["missed_fields"]) for case in validated["per_case"].values()
    )
    total_field_misses = sum(
        miss_count * cases for miss_count, cases in miss_histogram.items()
    )
    extraction_gap = components["extraction_max"] - components["extraction"]
    classification_gap = (
        components["classification_max"] - components["classification"]
    )
    field_priority = sorted(
        (
            {
                "field": field_name,
                "missed": details["missed"],
                "normalized_score_loss": details["normalized_score_loss"],
            }
            for field_name, details in fields.items()
        ),
        key=lambda item: (-item["normalized_score_loss"], item["field"]),
    )
    confusion_priority = sorted(
        (
            {
                "confusion": key,
                "cases": details["cases"],
                "classification_score_loss": details["classification_score_loss"],
            }
            for key, details in confusion_groups.items()
            if details["classification_score_loss"] > 0
        ),
        key=lambda item: (-item["classification_score_loss"], item["confusion"]),
    )
    dimension_atlas, dimension_blockers, dimension_hashes = (
        _build_dimension_atlas(
            predictions=predictions,
            scored=scored,
            per_case=validated["per_case"],
            components=components,
            source_sha256=hashes,
            layout_manifest_path=layout_manifest_path,
            trace_dimensions_path=trace_dimensions_path,
        )
    )
    dimension_coverage = {
        "field": "frozen_baseline",
        "adjudication_confusion": "frozen_baseline",
        **{
            dimension: details["status"]
            for dimension, details in dimension_atlas.items()
        },
    }

    return {
        "report_version": "mib_score_loss_atlas_v3",
        "evidence_class": "public_full_training_diagnostic_not_unseen",
        "source_sha256": hashes,
        "source_hash_basis": hash_basis,
        "dimension_source_sha256": dimension_hashes,
        "case_count": case_count,
        "target_score": target_score,
        "current_scores": {
            "extraction": components["extraction"],
            "classification": components["classification"],
            "calibration": components["calibration"],
            "total": components["total"],
        },
        "score_gaps": {
            "extraction": extraction_gap,
            "classification": classification_gap,
            "calibration": calibration_gap,
            "total_to_perfect": residual_loss,
            "gain_required_for_target": required_gain,
            "remaining_error_recovery_required": required_recovery_fraction,
        },
        "field_losses": fields,
        "field_priority": field_priority,
        "field_miss_case_summary": {
            "total_field_misses": total_field_misses,
            "mean_field_misses_per_case": (
                total_field_misses / case_count if case_count else 0.0
            ),
            "cases_with_zero_field_misses": miss_histogram[0],
            "cases_with_one_field_miss": miss_histogram[1],
            "cases_with_multiple_field_misses": sum(
                count for misses, count in miss_histogram.items() if misses >= 2
            ),
            "max_field_misses_in_a_case": max(miss_histogram, default=0),
            "field_miss_count_histogram": {
                str(misses): count
                for misses, count in sorted(miss_histogram.items())
            },
        },
        "pairwise_field_miss_diagnostics": _pairwise_miss_diagnostics(
            validated["per_case"]
        ),
        "decision_field_decoupling": _decision_field_decoupling(
            validated["per_case"]
        ),
        "confusion_losses": confusion_groups,
        "confusion_priority": confusion_priority,
        "required_dimension_coverage": dimension_coverage,
        "dimension_atlas": dimension_atlas,
        "dimension_measurement_blockers": dimension_blockers,
        "calibration": {
            "mean_brier": float(evaluation["raw"]["mean_confidence_brier"]),
            "score_loss": calibration_gap,
            "formula": "20 * max(0, 1 - 2 * mean_brier)",
        },
        "output_confidence_empirical_calibration_ceiling": (
            _empirical_calibration_ceiling(
                scored=scored,
                per_case=validated["per_case"],
                current_calibration_score=components["calibration"],
                current_total_score=components["total"],
            )
        ),
        "oracle_ceilings": {
            "perfect_extraction_only": components["total"] + extraction_gap,
            "perfect_classification_only": components["total"] + classification_gap,
            "perfect_calibration_only": components["total"] + calibration_gap,
            "perfect_extraction_and_classification": (
                components["total"] + extraction_gap + classification_gap
            ),
            "perfect_all_components": components["total_max"],
        },
        "validation": {
            "cross_artifact_checks": "passed",
            "score_version": SCORE_VERSION,
            "numeric_tolerance": FLOAT_TOLERANCE,
            "validated_dimensions": [
                "coverage_and_counts",
                "truth_prediction_and_case_score_adjudication",
                "confidence_and_brier",
                "per_field_match_status_points_and_maxima",
                "case_and_aggregate_raw_totals",
                "confusion_and_component_scores",
            ],
        },
        "limitations": [
            "All public labeled cases have already been evaluated; this is diagnostic evidence, not an unseen holdout.",
            "Oracle ceilings are additive or empirical in-sample bounds, not expected gains from a concrete implementation.",
            "Field corrections can change adjudication and confidence, so component effects are not causally independent.",
            "Public labels may omit private difficulty, damage-profile, trap, and unrecoverable-field metadata.",
            "Exact confidence values and low-support or sensitive modal outputs are suppressed.",
            "Trace dimensions remain blocked unless an exact-case current-source development capture is supplied.",
            "Whole-run runtime cannot allocate loss by cost bucket without per-case timings.",
            "The report contains aggregates only and must never be converted into runtime per-case rules.",
        ],
    }


def render_markdown(atlas: Mapping[str, Any]) -> str:
    scores = atlas["current_scores"]
    gaps = atlas["score_gaps"]
    miss_summary = atlas["field_miss_case_summary"]
    decoupling = atlas["decision_field_decoupling"]
    empirical_ceiling = atlas[
        "output_confidence_empirical_calibration_ceiling"
    ]
    lines = [
        "# WO-13 Full-Public Score-Loss Atlas",
        "",
        "This is aggregate diagnostic evidence over public labeled training data. "
        "It is not an unseen holdout, validation, private-test, or leaderboard score.",
        "",
        "## Evidence integrity",
        "",
        f"- Cross-artifact validation: `{atlas['validation']['cross_artifact_checks']}`",
        f"- Score version: `{atlas['validation']['score_version']}`",
        f"- Source-hash basis: `{atlas['source_hash_basis']}`",
    ]
    for name, digest in atlas["source_sha256"].items():
        lines.append(f"- `{name}` SHA-256: `{digest}`")
    for name, digest in atlas["dimension_source_sha256"].items():
        lines.append(f"- Dimension evidence `{name}` SHA-256: `{digest}`")
    lines.extend(
        [
            "",
            "## Score bridge",
            "",
            "| Component | Current | Gap to perfect |",
            "| --- | ---: | ---: |",
            f"| Extraction | {scores['extraction']:.6f} | {gaps['extraction']:.6f} |",
            f"| Classification | {scores['classification']:.6f} | {gaps['classification']:.6f} |",
            f"| Calibration | {scores['calibration']:.6f} | {gaps['calibration']:.6f} |",
            f"| **Total** | **{scores['total']:.6f}** | **{gaps['total_to_perfect']:.6f}** |",
            "",
            f"Target `{atlas['target_score']:.2f}` requires `+{gaps['gain_required_for_target']:.6f}` "
            f"points, or `{100 * gaps['remaining_error_recovery_required']:.2f}%` of all remaining error.",
            "",
            "## Extraction loss by field",
            "",
            "| Field | Missed | Match rate | Score loss | Default among misses | Modal wrong output |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in atlas["field_priority"]:
        details = atlas["field_losses"][item["field"]]
        mode = details["modal_wrong_output"]
        lines.append(
            f"| `{item['field']}` | {details['missed']} | "
            f"{100 * details['match_rate']:.2f}% | "
            f"{details['normalized_score_loss']:.6f} | "
            f"{details['default_on_missed_cases']}/{details['missed']} | "
            f"`{mode['value']}` ({mode['count']}) |"
        )
    lines.extend(
        [
            "",
            "## Field-miss concentration",
            "",
            f"- Total field misses: `{miss_summary['total_field_misses']}`",
            f"- Mean misses per case: `{miss_summary['mean_field_misses_per_case']:.6f}`",
            f"- Cases with no misses: `{miss_summary['cases_with_zero_field_misses']}`",
            f"- Cases with exactly one miss: `{miss_summary['cases_with_one_field_miss']}`",
            f"- Cases with multiple misses: `{miss_summary['cases_with_multiple_field_misses']}`",
            "",
            "### Pairwise miss co-occurrence",
            "",
            "| Field pair | Co-misses | Expected | Enrichment |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for pair in atlas["pairwise_field_miss_diagnostics"]:
        if pair["co_missed_cases"] <= 0:
            continue
        lift = pair["enrichment_lift"]
        lift_text = f"{lift:.3f}x" if lift is not None else "n/a"
        lines.append(
            f"| `{pair['field_a']}` + `{pair['field_b']}` | "
            f"{pair['co_missed_cases']} | "
            f"{pair['expected_co_misses_under_independence']:.3f} | "
            f"{lift_text} |"
        )
    lines.extend(
        [
            "",
            "## Wrong-decision / extraction decoupling",
            "",
            "| Decision outcome | All fields correct | One or more field misses |",
            "| --- | ---: | ---: |",
            f"| Correct | {decoupling['correct_decision_all_fields_correct']} | "
            f"{decoupling['correct_decision_one_or_more_field_misses']} |",
            f"| Wrong | {decoupling['wrong_decision_all_fields_correct']} | "
            f"{decoupling['wrong_decision_one_or_more_field_misses']} |",
            "",
            "## Classification and calibration loss",
            "",
            "| Truth → output | Cases | Class score loss | All fields correct | Mean Brier | Calibration loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, details in atlas["confusion_losses"].items():
        lines.append(
            f"| `{key}` | {details['cases']} | "
            f"{details['classification_score_loss']:.6f} | "
            f"{details['all_fields_correct_cases']} | "
            f"{details['mean_brier']:.6f} | "
            f"{details['calibration_score_loss_contribution']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Output + confidence empirical calibration ceiling",
            "",
            f"- Groups: `{empirical_ceiling['group_count']}` "
            f"(`{empirical_ceiling['grouping']}`)",
            f"- Empirical-oracle mean Brier: "
            f"`{empirical_ceiling['empirical_oracle_mean_brier']:.6f}`",
            f"- Empirical-oracle calibration: "
            f"`{empirical_ceiling['empirical_oracle_calibration_score']:.6f}/20`",
            f"- Calibration gain ceiling: "
            f"`+{empirical_ceiling['empirical_calibration_gain_ceiling']:.6f}`",
            f"- Total-score ceiling from this remapping alone: "
            f"`{empirical_ceiling['empirical_total_score_ceiling']:.6f}/150`",
            "",
            empirical_ceiling["interpretation"],
            "",
            "## Required dimension coverage",
            "",
            "| Dimension | Status |",
            "| --- | --- |",
        ]
    )
    for dimension in REQUIRED_ATLAS_DIMENSIONS:
        lines.append(
            f"| `{dimension}` | "
            f"`{atlas['required_dimension_coverage'][dimension]}` |"
        )
    lines.extend(
        [
            "",
            "Field and adjudication-confusion loss are quantified above. "
            "The tables below allocate loss across emitted K-safe groups. "
            "A dimension is only a complete reconciliation when no under-K "
            "residual was omitted.",
        ]
    )
    for dimension in (
        "page_template_family",
        *TRACE_DIMENSIONS,
        "confidence_bucket",
        "runtime_cost",
    ):
        details = atlas["dimension_atlas"][dimension]
        lines.extend(["", f"### {dimension.replace('_', ' ').title()}", ""])
        if "groups" in details:
            lines.append(f"- Status: `{details['status']}`")
            if details.get("qualification"):
                lines.append(f"- Qualification: {details['qualification']}")
            if details.get("complementary_suppression_applied"):
                lines.append(
                    "- Complementary suppression: an additional category was "
                    "coarsened into the K-safe suppression pool"
                )
            if details["emitted_case_count"] < details["case_count"]:
                lines.append(
                    "- Under-K residual: omitted from the aggregate output"
                )
            lines.append("")
            lines.extend(
                [
                    "| Category | Cases | Field misses | Wrong decisions | "
                    "Extraction loss | Classification loss | Calibration loss |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for group in details["groups"]:
                lines.append(
                    f"| `{group['category']}` | {group['case_count']} | "
                    f"{group['field_misses']} | "
                    f"{group['wrong_decision_cases']} | "
                    f"{group['extraction_score_loss']:.6f} | "
                    f"{group['classification_score_loss']:.6f} | "
                    f"{group['calibration_score_loss_contribution']:.6f} |"
                )
        else:
            lines.extend(
                [
                    f"- Status: `{details['status']}`",
                    f"- Reason: {details['reason']}",
                    f"- Required input: `{details['required_input']}`",
                ]
            )
    lines.extend(["", "### Remaining measurement blockers", ""])
    if atlas["dimension_measurement_blockers"]:
        for blocker in atlas["dimension_measurement_blockers"]:
            lines.append(
                f"- `{blocker['dimension']}` (`{blocker['status']}`): "
                f"{blocker['reason']}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Oracle ceilings", ""])
    for name, value in atlas["oracle_ceilings"].items():
        lines.append(f"- `{name}`: {value:.6f}/150")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in atlas["limitations"])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an aggregate-only MIB score-loss atlas."
    )
    parser.add_argument("--truth", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--case-scores", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--target-score", type=float, default=148.0)
    parser.add_argument(
        "--layout-manifest",
        help="label-blind exact-case layout manifest for page/template loss",
    )
    parser.add_argument(
        "--trace-dimensions",
        help=(
            "versioned development-only JSON trace evidence with exact-case "
            "allowlisted categories and per-case runtime"
        ),
    )
    args = parser.parse_args()

    truth_path = Path(args.truth)
    submission_path = Path(args.submission)
    evaluation_path = Path(args.evaluation)
    case_scores_path = Path(args.case_scores)
    layout_path = Path(args.layout_manifest) if args.layout_manifest else None
    trace_path = Path(args.trace_dimensions) if args.trace_dimensions else None
    atlas = build_atlas(
        truth_rows=_read_csv(truth_path),
        prediction_rows=_prediction_rows(submission_path),
        evaluation=_read_json(evaluation_path),
        case_scores=_read_jsonl(case_scores_path),
        target_score=args.target_score,
        source_sha256={
            "truth": _sha256_path(truth_path),
            "submission": _sha256_path(submission_path),
            "evaluation": _sha256_path(evaluation_path),
            "case_scores": _sha256_path(case_scores_path),
        },
        layout_manifest_path=layout_path,
        trace_dimensions_path=trace_path,
    )
    _write_json(Path(args.output_json), atlas)
    markdown_path = Path(args.output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(atlas), encoding="utf-8")
    print(
        f"score-loss atlas: current={atlas['current_scores']['total']:.6f} "
        f"target={atlas['target_score']:.2f} "
        f"required_recovery={100 * atlas['score_gaps']['remaining_error_recovery_required']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
