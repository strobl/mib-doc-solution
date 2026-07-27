"""Closed, identity-free context for final-output confidence calibration.

The decision pipeline owns this context.  A confidence calibrator may consume
it only after every recovery and policy stage has accepted its final result.
The contract intentionally contains no free-form strings, identifiers, raw
field values, filenames, hashes, or policy-reason suffixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import PredictionRow


FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION = 1
FINAL_CLASSES = ("APPROVED", "DENIED", "NEEDS_REVIEW")
FINAL_POLICY_ROUTES = (
    "binding_authority",
    "deterministic_policy",
    "revalidated_policy",
    "visible_policy_violation",
    "recovery_head",
)
FINAL_RECOVERY_ROUTES = ("primary", "late_visible", "rapid_visible")


def _unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return rendered


def _optional_unit_interval(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _unit_interval(value, name=name)


@dataclass(frozen=True)
class FinalConfidenceContext:
    """Strict semantic context for the already-accepted final decision."""

    schema_version: int
    final_class: str
    policy_route: str
    authoritative: bool
    visible_completeness: float
    has_conflict: bool
    ocr_disagreement: float
    recovery_route: str
    model_margin: float | None
    ensemble_agreement: float | None
    resolution_entropy: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or self.schema_version != FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported final-confidence context schema")
        if self.final_class not in FINAL_CLASSES:
            raise ValueError("final_class is outside the frozen class contract")
        if self.policy_route not in FINAL_POLICY_ROUTES:
            raise ValueError("policy_route is outside the frozen route contract")
        if not isinstance(self.authoritative, bool):
            raise TypeError("authoritative must be a boolean")
        if not isinstance(self.has_conflict, bool):
            raise TypeError("has_conflict must be a boolean")
        if self.recovery_route not in FINAL_RECOVERY_ROUTES:
            raise ValueError("recovery_route is outside the frozen route contract")
        object.__setattr__(
            self,
            "visible_completeness",
            _unit_interval(
                self.visible_completeness,
                name="visible_completeness",
            ),
        )
        object.__setattr__(
            self,
            "ocr_disagreement",
            _unit_interval(self.ocr_disagreement, name="ocr_disagreement"),
        )
        object.__setattr__(
            self,
            "model_margin",
            _optional_unit_interval(self.model_margin, name="model_margin"),
        )
        object.__setattr__(
            self,
            "ensemble_agreement",
            _optional_unit_interval(
                self.ensemble_agreement,
                name="ensemble_agreement",
            ),
        )
        object.__setattr__(
            self,
            "resolution_entropy",
            _unit_interval(
                self.resolution_entropy,
                name="resolution_entropy",
            ),
        )
        if self.authoritative != (
            self.policy_route == "binding_authority"
        ):
            raise ValueError(
                "authoritative and binding_authority route must be equivalent"
            )
        if (
            self.model_margin is None
        ) != (
            self.ensemble_agreement is None
        ):
            raise ValueError(
                "model diagnostics must be both present or both absent"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete fixed-key serialization contract."""

        return {
            "schema_version": self.schema_version,
            "final_class": self.final_class,
            "policy_route": self.policy_route,
            "authoritative": self.authoritative,
            "visible_completeness": self.visible_completeness,
            "has_conflict": self.has_conflict,
            "ocr_disagreement": self.ocr_disagreement,
            "recovery_route": self.recovery_route,
            "model_margin": self.model_margin,
            "ensemble_agreement": self.ensemble_agreement,
            "resolution_entropy": self.resolution_entropy,
        }


@dataclass(frozen=True)
class FinalPredictionWithConfidenceContext:
    """A final typed row bound to the context produced for that same row."""

    row: PredictionRow
    context: FinalConfidenceContext

    def __post_init__(self) -> None:
        if not isinstance(self.row, PredictionRow):
            raise TypeError("final-confidence input requires PredictionRow")
        if not isinstance(self.context, FinalConfidenceContext):
            raise TypeError(
                "final-confidence input requires FinalConfidenceContext"
            )
        if self.row.adjudication != self.context.final_class:
            raise ValueError(
                "final row adjudication does not match confidence context"
            )
