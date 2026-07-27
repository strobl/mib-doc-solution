"""Identity-free residual decision recovery with hard evidence gates.

The ordinary deterministic policy remains the primary decision maker.  This
module exposes a compact three-class model only for the residual
``NEEDS_REVIEW`` population and places the model behind evidence gates that it
cannot weaken:

1. clean, exactly scoped signed authority is binding;
2. an explicit clean visible policy violation is a denial;
3. an ordinary deterministic approval or denial is immutable;
4. only then may a model inspect an identity-free feature vector;
5. uncertainty, disagreement, or a failed class-specific guard returns review.

The feature contract is intentionally closed.  It contains only normalized
semantic states and aggregate evidence diagnostics.  No case/applicant value,
raw sponsor/date/free text, source path, hash, extraction order, or raw policy
reason suffix can enter a model artifact.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol, Sequence

from .adjudication import (
    AdjudicationOutcome,
    DecisionTrace,
    PolicyRuleSet,
)
from .extraction import CandidateEvidence, EvidenceType
from .fusion import candidate_has_complete_provenance
from .models import ADJUDICATION_VALUES, PredictionRow
from .resolution import (
    EvidencePrecedenceHierarchy,
    FieldState,
    ResolvedCase,
    ResolvedField,
)


MODEL_CLASSES = ("APPROVED", "DENIED", "NEEDS_REVIEW")
SCORED_EVIDENCE_FIELDS = (
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
RECOVERY_ROUTES = ("primary", "late_visible", "rapid_visible")

# Every value is a finite number in [0, 1].  Keeping both names and ranges
# frozen makes the artifact auditable and prevents a caller from smuggling an
# identifying string through a dynamically named feature.
FEATURE_NAMES = (
    "baseline_approved",
    "baseline_denied",
    "baseline_review",
    "authoritative_decision",
    "resolved_fraction",
    "visible_fraction",
    "exact_case_scope_fraction",
    "exact_subject_scope_fraction",
    "clean_fraction",
    "provenance_complete_fraction",
    "unknown_fraction",
    "contested_fraction",
    "link_confidence",
    "unresolved_linkage",
    "rescinded_decision",
    "packet_conflict",
    "packet_watermark",
    "binding_approval",
    "binding_denial",
    "binding_review",
    "rank_one_fraction",
    "rank_two_fraction",
    "rank_three_fraction",
    "rank_four_plus_fraction",
    "independent_agreement_strength",
    "evidence_disagreement",
    "provenance_strength",
    "multi_source_strength",
    "multi_page_strength",
    "fee_page_present",
    "attestation_page_present",
    "biometric_evidence_present",
    "manual_authority_present",
    "policy_explicit_violation",
    "policy_disqualifying_flag",
    "policy_embargo",
    "policy_transit",
    "policy_unpaid",
    "policy_stale",
    "policy_biohazard",
    "policy_stay_limit",
    "policy_barred_party",
    "policy_exception_denial",
    "policy_review_gap",
    "policy_review_conflict",
    "policy_review_visibility",
    "policy_review_waiver",
    "policy_review_other",
    "policy_strict_clear",
    "policy_support_strength",
    "route_primary",
    "route_late_visible",
    "route_rapid_visible",
)

_UNTRUSTED_CUE_TOKENS = (
    "watermark",
    "strikethrough",
    "struck",
    "strike",
    "crossed_out",
    "synthetic_default",
    "hidden",
    "white_text",
    "prompt_injection",
    "qr_prompt",
    "foreign_applicant",
    "decoy",
)
_GAP_REVIEW_PREFIXES = (
    "required_output_unknown:",
    "required_sponsor_unknown",
    "arrival_date_unknown",
    "fee_status_unknown",
    "risk_flags_unknown",
    "clean_biohazard_check_missing",
)
_VISIBILITY_REVIEW_PREFIXES = (
    "required_output_not_visible:",
    "required_sponsor_not_visible",
    "arrival_date_not_visible",
    "visa_class_not_visible",
    "risk_flags_not_visible",
)
_CONFLICT_REVIEW_PREFIXES = (
    "contested_field:",
    "unresolved_linkage:",
    "review_flag:",
    "conflicting_generalizable_exceptions",
)
_WAIVER_REVIEW_PREFIXES = (
    "unsupported_fee_waiver",
    "stale_diplomatic_note_missing",
)
_DENIAL_CATEGORY_PREFIXES = MappingProxyType(
    {
        "policy_disqualifying_flag": ("disqualifying_flag:",),
        "policy_embargo": ("embargoed_home_world:",),
        "policy_transit": ("transit_work_authorization",),
        "policy_unpaid": ("unpaid_without_valid_waiver",),
        "policy_stale": ("stale_application",),
        "policy_biohazard": ("biohazard_red",),
        "policy_stay_limit": ("stay_limit_exceeded:",),
        "policy_barred_party": ("barred_sponsor:",),
        "policy_exception_denial": ("validated_generalizable_exception",),
    }
)


class DeterministicAdjudicator(Protocol):
    """The normal policy seam consumed by the gated hybrid."""

    def adjudicate_case(self, resolved_case: ResolvedCase) -> AdjudicationOutcome:
        """Return one deterministic policy outcome."""


def _freeze_numeric_mapping(
    values: Mapping[str, float],
    *,
    expected_names: tuple[str, ...],
) -> Mapping[str, float]:
    copied = dict(values)
    if tuple(copied) != expected_names:
        raise ValueError("feature mapping must use the exact frozen order")
    checked: dict[str, float] = {}
    for name in expected_names:
        value = copied[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"feature {name!r} must be numeric")
        rendered = float(value)
        if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
            raise ValueError(f"feature {name!r} must be finite and in [0, 1]")
        checked[name] = rendered
    return MappingProxyType(checked)


@dataclass(frozen=True)
class IdentityFreeDecisionFeatures:
    """One immutable row in the frozen model feature space."""

    values: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _freeze_numeric_mapping(
                self.values,
                expected_names=FEATURE_NAMES,
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float],
    ) -> "IdentityFreeDecisionFeatures":
        return cls(values=dict(values))

    def as_vector(self) -> tuple[float, ...]:
        return tuple(self.values[name] for name in FEATURE_NAMES)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


def _finite_weight(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or abs(rendered) > 32.0:
        raise ValueError(f"{label} must be finite and within [-32, 32]")
    return rendered


@dataclass(frozen=True)
class CompactModelArtifact:
    """Strict, identity-free parameter artifact for a linear softmax member."""

    schema_version: int
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    intercepts: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported compact model schema")
        if tuple(self.class_names) != MODEL_CLASSES:
            raise ValueError("compact model classes do not match the contract")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("compact model features do not match the contract")
        if len(self.intercepts) != len(MODEL_CLASSES):
            raise ValueError("compact model requires one intercept per class")
        checked_intercepts = tuple(
            _finite_weight(value, label="intercept")
            for value in self.intercepts
        )
        if len(self.coefficients) != len(MODEL_CLASSES):
            raise ValueError("compact model requires one coefficient row per class")
        checked_coefficients: list[tuple[float, ...]] = []
        for row in self.coefficients:
            if len(row) != len(FEATURE_NAMES):
                raise ValueError(
                    "compact model coefficient rows must match FEATURE_NAMES"
                )
            checked_coefficients.append(
                tuple(
                    _finite_weight(value, label="coefficient")
                    for value in row
                )
            )
        object.__setattr__(self, "intercepts", checked_intercepts)
        object.__setattr__(
            self,
            "coefficients",
            tuple(checked_coefficients),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CompactModelArtifact":
        """Parse only the frozen parameter schema; unknown metadata is refused."""

        if set(value) != {
            "schema_version",
            "class_names",
            "feature_names",
            "intercepts",
            "coefficients",
        }:
            raise ValueError("compact model artifact has unsupported keys")
        class_names = value["class_names"]
        feature_names = value["feature_names"]
        intercepts = value["intercepts"]
        coefficients = value["coefficients"]
        if (
            not isinstance(class_names, (list, tuple))
            or not isinstance(feature_names, (list, tuple))
            or not isinstance(intercepts, (list, tuple))
            or not isinstance(coefficients, (list, tuple))
            or any(not isinstance(row, (list, tuple)) for row in coefficients)
        ):
            raise TypeError("compact model arrays must be lists or tuples")
        schema_version = value["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("compact model schema_version must be an integer")
        return cls(
            schema_version=schema_version,
            class_names=tuple(str(item) for item in class_names),
            feature_names=tuple(str(item) for item in feature_names),
            intercepts=tuple(intercepts),
            coefficients=tuple(tuple(row) for row in coefficients),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "class_names": list(self.class_names),
            "feature_names": list(self.feature_names),
            "intercepts": list(self.intercepts),
            "coefficients": [list(row) for row in self.coefficients],
        }


@dataclass(frozen=True)
class ModelPrediction:
    """Three-class probabilities plus fail-closed uncertainty diagnostics."""

    decision: str
    probabilities: Mapping[str, float]
    margin: float
    disagreement: float
    member_probabilities: tuple[Mapping[str, float], ...]

    def __post_init__(self) -> None:
        for label, diagnostic in (
            ("margin", self.margin),
            ("disagreement", self.disagreement),
        ):
            if (
                isinstance(diagnostic, bool)
                or not isinstance(diagnostic, (int, float))
                or not math.isfinite(float(diagnostic))
                or not 0.0 <= float(diagnostic) <= 1.0
            ):
                raise ValueError(
                    f"model {label} must be a finite number in [0, 1]"
                )
        object.__setattr__(self, "margin", float(self.margin))
        object.__setattr__(
            self,
            "disagreement",
            float(self.disagreement),
        )
        probabilities = dict(self.probabilities)
        if tuple(probabilities) != MODEL_CLASSES:
            raise ValueError("model probabilities must use the frozen class order")
        checked: dict[str, float] = {}
        for class_name in MODEL_CLASSES:
            value = probabilities[class_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("model probabilities must be numeric")
            rendered = float(value)
            if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
                raise ValueError("model probabilities must be finite and normalized")
            checked[class_name] = rendered
        if not math.isclose(sum(checked.values()), 1.0, abs_tol=1e-12):
            raise ValueError("model probabilities must sum to one")
        members: list[Mapping[str, float]] = []
        if not 1 <= len(self.member_probabilities) <= 8:
            raise ValueError("prediction must expose one to eight model members")
        for member in self.member_probabilities:
            copied_member = dict(member)
            if tuple(copied_member) != MODEL_CLASSES:
                raise ValueError("member probabilities use an invalid class order")
            checked_member: dict[str, float] = {}
            for class_name in MODEL_CLASSES:
                member_value = copied_member[class_name]
                if (
                    isinstance(member_value, bool)
                    or not isinstance(member_value, (int, float))
                    or not math.isfinite(float(member_value))
                    or not 0.0 <= float(member_value) <= 1.0
                ):
                    raise ValueError("member probabilities must be finite in [0, 1]")
                checked_member[class_name] = float(member_value)
            if not math.isclose(
                sum(checked_member.values()),
                1.0,
                abs_tol=1e-12,
            ):
                raise ValueError("member probabilities must sum to one")
            members.append(MappingProxyType(checked_member))

        expected_average = {
            class_name: math.fsum(member[class_name] for member in members)
            / len(members)
            for class_name in MODEL_CLASSES
        }
        if any(
            not math.isclose(
                checked[class_name],
                expected_average[class_name],
                abs_tol=1e-12,
            )
            for class_name in MODEL_CLASSES
        ):
            raise ValueError("aggregate probabilities do not match model members")
        expected_decision = _top_decision(checked)
        ordered = sorted(checked.values(), reverse=True)
        expected_margin = ordered[0] - ordered[1]
        expected_disagreement = max(
            (
                max(member[class_name] for member in members)
                - min(member[class_name] for member in members)
                for class_name in MODEL_CLASSES
            ),
            default=0.0,
        )
        if self.decision != expected_decision:
            raise ValueError("model decision must equal the deterministic argmax")
        if not math.isclose(self.margin, expected_margin, abs_tol=1e-12):
            raise ValueError("model margin does not match the probabilities")
        if not math.isclose(
            self.disagreement,
            expected_disagreement,
            abs_tol=1e-12,
        ):
            raise ValueError("model disagreement does not match its members")
        object.__setattr__(self, "probabilities", MappingProxyType(checked))
        object.__setattr__(self, "member_probabilities", tuple(members))


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    maximum = max(logits)
    exponentials = tuple(math.exp(value - maximum) for value in logits)
    denominator = sum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("compact model produced invalid logits")
    return tuple(value / denominator for value in exponentials)


def _top_decision(probabilities: Mapping[str, float]) -> str:
    # A numerical tie abstains to review.  A denial wins the remaining tie
    # over approval, which is the safer deterministic ordering.
    priority = {"APPROVED": 0, "DENIED": 1, "NEEDS_REVIEW": 2}
    return max(
        MODEL_CLASSES,
        key=lambda name: (probabilities[name], priority[name]),
    )


class CompactThreeClassModel:
    """Small deterministic softmax ensemble over the frozen feature vector."""

    def __init__(self, members: Iterable[CompactModelArtifact]) -> None:
        checked = tuple(members)
        if not 1 <= len(checked) <= 8:
            raise ValueError("compact model requires between one and eight members")
        self._members = checked

    @classmethod
    def from_artifact(
        cls,
        artifact: CompactModelArtifact | Mapping[str, object],
    ) -> "CompactThreeClassModel":
        parsed = (
            artifact
            if isinstance(artifact, CompactModelArtifact)
            else CompactModelArtifact.from_mapping(artifact)
        )
        return cls((parsed,))

    @classmethod
    def from_artifacts(
        cls,
        artifacts: Iterable[CompactModelArtifact | Mapping[str, object]],
    ) -> "CompactThreeClassModel":
        return cls(
            tuple(
                artifact
                if isinstance(artifact, CompactModelArtifact)
                else CompactModelArtifact.from_mapping(artifact)
                for artifact in artifacts
            )
        )

    @classmethod
    def fit(
        cls,
        feature_rows: Sequence[IdentityFreeDecisionFeatures],
        labels: Sequence[str],
        *,
        smoothing: float = 1.0,
    ) -> "CompactThreeClassModel":
        """Fit an order-invariant, smoothed nearest-centroid linear head.

        The outer grouped-CV runner owns all splitting.  This method consumes
        only already-built frozen feature rows and class labels, so there is no
        identity-bearing sample object to memorize.  Smoothed Bernoulli
        class centroids produce a compact linear softmax artifact without an
        optional ML runtime dependency.
        """

        if len(feature_rows) != len(labels) or not feature_rows:
            raise ValueError("fit requires equally sized non-empty rows and labels")
        if (
            isinstance(smoothing, bool)
            or not isinstance(smoothing, (int, float))
            or not math.isfinite(float(smoothing))
            or float(smoothing) <= 0.0
        ):
            raise ValueError("smoothing must be a finite positive number")
        rendered_smoothing = float(smoothing)
        samples = tuple(
            sorted(
                (
                    (row.as_vector(), str(label))
                    for row, label in zip(feature_rows, labels)
                ),
                key=lambda item: (item[1], item[0]),
            )
        )
        if any(label not in MODEL_CLASSES for _row, label in samples):
            raise ValueError("fit labels must use the frozen three classes")

        sample_count = len(samples)
        intercepts: list[float] = []
        coefficient_rows: list[tuple[float, ...]] = []
        for class_name in MODEL_CLASSES:
            class_vectors = tuple(
                vector for vector, label in samples if label == class_name
            )
            class_count = len(class_vectors)
            prior = (
                class_count + rendered_smoothing
            ) / (
                sample_count
                + rendered_smoothing * len(MODEL_CLASSES)
            )
            feature_means = tuple(
                (
                    math.fsum(vector[index] for vector in class_vectors)
                    + rendered_smoothing
                )
                / (class_count + 2.0 * rendered_smoothing)
                for index in range(len(FEATURE_NAMES))
            )
            coefficients = tuple(2.0 * mean - 1.0 for mean in feature_means)
            intercept = math.log(prior) - 0.5 * math.fsum(
                coefficient * coefficient for coefficient in coefficients
            )
            intercepts.append(intercept)
            coefficient_rows.append(coefficients)
        artifact = CompactModelArtifact(
            schema_version=1,
            class_names=MODEL_CLASSES,
            feature_names=FEATURE_NAMES,
            intercepts=tuple(intercepts),
            coefficients=tuple(coefficient_rows),
        )
        return cls((artifact,))

    @property
    def members(self) -> tuple[CompactModelArtifact, ...]:
        return self._members

    def predict(
        self,
        features: IdentityFreeDecisionFeatures,
    ) -> ModelPrediction:
        vector = features.as_vector()
        member_probabilities: list[tuple[float, ...]] = []
        for artifact in self._members:
            logits = tuple(
                artifact.intercepts[class_index]
                + sum(
                    coefficient * feature
                    for coefficient, feature in zip(
                        artifact.coefficients[class_index],
                        vector,
                    )
                )
                for class_index in range(len(MODEL_CLASSES))
            )
            member_probabilities.append(_softmax(logits))

        averaged_values = tuple(
            sum(member[class_index] for member in member_probabilities)
            / len(member_probabilities)
            for class_index in range(len(MODEL_CLASSES))
        )
        # Normalize once more to eliminate summation drift before enforcing the
        # exact probability contract.
        total = sum(averaged_values)
        averaged_values = tuple(value / total for value in averaged_values)
        averaged = {
            name: averaged_values[index]
            for index, name in enumerate(MODEL_CLASSES)
        }
        decision = _top_decision(averaged)
        ordered = sorted(averaged.values(), reverse=True)
        margin = ordered[0] - ordered[1]
        disagreement = max(
            (
                max(member[class_index] for member in member_probabilities)
                - min(member[class_index] for member in member_probabilities)
                for class_index in range(len(MODEL_CLASSES))
            ),
            default=0.0,
        )
        return ModelPrediction(
            decision=decision,
            probabilities=averaged,
            margin=min(1.0, max(0.0, margin)),
            disagreement=min(1.0, max(0.0, disagreement)),
            member_probabilities=tuple(
                {
                    name: member[index]
                    for index, name in enumerate(MODEL_CLASSES)
                }
                for member in member_probabilities
            ),
        )


class EvidenceCompletionOnlyModel:
    """Research comparator using no labels and no semantic raw values."""

    @staticmethod
    def predict(
        features: IdentityFreeDecisionFeatures,
    ) -> ModelPrediction:
        values = features.values
        if values["baseline_approved"] == 1.0:
            probabilities = {
                "APPROVED": 0.98,
                "DENIED": 0.01,
                "NEEDS_REVIEW": 0.01,
            }
        elif values["baseline_denied"] == 1.0:
            probabilities = {
                "APPROVED": 0.01,
                "DENIED": 0.98,
                "NEEDS_REVIEW": 0.01,
            }
        elif (
            values["clean_fraction"] == 1.0
            and values["exact_case_scope_fraction"] == 1.0
            and values["exact_subject_scope_fraction"] == 1.0
            and values["packet_conflict"] == 0.0
            and values["packet_watermark"] == 0.0
            and values["unresolved_linkage"] == 0.0
            and values["policy_explicit_violation"] == 0.0
        ):
            probabilities = {
                "APPROVED": 0.90,
                "DENIED": 0.01,
                "NEEDS_REVIEW": 0.09,
            }
        else:
            probabilities = {
                "APPROVED": 0.05,
                "DENIED": 0.05,
                "NEEDS_REVIEW": 0.90,
            }
        decision = _top_decision(probabilities)
        ordered = sorted(probabilities.values(), reverse=True)
        return ModelPrediction(
            decision=decision,
            probabilities=probabilities,
            margin=ordered[0] - ordered[1],
            disagreement=0.0,
            member_probabilities=(probabilities,),
        )


@dataclass(frozen=True)
class FeatureLevelHybridDecision:
    """Pure feature-level result used by grouped out-of-fold comparison."""

    decision: str
    applied: bool
    route: str
    veto_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.decision not in MODEL_CLASSES:
            raise ValueError("feature-level hybrid returned an invalid decision")


class GatedHybridDecisionRule:
    """The same fail-closed class gate without a case object or row mutation."""

    def __init__(
        self,
        *,
        approval_threshold: float = 0.85,
        denial_threshold: float = 0.90,
        margin_threshold: float = 0.20,
        maximum_disagreement: float = 0.08,
    ) -> None:
        checked: dict[str, float] = {}
        for label, value in (
            ("approval_threshold", approval_threshold),
            ("denial_threshold", denial_threshold),
            ("margin_threshold", margin_threshold),
            ("maximum_disagreement", maximum_disagreement),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{label} must be finite and in [0, 1]")
            checked[label] = float(value)
        self._approval_threshold = checked["approval_threshold"]
        self._denial_threshold = checked["denial_threshold"]
        self._margin_threshold = checked["margin_threshold"]
        self._maximum_disagreement = checked["maximum_disagreement"]

    def decide(
        self,
        baseline_decision: str,
        features: IdentityFreeDecisionFeatures,
        prediction: ModelPrediction,
    ) -> FeatureLevelHybridDecision:
        if baseline_decision not in MODEL_CLASSES:
            raise ValueError("baseline decision is outside the three-class contract")
        values = features.values
        binding_flags = {
            "APPROVED": values["binding_approval"],
            "DENIED": values["binding_denial"],
            "NEEDS_REVIEW": values["binding_review"],
        }
        active_binding = tuple(
            decision
            for decision, active in binding_flags.items()
            if active == 1.0
        )
        if len(active_binding) > 1:
            return FeatureLevelHybridDecision(
                decision="NEEDS_REVIEW",
                applied=baseline_decision != "NEEDS_REVIEW",
                route="invalid_binding_review",
                veto_reasons=("conflicting_binding_features",),
            )
        if active_binding:
            decision = active_binding[0]
            return FeatureLevelHybridDecision(
                decision=decision,
                applied=decision != baseline_decision,
                route="binding_authority",
            )
        if values["authoritative_decision"] == 1.0:
            return FeatureLevelHybridDecision(
                decision=baseline_decision,
                applied=False,
                route="validated_authoritative_policy",
            )
        if values["policy_explicit_violation"] == 1.0:
            return FeatureLevelHybridDecision(
                decision="DENIED",
                applied=baseline_decision != "DENIED",
                route="visible_policy_violation",
            )
        if baseline_decision != "NEEDS_REVIEW":
            return FeatureLevelHybridDecision(
                decision=baseline_decision,
                applied=False,
                route="deterministic_policy",
            )

        uncertainty_vetoes = []
        if prediction.margin < self._margin_threshold:
            uncertainty_vetoes.append("insufficient_margin")
        if prediction.disagreement > self._maximum_disagreement:
            uncertainty_vetoes.append("ensemble_disagreement")
        if prediction.decision == "NEEDS_REVIEW":
            uncertainty_vetoes.append("model_abstained")
        if uncertainty_vetoes:
            return FeatureLevelHybridDecision(
                decision="NEEDS_REVIEW",
                applied=False,
                route="uncertainty_review",
                veto_reasons=tuple(sorted(uncertainty_vetoes)),
            )

        if prediction.decision == "APPROVED":
            vetoes: list[str] = []
            for name in (
                "resolved_fraction",
                "visible_fraction",
                "exact_case_scope_fraction",
                "exact_subject_scope_fraction",
                "clean_fraction",
            ):
                if values[name] != 1.0:
                    vetoes.append(f"incomplete_{name}")
            for name in (
                "unresolved_linkage",
                "rescinded_decision",
                "packet_conflict",
                "packet_watermark",
                "policy_explicit_violation",
                "policy_disqualifying_flag",
                "policy_embargo",
                "policy_transit",
                "policy_unpaid",
                "policy_stale",
                "policy_biohazard",
                "policy_stay_limit",
                "policy_barred_party",
                "policy_exception_denial",
                "policy_review_gap",
                "policy_review_conflict",
                "policy_review_visibility",
                "policy_review_waiver",
                "policy_review_other",
            ):
                if values[name] != 0.0:
                    vetoes.append(
                        {
                            "packet_conflict": "evidence_conflict",
                            "packet_watermark": "untrusted_visual_content",
                        }.get(name, name)
                    )
            if values["link_confidence"] != 1.0:
                vetoes.append("unresolved_subject_scope")
            if prediction.probabilities["APPROVED"] < self._approval_threshold:
                vetoes.append("approval_probability_below_threshold")
            if vetoes:
                return FeatureLevelHybridDecision(
                    decision="NEEDS_REVIEW",
                    applied=False,
                    route="approval_guard_review",
                    veto_reasons=tuple(sorted(set(vetoes))),
                )
            return FeatureLevelHybridDecision(
                decision="APPROVED",
                applied=True,
                route="gated_model_approval",
            )

        vetoes = ["denial_requires_visible_violation"]
        if prediction.probabilities["DENIED"] < self._denial_threshold:
            vetoes.append("denial_probability_below_threshold")
        return FeatureLevelHybridDecision(
            decision="NEEDS_REVIEW",
            applied=False,
            route="denial_guard_review",
            veto_reasons=tuple(sorted(vetoes)),
        )


def _has_untrusted_cue(candidate: CandidateEvidence) -> bool:
    cues = tuple(
        re.sub(r"[^a-z0-9]+", "_", str(cue).strip().casefold()).strip("_")
        for cue in candidate.visual_cues
    )
    return any(
        token in cue
        for cue in cues
        for token in _UNTRUSTED_CUE_TOKENS
    )


def _clean_visible_candidate(candidate: CandidateEvidence) -> bool:
    return bool(
        candidate.value is not None
        and candidate.legible
        and not candidate.superseded
        and candidate.source == "visible_ocr"
        and candidate.evidence_type is not EvidenceType.TEXT_LAYER
        and not _has_untrusted_cue(candidate)
    )


def _is_substantive_value(field_name: str, value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.strip().split()).casefold()
    placeholder_key = re.sub(r"[^a-z0-9]+", "", normalized)
    if placeholder_key in {
        "",
        "na",
        "nil",
        "null",
        "unknown",
        "tbd",
        "pending",
        "missing",
        "unspecified",
        "notapplicable",
        "notavailable",
        "notprovided",
        "nodata",
        "placeholder",
    }:
        return False
    if field_name == "sponsor_id" and normalized == "spn-0000":
        return False
    if field_name in {"arrival_date", "packet_receipt_date"} and (
        normalized == "1900-01-01"
    ):
        return False
    if field_name in SCORED_EVIDENCE_FIELDS:
        if normalized == "other":
            return False
        if normalized == "none" and field_name != "risk_flags":
            return False
    return True


def _exact_scope(
    candidate: CandidateEvidence,
    resolved_case: ResolvedCase,
) -> bool:
    return bool(
        resolved_case.active_applicant is not None
        and candidate.case_id_hint == resolved_case.case_id
        and candidate.applicant_hint == resolved_case.active_applicant
    )


def _clean_scoped_field(
    resolved_case: ResolvedCase,
    field_name: str,
) -> ResolvedField | None:
    field = resolved_case.fields.get(field_name)
    winner = field.winning_evidence if field is not None else None
    if (
        field is None
        or field.state is not FieldState.RESOLVED
        or not _is_substantive_value(field_name, field.value)
        or winner is None
        or not _clean_visible_candidate(winner)
        or not _exact_scope(winner, resolved_case)
    ):
        return None
    return field


def _parse_flags(value: str | None) -> frozenset[str]:
    if not isinstance(value, str) or value.strip().casefold() in {"", "none"}:
        return frozenset()
    return frozenset(
        item.strip().casefold().replace(" ", "_")
        for item in value.split("|")
        if item.strip()
    )


def _parse_date(value: str | None) -> date | None:
    if value in {None, "1900-01-01"}:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _parse_positive_int(value: str | None) -> int | None:
    if not isinstance(value, str) or not value.isdigit():
        return None
    rendered = int(value)
    return rendered if rendered > 0 else None


def _clean_value(
    resolved_case: ResolvedCase,
    field_name: str,
) -> str | None:
    field = _clean_scoped_field(resolved_case, field_name)
    return field.value if field is not None else None


def _visible_marker(
    resolved_case: ResolvedCase,
    field_name: str,
    expected_value: str,
) -> bool:
    return _clean_value(resolved_case, field_name) == expected_value


def _binding_decision(resolved_case: ResolvedCase) -> str | None:
    field = _clean_scoped_field(resolved_case, "adjudication")
    winner = field.winning_evidence if field is not None else None
    if (
        field is not None
        and field.value in ADJUDICATION_VALUES
        and winner is not None
        and winner.evidence_type
        in {EvidenceType.ADJUDICATOR_STAMP, EvidenceType.SIGNED_MANUAL_NOTE}
    ):
        return field.value
    return None


def _baseline_authority_supported(
    resolved_case: ResolvedCase,
    baseline: AdjudicationOutcome,
) -> bool:
    """Verify the looser scope contract used by the normal policy engine."""

    field = resolved_case.fields.get("adjudication")
    winner = field.winning_evidence if field is not None else None
    return bool(
        baseline.trace.authoritative_source
        and baseline.row.adjudication == baseline.trace.decision
        and baseline.row.adjudication in MODEL_CLASSES
        and field is not None
        and field.state is FieldState.RESOLVED
        and field.value == baseline.row.adjudication
        and winner is not None
        and _clean_visible_candidate(winner)
        and winner.evidence_type
        in {EvidenceType.ADJUDICATOR_STAMP, EvidenceType.SIGNED_MANUAL_NOTE}
        and winner.case_id_hint in {None, resolved_case.case_id}
        and winner.applicant_hint in {None, resolved_case.active_applicant}
    )


def _explicit_visible_violation_categories(
    resolved_case: ResolvedCase,
    rules: PolicyRuleSet,
) -> frozenset[str]:
    """Derive only high-precision violations from clean, exactly scoped facts."""

    categories: set[str] = set()
    flags = _parse_flags(_clean_value(resolved_case, "risk_flags"))
    if flags & rules.disqualifying_flags:
        categories.add("policy_disqualifying_flag")

    home_world = _clean_value(resolved_case, "home_world")
    visa_class = _clean_value(resolved_case, "visa_class")
    if home_world in rules.embargoed_worlds or (
        home_world in rules.non_diplomatic_embargoed_worlds
        and visa_class is not None
        and visa_class != "DIP-1"
    ):
        categories.add("policy_embargo")
    if visa_class == "TRANSIT-7":
        categories.add("policy_transit")

    fee_status = _clean_value(resolved_case, "fee_status")
    hardship_waiver = _visible_marker(
        resolved_case,
        "hardship_waiver",
        "valid",
    )
    if fee_status == "unpaid" and not hardship_waiver:
        categories.add("policy_unpaid")

    sponsor = _clean_value(resolved_case, "sponsor_id")
    if (
        visa_class is not None
        and visa_class != "DIP-1"
        and sponsor in rules.barred_sponsors
    ):
        categories.add("policy_barred_party")

    arrival = _parse_date(_clean_value(resolved_case, "arrival_date"))
    receipt = _parse_date(_clean_value(resolved_case, "packet_receipt_date"))
    if (
        arrival is not None
        and visa_class is not None
        and visa_class != "DIP-1"
        and ((receipt or rules.snapshot_receipt_date) - arrival).days
        > rules.stale_after_days
    ):
        categories.add("policy_stale")

    if _clean_value(resolved_case, "biohazard_check") == "red":
        categories.add("policy_biohazard")

    duration = _parse_positive_int(
        _clean_value(resolved_case, "stay_duration_days")
    )
    if (
        duration is not None
        and visa_class in rules.stay_limits
        and duration > rules.stay_limits[visa_class]
    ):
        categories.add("policy_stay_limit")
    return frozenset(categories)


def _fraction(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, float(numerator) / float(denominator)))


class IdentityFreeFeatureBuilder:
    """Build the closed feature row without serializing semantic raw values."""

    def __init__(self, *, rules: PolicyRuleSet | None = None) -> None:
        self._rules = rules or PolicyRuleSet()

    def build(
        self,
        resolved_case: ResolvedCase,
        baseline: AdjudicationOutcome,
        *,
        recovery_route: str = "primary",
    ) -> IdentityFreeDecisionFeatures:
        if recovery_route not in RECOVERY_ROUTES:
            raise ValueError("unsupported recovery route")
        if (
            baseline.row.adjudication not in MODEL_CLASSES
            or baseline.trace.decision not in MODEL_CLASSES
        ):
            raise ValueError("baseline decision is outside the three-class contract")

        fields = tuple(
            resolved_case.fields.get(field_name)
            for field_name in SCORED_EVIDENCE_FIELDS
        )
        winners = tuple(
            field.winning_evidence
            if field is not None and field.state is FieldState.RESOLVED
            else None
            for field in fields
        )
        resolved_count = sum(
            field is not None
            and field.state is FieldState.RESOLVED
            and _is_substantive_value(field.field_name, field.value)
            for field in fields
        )
        unknown_count = sum(
            field is None or field.state is FieldState.UNKNOWN
            for field in fields
        )
        contested_count = sum(
            field is not None and field.state is FieldState.CONTESTED
            for field in fields
        )
        visible_count = sum(
            field is not None
            and winner is not None
            and _is_substantive_value(field.field_name, field.value)
            and _clean_visible_candidate(winner)
            for field, winner in zip(fields, winners)
        )
        case_scope_count = sum(
            field is not None
            and winner is not None
            and _is_substantive_value(field.field_name, field.value)
            and winner.case_id_hint == resolved_case.case_id
            for field, winner in zip(fields, winners)
        )
        subject_scope_count = sum(
            field is not None
            and winner is not None
            and _is_substantive_value(field.field_name, field.value)
            and resolved_case.active_applicant is not None
            and winner.applicant_hint == resolved_case.active_applicant
            for field, winner in zip(fields, winners)
        )
        clean_count = sum(
            field is not None
            and winner is not None
            and _is_substantive_value(field.field_name, field.value)
            and _clean_visible_candidate(winner)
            and _exact_scope(winner, resolved_case)
            for field, winner in zip(fields, winners)
        )
        provenance_count = sum(
            winner is not None
            and candidate_has_complete_provenance(
                winner,
                expected_case_id=resolved_case.case_id,
                active_applicant=resolved_case.active_applicant,
            )
            for winner in winners
        )

        rank_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for winner in winners:
            if winner is None:
                continue
            rank = EvidencePrecedenceHierarchy.rank(winner.evidence_type)
            rank_counts[min(rank, 4)] += 1

        traces = tuple(
            field.fusion_trace
            for field in fields
            if field is not None and field.fusion_trace is not None
        )
        agreement_strength = (
            sum(
                _fraction(
                    trace.independent_agreement_count,
                    trace.independent_evidence_count,
                )
                for trace in traces
            )
            / len(traces)
            if traces
            else 0.0
        )
        disagreement = (
            sum(trace.disagreement_ratio for trace in traces) / len(traces)
            if traces
            else 0.0
        )
        provenance_strength = (
            sum(trace.provenance_completeness for trace in traces)
            / len(traces)
            if traces
            else 0.0
        )
        multi_source_strength = (
            sum(
                min(1.0, trace.independent_evidence_type_count / 2.0)
                for trace in traces
            )
            / len(traces)
            if traces
            else 0.0
        )
        multi_page_strength = (
            sum(
                min(1.0, trace.independent_page_count / 2.0)
                for trace in traces
            )
            / len(traces)
            if traces
            else 0.0
        )

        all_considered = tuple(
            candidate
            for field in resolved_case.fields.values()
            for candidate in field.considered
        )
        packet_watermark = any(
            _has_untrusted_cue(candidate)
            or candidate.source != "visible_ocr"
            or candidate.evidence_type is EvidenceType.TEXT_LAYER
            for candidate in all_considered
        )
        packet_conflict = bool(
            resolved_case.contested_fields
            or any(
                trace.disagreement_count > 0
                or trace.safety_count("same_rank_conflict_count") > 0
                for trace in traces
            )
        )
        binding = _binding_decision(resolved_case)
        explicit_categories = _explicit_visible_violation_categories(
            resolved_case,
            self._rules,
        )

        denial_reasons = tuple(str(item) for item in baseline.trace.denial_reasons)
        review_reasons = tuple(str(item) for item in baseline.trace.review_reasons)
        approval_facts = tuple(str(item) for item in baseline.trace.approval_facts)
        trace_category_values = {
            feature_name: float(
                any(
                    reason.startswith(prefix)
                    for reason in denial_reasons
                    for prefix in prefixes
                )
                or feature_name in explicit_categories
            )
            for feature_name, prefixes in _DENIAL_CATEGORY_PREFIXES.items()
        }
        review_gap = any(
            reason.startswith(prefix)
            for reason in review_reasons
            for prefix in _GAP_REVIEW_PREFIXES
        )
        review_conflict = any(
            reason.startswith(prefix)
            for reason in review_reasons
            for prefix in _CONFLICT_REVIEW_PREFIXES
        )
        review_visibility = any(
            reason.startswith(prefix)
            for reason in review_reasons
            for prefix in _VISIBILITY_REVIEW_PREFIXES
        )
        review_waiver = any(
            reason.startswith(prefix)
            for reason in review_reasons
            for prefix in _WAIVER_REVIEW_PREFIXES
        )
        categorized_review = (
            review_gap or review_conflict or review_visibility or review_waiver
        )
        unsupported_authority = bool(
            baseline.trace.authoritative_source
            and not _baseline_authority_supported(resolved_case, baseline)
        )
        manual_authority = any(
            candidate.evidence_type
            in {EvidenceType.ADJUDICATOR_STAMP, EvidenceType.SIGNED_MANUAL_NOTE}
            and _clean_visible_candidate(candidate)
            and _exact_scope(candidate, resolved_case)
            for candidate in all_considered
        )
        biometric_present = any(
            candidate.evidence_type is EvidenceType.BIOMETRIC_SLIP
            and _clean_visible_candidate(candidate)
            and _exact_scope(candidate, resolved_case)
            for candidate in all_considered
        )

        values = {
            "baseline_approved": float(
                baseline.row.adjudication == "APPROVED"
                and baseline.trace.decision == "APPROVED"
            ),
            "baseline_denied": float(
                baseline.row.adjudication == "DENIED"
                and baseline.trace.decision == "DENIED"
            ),
            "baseline_review": float(
                baseline.row.adjudication == "NEEDS_REVIEW"
                and baseline.trace.decision == "NEEDS_REVIEW"
            ),
            "authoritative_decision": float(
                _baseline_authority_supported(resolved_case, baseline)
            ),
            "resolved_fraction": _fraction(
                resolved_count,
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "visible_fraction": _fraction(
                visible_count,
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "exact_case_scope_fraction": _fraction(
                case_scope_count,
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "exact_subject_scope_fraction": _fraction(
                subject_scope_count,
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "clean_fraction": _fraction(
                clean_count,
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "provenance_complete_fraction": _fraction(
                provenance_count,
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "unknown_fraction": _fraction(
                unknown_count,
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "contested_fraction": _fraction(
                contested_count,
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "link_confidence": float(
                resolved_case.active_applicant is not None
                and not resolved_case.unresolved_linkage
                and not resolved_case.unresolved_reasons
            ),
            "unresolved_linkage": float(
                resolved_case.unresolved_linkage
                or bool(resolved_case.unresolved_reasons)
            ),
            "rescinded_decision": float(resolved_case.rescinded_decision),
            "packet_conflict": float(packet_conflict),
            "packet_watermark": float(packet_watermark),
            "binding_approval": float(binding == "APPROVED"),
            "binding_denial": float(binding == "DENIED"),
            "binding_review": float(binding == "NEEDS_REVIEW"),
            "rank_one_fraction": _fraction(
                rank_counts[1],
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "rank_two_fraction": _fraction(
                rank_counts[2],
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "rank_three_fraction": _fraction(
                rank_counts[3],
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "rank_four_plus_fraction": _fraction(
                rank_counts[4],
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "independent_agreement_strength": agreement_strength,
            "evidence_disagreement": disagreement,
            "provenance_strength": provenance_strength,
            "multi_source_strength": multi_source_strength,
            "multi_page_strength": multi_page_strength,
            "fee_page_present": float(
                _visible_marker(
                    resolved_case,
                    "page_type_present_fee_receipt",
                    "present",
                )
            ),
            "attestation_page_present": float(
                _visible_marker(
                    resolved_case,
                    "page_type_present_sponsor_attestation",
                    "present",
                )
            ),
            "biometric_evidence_present": float(biometric_present),
            "manual_authority_present": float(manual_authority),
            "policy_explicit_violation": float(bool(explicit_categories)),
            **trace_category_values,
            "policy_review_gap": float(review_gap),
            "policy_review_conflict": float(review_conflict),
            "policy_review_visibility": float(review_visibility),
            "policy_review_waiver": float(review_waiver),
            "policy_review_other": float(
                (bool(review_reasons) and not categorized_review)
                or bool(denial_reasons)
                or bool(baseline.trace.exception_ids)
                or unsupported_authority
            ),
            "policy_strict_clear": float(
                "strict_approval_bar_cleared" in approval_facts
            ),
            "policy_support_strength": _fraction(
                len(
                    {
                        fact.split(":", 1)[0]
                        for fact in approval_facts
                        if fact != "strict_approval_bar_cleared"
                    }
                ),
                len(SCORED_EVIDENCE_FIELDS),
            ),
            "route_primary": float(recovery_route == "primary"),
            "route_late_visible": float(recovery_route == "late_visible"),
            "route_rapid_visible": float(recovery_route == "rapid_visible"),
        }
        return IdentityFreeDecisionFeatures.from_mapping(values)


@dataclass(frozen=True)
class HybridDecision:
    """Auditable result of the hard-order gate and optional residual model."""

    outcome: AdjudicationOutcome
    route: str
    applied: bool
    model_prediction: ModelPrediction | None = None
    veto_reasons: tuple[str, ...] = ()


def _replacement_outcome(
    baseline: AdjudicationOutcome,
    decision: str,
    *,
    authoritative: bool,
    reason: str,
) -> AdjudicationOutcome:
    """Change only adjudication; WO-19 owns any confidence recalibration."""

    row = replace(baseline.row, adjudication=decision)
    trace = DecisionTrace(
        decision=decision,
        authoritative_source=authoritative,
        denial_reasons=(reason,) if decision == "DENIED" else (),
        review_reasons=(reason,) if decision == "NEEDS_REVIEW" else (),
        approval_facts=(reason,) if decision == "APPROVED" else (),
        exception_ids=baseline.trace.exception_ids,
    )
    return AdjudicationOutcome(row=row, trace=trace)


class GatedHybridDecisionRecoveryAdjudicator:
    """Apply a promoted candidate only after immutable deterministic policy."""

    def __init__(
        self,
        baseline: DeterministicAdjudicator,
        model: CompactThreeClassModel | EvidenceCompletionOnlyModel,
        *,
        feature_builder: IdentityFreeFeatureBuilder | None = None,
        rules: PolicyRuleSet | None = None,
        approval_threshold: float = 0.85,
        denial_threshold: float = 0.90,
        margin_threshold: float = 0.20,
        maximum_disagreement: float = 0.08,
        enabled: bool = False,
    ) -> None:
        for label, value in (
            ("approval_threshold", approval_threshold),
            ("denial_threshold", denial_threshold),
            ("margin_threshold", margin_threshold),
            ("maximum_disagreement", maximum_disagreement),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{label} must be finite and in [0, 1]")
        self._baseline = baseline
        self._model = model
        self._rules = rules or PolicyRuleSet()
        self._features = feature_builder or IdentityFreeFeatureBuilder(
            rules=self._rules,
        )
        self._approval_threshold = float(approval_threshold)
        self._denial_threshold = float(denial_threshold)
        self._margin_threshold = float(margin_threshold)
        self._maximum_disagreement = float(maximum_disagreement)
        self._decision_rule = GatedHybridDecisionRule(
            approval_threshold=self._approval_threshold,
            denial_threshold=self._denial_threshold,
            margin_threshold=self._margin_threshold,
            maximum_disagreement=self._maximum_disagreement,
        )
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """A candidate remains off until external grouped gates promote it."""

        return self._enabled

    def evaluate_case(
        self,
        resolved_case: ResolvedCase,
        *,
        recovery_route: str = "primary",
    ) -> HybridDecision:
        baseline = self._baseline.adjudicate_case(resolved_case)

        binding = _binding_decision(resolved_case)
        if binding is not None:
            if (
                baseline.row.adjudication == binding
                and baseline.trace.decision == binding
                and baseline.trace.authoritative_source
            ):
                outcome = baseline
            else:
                outcome = _replacement_outcome(
                    baseline,
                    binding,
                    authoritative=True,
                    reason="binding_visible_authority",
                )
            return HybridDecision(
                outcome=outcome,
                route="binding_authority",
                applied=outcome is not baseline,
            )

        if _baseline_authority_supported(resolved_case, baseline):
            return HybridDecision(
                outcome=baseline,
                route="validated_authoritative_policy",
                applied=False,
            )

        explicit_categories = _explicit_visible_violation_categories(
            resolved_case,
            self._rules,
        )
        if explicit_categories:
            if (
                baseline.row.adjudication == "DENIED"
                and baseline.trace.decision == "DENIED"
            ):
                outcome = baseline
            else:
                outcome = _replacement_outcome(
                    baseline,
                    "DENIED",
                    authoritative=False,
                    reason="explicit_visible_policy_violation",
                )
            return HybridDecision(
                outcome=outcome,
                route="visible_policy_violation",
                applied=outcome is not baseline,
            )

        # The model is residual.  It cannot reinterpret a deterministic policy
        # approval/denial, an authoritative decision, or malformed disagreement
        # between the typed row and trace.
        if (
            baseline.row.adjudication != "NEEDS_REVIEW"
            or baseline.trace.decision != "NEEDS_REVIEW"
            or baseline.trace.authoritative_source
        ):
            return HybridDecision(
                outcome=baseline,
                route="deterministic_policy",
                applied=False,
            )
        if not self._enabled:
            return HybridDecision(
                outcome=baseline,
                route="candidate_disabled",
                applied=False,
            )

        features = self._features.build(
            resolved_case,
            baseline,
            recovery_route=recovery_route,
        )
        prediction = self._model.predict(features)
        feature_decision = self._decision_rule.decide(
            "NEEDS_REVIEW",
            features,
            prediction,
        )
        if feature_decision.decision == "APPROVED":
            return HybridDecision(
                outcome=_replacement_outcome(
                    baseline,
                    "APPROVED",
                    authoritative=False,
                    reason="gated_identity_free_recovery",
                ),
                route="gated_model_approval",
                applied=True,
                model_prediction=prediction,
            )
        return HybridDecision(
            outcome=baseline,
            route=feature_decision.route,
            applied=False,
            model_prediction=prediction,
            veto_reasons=feature_decision.veto_reasons,
        )

    def adjudicate_case(self, resolved_case: ResolvedCase) -> AdjudicationOutcome:
        return self.evaluate_case(resolved_case).outcome

    def adjudicate(self, resolved_case: ResolvedCase) -> PredictionRow:
        return self.adjudicate_case(resolved_case).row
