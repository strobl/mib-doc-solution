"""Fail-closed RapidOCR recovery for genuinely unresolved output fields.

The primary Tesseract pass remains the source of case scope, policy, and
confidence.  RapidOCR is a second, independently resolved reading of the same
rendered pixels.  It may fill only primary ``FieldState.UNKNOWN`` output
values.  One frozen semantic head may turn a remaining review into a denial
when those newly visible values establish a published disqualifier; it never
creates an approval.  One separate frozen tie-breaker may repair only the
serialized applicant name when the already-extracted primary evidence
contains one stronger, exact-case biometric value.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .adjudication import AdjudicationOutcome, PolicyRuleSet
from .extraction import (
    CandidateEvidence,
    EvidenceType,
    OcrToken,
    VisibleEvidenceExtractor,
)
from .ingestion import Rect, RenderedCase, RenderedPage
from .models import PredictionRow
from .recovery_audit import (
    CandidateValidationPolicy,
    RecoveryAuditOverlay,
    RecoveryFieldAudit,
    VisibleRecoveryResult,
    observed_applicant_scopes,
    recovered_field_audit,
    unchanged_field_audit,
    visible_repair_field_audit,
)
from .resolution import FieldState, ResolvedCase, ResolvedField


# These are the scored output values for which the frozen liberal recovery was
# measured.  Case identity and adjudication are deliberately absent.  Risk is
# handled by its narrower, non-``none`` condition below.
RAPID_OUTPUT_FIELDS = frozenset(
    {
        "applicant_name",
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
        "fee_status",
    }
)
RAPID_RISK_FIELD = "risk_flags"
RAPID_RECOVERY_ROUTE_ID = "targeted_rapidocr"
RAPID_RISK_ROUTE_FIELDS = frozenset(
    {
        "species_code",
        "home_world",
        "visa_class",
        "arrival_date",
        "declared_purpose",
        "fee_status",
    }
)
AUTHORITATIVE_RAPID_TYPES = frozenset(
    {EvidenceType.ADJUDICATOR_STAMP, EvidenceType.SIGNED_MANUAL_NOTE}
)
AUTHORITATIVE_MINIMUM_CONFIDENCE = 0.90
BIOMETRIC_APPLICANT_MINIMUM_CONFIDENCE = 0.80
SOURCE_PRIORITY_MINIMUM_CONFIDENCE = 0.90
RAPID_BAD_CUES = frozenset({"strikethrough", "sample_denial_watermark"})
SEMANTIC_DENIAL_CONFIDENCE = 0.9166666666666666
SEMANTIC_EVIDENCE_MINIMUM_CONFIDENCE = 0.90
REVIEW_APPROVAL_CONFIDENCE = 0.80
XW1_MULTISOURCE_REVIEW_APPROVAL_CONFIDENCE = 0.98
XW1_MULTISOURCE_COMPLETE_REVIEW_RECOVERY = (
    "xw1_multisource_complete_review_recovery"
)
_COMPLETE_REVIEW_OUTPUT_FIELDS = (
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
_INCOMPLETE_REVIEW_VALUES = frozenset({"", "unknown", "null"})
_REVIEW_APPROVAL_SNAPSHOT_DATE = date(2026, 7, 7)
SEMANTIC_POLICY_RULES = PolicyRuleSet()
SEMANTIC_EVIDENCE_FIELDS = frozenset(
    {"risk_flags", "home_world", "visa_class", "sponsor_id"}
)
SEMANTIC_DENIAL_RULE_IDS = (
    "semantic_disqualifying_risk",
    "semantic_absolute_embargo",
    "semantic_wolf_non_diplomatic",
    "semantic_rapid_barred_sponsor",
)


class PrimaryAdjudicator(Protocol):
    def adjudicate_case(self, resolved_case: ResolvedCase) -> AdjudicationOutcome:
        """Return the primary policy row and its decision trace."""


class RapidOcrEngine:
    """Adapt the wheel-bundled RapidOCR models to ``VisibleEvidenceExtractor``.

    Import and model construction are lazy: importing the MIB package or
    running fake-based unit tests does not require RapidOCR to be installed.
    Production instances use a primitive string model root because the pinned
    OmegaConf 2.0.0 cannot store ``Path`` objects.  All three models are read
    from the installed wheel; no URL or user cache is consulted.
    """

    _VERSION = "3.9.2"
    _MODEL_SHA256 = {
        "PP-OCRv6_det_small.onnx": (
            "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"
        ),
        "PP-OCRv6_rec_small.onnx": (
            "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"
        ),
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx": (
            "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"
        ),
    }
    _TEXT_SCORE = 0.30

    def __init__(
        self,
        *,
        engine_factory: Callable[..., Any] | None = None,
        package_root: Path | str | None = None,
    ) -> None:
        verify_bundled_models = engine_factory is None
        if engine_factory is None:
            try:
                import rapidocr as rapidocr_package
            except ImportError as exc:  # pragma: no cover - production image path
                raise RuntimeError("RapidOCR is not installed") from exc
            engine_factory = rapidocr_package.RapidOCR
            package_file = getattr(rapidocr_package, "__file__", None)
            if package_file is None:
                raise RuntimeError("RapidOCR package location is unavailable")
            package_root = Path(package_file).resolve().parent
        if package_root is None:
            raise ValueError("package_root is required with a custom engine_factory")

        model_root_path = Path(package_root).resolve() / "models"
        if verify_bundled_models:
            self._verify_bundled_models(model_root_path)
        model_root = str(model_root_path)
        self._engine = engine_factory(
            params={
                "Global.model_root_dir": model_root,
                "Global.log_level": "error",
                "Global.text_score": self._TEXT_SCORE,
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )

    @classmethod
    def _verify_bundled_models(cls, model_root: Path) -> None:
        """Fail closed when the installed OCR weights differ from the pin."""

        for filename, expected_sha256 in sorted(cls._MODEL_SHA256.items()):
            model_path = model_root / filename
            try:
                actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise RuntimeError(
                    f"RapidOCR model is unavailable: {filename}"
                ) from exc
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"RapidOCR model digest mismatch: {filename}"
                )

    @property
    def provenance_id(self) -> str:
        """Identify the exact pinned model set and inference configuration."""

        model_set = ",".join(
            f"{filename}={digest}"
            for filename, digest in sorted(self._MODEL_SHA256.items())
        )
        return (
            f"rapidocr:version={self._VERSION}:models={model_set}:"
            f"text_score={self._TEXT_SCORE:.2f}:ort_threads=1/1"
        )

    def read_page(self, page: RenderedPage) -> tuple[OcrToken, ...]:
        result = self._engine(page.image_png)
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None or texts is None or scores is None:
            return ()

        tokens: list[OcrToken] = []
        for index, (box, text, score) in enumerate(
            zip(boxes, texts, scores),
            start=1,
        ):
            rendered = str(text).strip()
            if not rendered:
                continue
            points = tuple(box)
            if not points:
                continue
            xs = tuple(float(point[0]) for point in points)
            ys = tuple(float(point[1]) for point in points)
            tokens.append(
                OcrToken(
                    page_index=page.index,
                    text=rendered,
                    confidence=max(0.0, min(1.0, float(score))),
                    box=Rect(min(xs), min(ys), max(xs), max(ys)),
                    block_num=index,
                    paragraph_num=1,
                    line_num=1,
                    word_num=1,
                )
            )
        return tuple(tokens)


def build_rapid_extractor() -> VisibleEvidenceExtractor:
    """Create the exact full-page RapidOCR extractor used by the frozen run."""

    return VisibleEvidenceExtractor(
        ocr_engine=RapidOcrEngine(),
        psm6_refinement=False,
        consensus_retry=False,
        fee_receipt_retry=False,
        sparse_intake_retry=False,
        orientation_retry=False,
        trusted_scope_repair=False,
        risk_flag_retry=False,
        ocr_route_id=RAPID_RECOVERY_ROUTE_ID,
        ocr_view_id="rendered_page",
    )


class RapidOutputRecoveryProcessor:
    """Run primary OCR once, then conservatively repair serialized output.

    A separate RapidOCR extractor is initialized lazily per worker thread.
    This avoids sharing ONNX Runtime sessions across the four-worker batch
    pool.  Any RapidOCR import, initialization, extraction, linking,
    resolution, or overlay failure returns the primary-only repaired row.
    """

    def __init__(
        self,
        *,
        renderer: Any,
        primary_extractor: Any,
        linker: Any,
        resolver: Any,
        adjudicator: PrimaryAdjudicator,
        rapid_extractor_factory: Callable[[], Any] = build_rapid_extractor,
    ) -> None:
        self._renderer = renderer
        self._primary_extractor = primary_extractor
        self._linker = linker
        self._resolver = resolver
        self._adjudicator = adjudicator
        self._rapid_extractor_factory = rapid_extractor_factory
        self._local = threading.local()

    def _rapid_extractor(self) -> Any:
        extractor = getattr(self._local, "rapid_extractor", None)
        if extractor is None:
            extractor = self._rapid_extractor_factory()
            self._local.rapid_extractor = extractor
        return extractor

    def _fusion_enabled(self) -> bool:
        """Return whether the resolver owns coherent candidate fusion.

        Older/custom resolvers predate the explicit mode flag, so a missing
        attribute selects the isolated legacy resolver/output-repair branch.
        The frozen full pre-WO16 control is evaluated from its own revision;
        this compatibility branch is not a claim that the current end-to-end
        pipeline is identical to that historical revision.
        """

        return bool(getattr(self._resolver, "fusion_enabled", False))

    @staticmethod
    def _unknown_output_fields(resolved: ResolvedCase) -> frozenset[str]:
        return frozenset(
            field_name
            for field_name in RAPID_OUTPUT_FIELDS
            if (field := resolved.fields.get(field_name)) is not None
            and field.state is FieldState.UNKNOWN
        )

    @staticmethod
    def _risk_gap_is_safe(field: ResolvedField | None) -> bool:
        """Reproduce the frozen non-``none`` risk-routing safety condition."""

        if field is None or field.state is not FieldState.UNKNOWN:
            return False
        return all(
            candidate.value is None
            and not candidate.superseded
            and not (RAPID_BAD_CUES & set(candidate.visual_cues))
            for candidate in field.considered
        )

    @classmethod
    def _recover_non_none_risk(
        cls,
        resolved: ResolvedCase,
        unknown_fields: Iterable[str],
    ) -> bool:
        """Route the risk field only under the frozen liberal experiment gate.

        A safe unresolved primary risk needs either its own visible primary
        anchor or a separately unresolved non-identity field that already
        justifies the full-page Rapid pass.  Rapid's literal ``none`` never
        replaces the primary output; only a resolved non-``none`` value can.
        """

        risk_field = resolved.fields.get(RAPID_RISK_FIELD)
        if not cls._risk_gap_is_safe(risk_field):
            return False
        return bool(
            risk_field is not None
            and (
                risk_field.considered
                or RAPID_RISK_ROUTE_FIELDS.intersection(unknown_fields)
            )
        )

    @classmethod
    def _authoritative_rapid_decision(
        cls,
        *,
        case_id: str,
        source_sha256: str,
        primary_resolved: ResolvedCase,
        primary_outcome: AdjudicationOutcome,
        rapid_candidates: Iterable[CandidateEvidence],
    ) -> str | None:
        """Return one unanimous exact-case signed decision, otherwise abstain."""

        active_applicant = primary_resolved.active_applicant
        if (
            primary_outcome.row.adjudication != "NEEDS_REVIEW"
            or active_applicant is None
            or "authoritative_visible_decision"
            in primary_outcome.trace.review_reasons
        ):
            return None
        eligible = tuple(
            candidate
            for candidate in rapid_candidates
            if candidate.field_name == "adjudication"
            and candidate.value in {"APPROVED", "DENIED"}
            and candidate.evidence_type in AUTHORITATIVE_RAPID_TYPES
            and candidate.legible
            and not candidate.superseded
            and candidate.ocr_confidence >= AUTHORITATIVE_MINIMUM_CONFIDENCE
            and candidate.source == "visible_ocr"
            and candidate.case_id_hint == case_id
            and candidate.applicant_hint in {None, active_applicant}
            and not (RAPID_BAD_CUES & set(candidate.visual_cues))
            and cls._complete_candidate_provenance(
                candidate,
                source_sha256=source_sha256,
                linked_applicant=active_applicant,
            )
            and any(
                provenance.route_id == RAPID_RECOVERY_ROUTE_ID
                for provenance in candidate.ocr_provenance
            )
        )
        decisions = {candidate.value for candidate in eligible}
        return next(iter(decisions)) if len(decisions) == 1 else None

    @staticmethod
    def _primary_authoritative_decision(outcome: AdjudicationOutcome) -> bool:
        """Treat any primary authoritative trace as a semantic-head veto."""

        marker = "authoritative_visible_decision"
        return bool(
            outcome.trace.authoritative_source
            or marker in outcome.trace.denial_reasons
            or marker in outcome.trace.review_reasons
            or marker in outcome.trace.approval_facts
        )

    @staticmethod
    def _has_authoritative_rapid_decision(
        *,
        case_id: str,
        primary_resolved: ResolvedCase,
        rapid_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence],
    ) -> bool:
        """Veto the semantic head on any exact-case Rapid authority.

        Unlike the narrow authoritative output override, the veto includes an
        explicit ``NEEDS_REVIEW``.  Abstaining is safer than allowing a
        lower-precedence policy fact to replace a visible signed decision.
        """

        applicant_aliases = {
            None,
            primary_resolved.active_applicant,
            rapid_resolved.active_applicant,
        }
        return any(
            isinstance(candidate, CandidateEvidence)
            and candidate.field_name == "adjudication"
            and candidate.value in {"APPROVED", "DENIED", "NEEDS_REVIEW"}
            and candidate.evidence_type in AUTHORITATIVE_RAPID_TYPES
            and candidate.legible
            and not candidate.superseded
            and candidate.ocr_confidence >= AUTHORITATIVE_MINIMUM_CONFIDENCE
            and candidate.source == "visible_ocr"
            and candidate.case_id_hint == case_id
            and candidate.applicant_hint in applicant_aliases
            and not (RAPID_BAD_CUES & set(candidate.visual_cues))
            for candidate in rapid_candidates
        )

    @staticmethod
    def _unsafe_pages(
        candidates: Iterable[CandidateEvidence],
    ) -> frozenset[int]:
        """Return pages carrying a visible strikethrough or sample watermark."""

        return frozenset(
            candidate.page_index
            for candidate in candidates
            if isinstance(candidate, CandidateEvidence)
            and RAPID_BAD_CUES.intersection(candidate.visual_cues)
        )

    @staticmethod
    def _complete_candidate_provenance(
        candidate: CandidateEvidence,
        *,
        source_sha256: str,
        linked_applicant: str | None,
    ) -> bool:
        """Validate the physical observation before it can affect a decision."""

        if not candidate.ocr_provenance:
            return False
        allowed_scopes = {None, linked_applicant}
        return all(
            provenance.observation.source_sha256 == source_sha256
            and provenance.observation.page_index == candidate.page_index
            and provenance.observation.applicant_scope in allowed_scopes
            and provenance.observation.box.width > 0.0
            and provenance.observation.box.height > 0.0
            for provenance in candidate.ocr_provenance
        )

    @staticmethod
    def _visible_resolved_value(
        *,
        case_id: str,
        source_sha256: str,
        resolved: ResolvedCase,
        field_name: str,
        unsafe_pages: frozenset[int],
        minimum_confidence: float = SEMANTIC_EVIDENCE_MINIMUM_CONFIDENCE,
    ) -> str | None:
        """Return one exact-case visible winner, never a serialization prior."""

        resolved_field = resolved.fields.get(field_name)
        if (
            resolved_field is None
            or resolved_field.state is not FieldState.RESOLVED
            or resolved_field.value is None
        ):
            return None
        evidence = resolved_field.winning_evidence
        if (
            evidence is None
            or evidence.field_name != field_name
            or evidence.value != resolved_field.value
            or not evidence.legible
            or evidence.superseded
            or evidence.source != "visible_ocr"
            or evidence.evidence_type is EvidenceType.TEXT_LAYER
            or evidence.ocr_confidence < minimum_confidence
            or evidence.case_id_hint != case_id
            or evidence.applicant_hint not in {None, resolved.active_applicant}
            or evidence.page_index in unsafe_pages
            or RAPID_BAD_CUES.intersection(evidence.visual_cues)
            or not RapidOutputRecoveryProcessor._complete_candidate_provenance(
                evidence,
                source_sha256=source_sha256,
                linked_applicant=resolved.active_applicant,
            )
        ):
            return None
        return resolved_field.value

    @classmethod
    def _has_explicit_visible_none_risk(
        cls,
        *,
        case_id: str,
        source_sha256: str,
        primary_candidates: Iterable[CandidateEvidence],
        primary_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence],
        rapid_resolved: ResolvedCase | None,
    ) -> bool:
        """Require a resolved visible ``none`` fact, never the row fallback."""

        sources = (
            (tuple(primary_candidates), primary_resolved),
            (tuple(rapid_candidates), rapid_resolved),
        )
        for candidates, resolved in sources:
            if resolved is None:
                continue
            value = cls._visible_resolved_value(
                case_id=case_id,
                source_sha256=source_sha256,
                resolved=resolved,
                field_name=RAPID_RISK_FIELD,
                unsafe_pages=cls._unsafe_pages(candidates),
                minimum_confidence=REVIEW_APPROVAL_CONFIDENCE,
            )
            if (
                value is not None
                and " ".join(value.strip().split()).casefold() == "none"
            ):
                return True
        return False

    @staticmethod
    def _parse_risk_flags(value: str | None) -> frozenset[str]:
        if value in {None, "", "none"}:
            return frozenset()
        return frozenset(
            item.strip()
            for item in value.split("|")
            if item.strip() and item.strip() != "none"
        )

    @classmethod
    def _semantic_denial_rules(
        cls,
        *,
        case_id: str,
        source_sha256: str,
        payload: dict[str, object],
        primary_candidates: Iterable[CandidateEvidence],
        rapid_candidates: Iterable[CandidateEvidence],
        primary_resolved: ResolvedCase,
        rapid_resolved: ResolvedCase,
        primary_outcome: AdjudicationOutcome,
        unknown_fields: frozenset[str],
        recover_risk: bool,
    ) -> tuple[str, ...]:
        """Match the frozen, identity-free post-Rapid denial head.

        Primary and Rapid values stay separate so the barred-sponsor rule can
        require two independently recovered Rapid winners. Values used only
        for JSON schema completion never enter this method as evidence.
        """

        if (
            primary_outcome.row.adjudication != "NEEDS_REVIEW"
            or primary_outcome.trace.decision != "NEEDS_REVIEW"
            or cls._primary_authoritative_decision(primary_outcome)
            or cls._has_authoritative_rapid_decision(
                case_id=case_id,
                primary_resolved=primary_resolved,
                rapid_resolved=rapid_resolved,
                rapid_candidates=rapid_candidates,
            )
        ):
            return ()

        primary_unsafe_pages = cls._unsafe_pages(primary_candidates)
        rapid_unsafe_pages = cls._unsafe_pages(rapid_candidates)
        primary_values = {
            field_name: value
            for field_name in SEMANTIC_EVIDENCE_FIELDS
            if (
                value := cls._visible_resolved_value(
                    case_id=case_id,
                    source_sha256=source_sha256,
                    resolved=primary_resolved,
                    field_name=field_name,
                    unsafe_pages=primary_unsafe_pages,
                )
            )
            is not None
            and payload.get(field_name) == value
        }

        rapid_values: dict[str, str] = {}
        for field_name in SEMANTIC_EVIDENCE_FIELDS & unknown_fields:
            value = cls._visible_resolved_value(
                case_id=case_id,
                source_sha256=source_sha256,
                resolved=rapid_resolved,
                field_name=field_name,
                unsafe_pages=rapid_unsafe_pages,
            )
            if value is not None and payload.get(field_name) == value:
                rapid_values[field_name] = value
        if recover_risk:
            rapid_risk = cls._visible_resolved_value(
                case_id=case_id,
                source_sha256=source_sha256,
                resolved=rapid_resolved,
                field_name=RAPID_RISK_FIELD,
                unsafe_pages=rapid_unsafe_pages,
            )
            if rapid_risk not in {None, "none"} and payload.get(
                RAPID_RISK_FIELD
            ) == rapid_risk:
                rapid_values[RAPID_RISK_FIELD] = rapid_risk

        values = {**primary_values, **rapid_values}
        matches: list[str] = []
        if cls._parse_risk_flags(values.get("risk_flags")) & (
            SEMANTIC_POLICY_RULES.disqualifying_flags
        ):
            matches.append(SEMANTIC_DENIAL_RULE_IDS[0])
        if values.get("home_world") in SEMANTIC_POLICY_RULES.embargoed_worlds:
            matches.append(SEMANTIC_DENIAL_RULE_IDS[1])
        if (
            values.get("home_world")
            in SEMANTIC_POLICY_RULES.non_diplomatic_embargoed_worlds
            and values.get("visa_class") not in {None, "DIP-1"}
        ):
            matches.append(SEMANTIC_DENIAL_RULE_IDS[2])
        if (
            rapid_values.get("sponsor_id") in SEMANTIC_POLICY_RULES.barred_sponsors
            and rapid_values.get("visa_class") not in {None, "DIP-1"}
        ):
            matches.append(SEMANTIC_DENIAL_RULE_IDS[3])
        return tuple(matches)

    @classmethod
    def _recoverable_rapid_field(
        cls,
        *,
        case_id: str,
        source_sha256: str,
        resolved: ResolvedCase,
        field_name: str,
    ) -> ResolvedField | None:
        """Return a visible, provenance-complete Rapid winner for serialization."""

        field = resolved.fields.get(field_name)
        if (
            field is None
            or field.state is not FieldState.RESOLVED
            or field.value is None
        ):
            return None
        evidence = field.winning_evidence
        if (
            evidence is None
            or evidence.field_name != field_name
            or evidence.value != field.value
            or not evidence.legible
            or evidence.superseded
            or evidence.source != "visible_ocr"
            or evidence.evidence_type is EvidenceType.TEXT_LAYER
            or evidence.case_id_hint != case_id
            or evidence.applicant_hint not in {None, resolved.active_applicant}
            or RAPID_BAD_CUES.intersection(evidence.visual_cues)
            or not cls._complete_candidate_provenance(
                evidence,
                source_sha256=source_sha256,
                linked_applicant=resolved.active_applicant,
            )
            or not any(
                provenance.route_id == RAPID_RECOVERY_ROUTE_ID
                for provenance in evidence.ocr_provenance
            )
        ):
            return None
        return field

    @staticmethod
    def _audit_primary_field(
        resolved: ResolvedCase,
        field_name: str,
    ) -> ResolvedField:
        """Return the primary field, making an absent resolver slot explicit."""

        field = resolved.fields.get(field_name)
        if field is not None:
            return field
        return ResolvedField(
            field_name=field_name,
            state=FieldState.UNKNOWN,
            value=None,
            winning_evidence=None,
            considered=(),
            reason="field absent from primary resolver output",
        )

    @classmethod
    def _visible_recovery_result(
        cls,
        *,
        row: PredictionRow,
        primary_resolved: ResolvedCase,
        recovered_audits: dict[str, RecoveryFieldAudit] | None = None,
        linked_recovery_scope: str | None = None,
        final_fusion_audit_counts: Mapping[str, int] | None = None,
    ) -> VisibleRecoveryResult:
        """Pair the row with a complete immutable field-state audit overlay."""

        recovered_audits = dict(recovered_audits or {})
        audits: list[RecoveryFieldAudit] = []
        for field_name in _COMPLETE_REVIEW_OUTPUT_FIELDS:
            recovered = recovered_audits.get(field_name)
            if recovered is not None:
                audits.append(recovered)
                continue
            audits.append(
                unchanged_field_audit(
                    primary=cls._audit_primary_field(
                        primary_resolved,
                        field_name,
                    ),
                    serialized_value=getattr(row, field_name),
                    linked_recovery_scope=linked_recovery_scope,
                )
            )
        return VisibleRecoveryResult(
            row=row,
            audit=RecoveryAuditOverlay.from_fields(
                case_id=row.case_id,
                fields=audits,
            ),
            fusion_audit_counts=(
                primary_resolved.fusion_audit_counts
                if final_fusion_audit_counts is None
                else final_fusion_audit_counts
            ),
        )

    @staticmethod
    def _scope_key(value: str) -> str:
        return " ".join(value.split()).casefold()

    @classmethod
    def _recovery_scope_contract(
        cls,
        *,
        field_name: str,
        candidate: CandidateEvidence,
        primary_resolved: ResolvedCase,
        rapid_resolved: ResolvedCase,
    ) -> tuple[str | None, str] | None:
        """Bind recovered facts to the pre-existing active applicant.

        Applicant-name recovery is the sole exception: when primary linking
        has no active applicant, one unambiguous Rapid-linked name may establish
        the serialized name.  This records linkage; it does not fuse cases.
        """

        if field_name == "applicant_name" and primary_resolved.active_applicant is None:
            linked_scope = rapid_resolved.active_applicant
        else:
            if (
                primary_resolved.unresolved_linkage
                or primary_resolved.active_applicant is None
            ):
                return None
            linked_scope = primary_resolved.active_applicant
            if (
                rapid_resolved.active_applicant is not None
                and cls._scope_key(rapid_resolved.active_applicant)
                != cls._scope_key(linked_scope)
            ):
                return None

        if (
            linked_scope is None
            or rapid_resolved.unresolved_linkage
        ):
            return None
        scopes = observed_applicant_scopes(candidate)
        if any(
            cls._scope_key(scope) != cls._scope_key(linked_scope)
            for scope in scopes
        ):
            return None
        expected_observed_scope = linked_scope if scopes else None
        return expected_observed_scope, linked_scope

    @classmethod
    def _primary_visible_repair_audit(
        cls,
        *,
        field_name: str,
        candidate: CandidateEvidence,
        source_sha256: str,
        serialization_before: str,
        serialization_after: str,
        primary_resolved: ResolvedCase,
    ) -> RecoveryFieldAudit | None:
        """Validate and record an output-only repair over primary evidence."""

        if primary_resolved.unresolved_linkage:
            return None
        scopes = observed_applicant_scopes(candidate)
        if field_name == "applicant_name":
            if candidate.value is None:
                return None
            linked_scope = candidate.value
        else:
            linked_scope = primary_resolved.active_applicant
            if linked_scope is None:
                return None
        if any(
            cls._scope_key(scope) != cls._scope_key(linked_scope)
            for scope in scopes
        ):
            return None
        route_ids = tuple(
            sorted(
                {
                    provenance.route_id
                    for provenance in candidate.ocr_provenance
                    if (
                        provenance.view_box == candidate.box
                        or provenance.observation.box == candidate.box
                    )
                }
            )
        )
        if not route_ids:
            return None
        audit, _validation = visible_repair_field_audit(
            primary=cls._audit_primary_field(primary_resolved, field_name),
            serialization_before=serialization_before,
            serialization_after=serialization_after,
            candidate=candidate,
            recovery_source=route_ids[0],
            linked_recovery_scope=linked_scope,
            validation_policy=CandidateValidationPolicy(
                expected_field_name=field_name,
                expected_source_sha256=source_sha256,
                expected_page_index=candidate.page_index,
                expected_applicant_scope=linked_scope if scopes else None,
                minimum_confidence=0.0,
            ),
        )
        return audit

    @staticmethod
    def _review_approval_arrival_age(value: str) -> int | None:
        """Return the frozen snapshot age for one exact ISO arrival date."""

        try:
            normalized = value.strip()
            if normalized == "1900-01-01":
                return None
            arrival = date.fromisoformat(normalized)
        except (AttributeError, TypeError, ValueError):
            return None
        return (_REVIEW_APPROVAL_SNAPSHOT_DATE - arrival).days

    @classmethod
    def _review_approval_head(
        cls,
        *,
        final_row: PredictionRow,
        source_sha256: str,
        primary_candidates: Iterable[CandidateEvidence],
        primary_outcome: AdjudicationOutcome,
        primary_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence] = (),
        rapid_resolved: ResolvedCase | None = None,
    ) -> PredictionRow:
        """Apply the frozen identity-free three-branch review approval head.

        Existing primary or Rapid authority always vetoes this lower-precedence
        statistical recovery.  Candidate values and identities are never read:
        the first branch uses only the count of primary applicant candidates.
        """

        if (
            final_row.adjudication != "NEEDS_REVIEW"
            or not cls._has_explicit_visible_none_risk(
                case_id=primary_resolved.case_id,
                source_sha256=source_sha256,
                primary_candidates=primary_candidates,
                primary_resolved=primary_resolved,
                rapid_candidates=rapid_candidates,
                rapid_resolved=rapid_resolved,
            )
            or cls._primary_authoritative_decision(primary_outcome)
            or (
                rapid_resolved is not None
                and cls._has_authoritative_rapid_decision(
                    case_id=primary_resolved.case_id,
                    primary_resolved=primary_resolved,
                    rapid_resolved=rapid_resolved,
                    rapid_candidates=rapid_candidates,
                )
            )
        ):
            return final_row

        applicant_candidate_count = sum(
            isinstance(candidate, CandidateEvidence)
            and candidate.field_name == "applicant_name"
            for candidate in primary_candidates
        )
        arrival_age = cls._review_approval_arrival_age(
            final_row.arrival_date
        )
        trace = primary_outcome.trace
        matches = bool(
            applicant_candidate_count > 5
            or (
                arrival_age is not None
                and arrival_age > 71
                and "no_visible_biohazard_risk"
                in trace.approval_facts
            )
            or (
                arrival_age is not None
                and arrival_age <= 48
                and "required_sponsor_unknown" in trace.review_reasons
            )
        )
        if not matches:
            return final_row

        payload = final_row.to_dict()
        payload["adjudication"] = "APPROVED"
        payload["confidence"] = REVIEW_APPROVAL_CONFIDENCE
        return PredictionRow.from_mapping(
            payload,
            fallback_case_id=final_row.case_id,
        )

    @classmethod
    def _clean_multisource_candidate(
        cls,
        candidate: object,
        *,
        source_sha256: str,
        linked_applicant: str,
    ) -> bool:
        """Accept only live, legible facts read from rendered pixels."""

        return bool(
            isinstance(candidate, CandidateEvidence)
            and candidate.value is not None
            and candidate.legible
            and not candidate.superseded
            and candidate.source == "visible_ocr"
            and candidate.evidence_type is not EvidenceType.TEXT_LAYER
            and not RAPID_BAD_CUES.intersection(candidate.visual_cues)
            and cls._complete_candidate_provenance(
                candidate,
                source_sha256=source_sha256,
                linked_applicant=linked_applicant,
            )
        )

    @staticmethod
    def _complete_review_output(final_row: PredictionRow) -> bool:
        """Reject every schema fallback or substantively unknown output."""

        for field_name in _COMPLETE_REVIEW_OUTPUT_FIELDS:
            value = getattr(final_row, field_name)
            normalized = " ".join(value.strip().split()).casefold()
            if normalized in _INCOMPLETE_REVIEW_VALUES:
                return False
        return bool(
            final_row.sponsor_id != "SPN-0000"
            and final_row.arrival_date != "1900-01-01"
        )

    @classmethod
    def _same_page_source_fact(
        cls,
        *,
        case_id: str,
        source_sha256: str,
        active_applicant: str,
        field_name: str,
        expected_value: str,
        evidence_type: EvidenceType,
        candidates: tuple[CandidateEvidence, ...],
    ) -> bool:
        """Require exact-case facts with an exact same-page name anchor."""

        facts = tuple(
            candidate
            for candidate in candidates
            if cls._clean_multisource_candidate(
                candidate,
                source_sha256=source_sha256,
                linked_applicant=active_applicant,
            )
            and candidate.field_name == field_name
            and candidate.value == expected_value
            and candidate.evidence_type is evidence_type
            and candidate.case_id_hint == case_id
            and candidate.applicant_hint in {None, active_applicant}
        )
        if not facts:
            return False
        anchored_pages = {
            candidate.page_index
            for candidate in candidates
            if cls._clean_multisource_candidate(
                candidate,
                source_sha256=source_sha256,
                linked_applicant=active_applicant,
            )
            and candidate.field_name == "applicant_name"
            and candidate.value == active_applicant
            and candidate.evidence_type is evidence_type
            and candidate.case_id_hint == case_id
            and candidate.applicant_hint in {None, active_applicant}
        }
        return all(candidate.page_index in anchored_pages for candidate in facts)

    @classmethod
    def _multisource_conflict(
        cls,
        *,
        case_id: str,
        source_sha256: str,
        active_applicant: str,
        expected: tuple[tuple[str, str, EvidenceType], ...],
        candidates: tuple[CandidateEvidence, ...],
    ) -> bool:
        """Veto any relevant live source disagreement or applicant mismatch."""

        for field_name, expected_value, evidence_type in expected:
            if any(
                cls._clean_multisource_candidate(
                    candidate,
                    source_sha256=source_sha256,
                    linked_applicant=active_applicant,
                )
                and candidate.field_name == field_name
                and candidate.evidence_type is evidence_type
                and candidate.case_id_hint in {None, case_id}
                and (
                    candidate.value != expected_value
                    or candidate.applicant_hint not in {None, active_applicant}
                )
                for candidate in candidates
            ):
                return True
        return False

    @classmethod
    def _xw1_multisource_complete_review_recovery(
        cls,
        *,
        final_row: PredictionRow,
        source_sha256: str,
        primary_candidates: Iterable[CandidateEvidence],
        primary_outcome: AdjudicationOutcome,
        primary_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence] = (),
        rapid_resolved: ResolvedCase | None = None,
    ) -> PredictionRow:
        """Apply the audited conservative XW-1 multisource approval rule.

        The rule is deliberately lower precedence than every denial or signed
        decision.  It accepts only one fully populated final review shape and
        requires three exact-case facts across two independent structured
        source types, each tied to the active applicant on the same page.
        """

        primary_candidates = tuple(
            candidate
            for candidate in primary_candidates
            if isinstance(candidate, CandidateEvidence)
        )
        rapid_candidates = tuple(
            candidate
            for candidate in rapid_candidates
            if isinstance(candidate, CandidateEvidence)
        )
        all_candidates = primary_candidates + rapid_candidates
        trace = primary_outcome.trace
        active_applicant = primary_resolved.active_applicant
        if (
            final_row.adjudication != "NEEDS_REVIEW"
            or primary_outcome.row.adjudication != "NEEDS_REVIEW"
            or trace.decision != "NEEDS_REVIEW"
            or primary_outcome.row.confidence > 0.25
            or final_row.case_id != primary_resolved.case_id
            or primary_outcome.row.case_id != primary_resolved.case_id
            or active_applicant is None
            or final_row.applicant_name != active_applicant
            or final_row.visa_class != "XW-1"
            or not cls._has_explicit_visible_none_risk(
                case_id=primary_resolved.case_id,
                source_sha256=source_sha256,
                primary_candidates=primary_candidates,
                primary_resolved=primary_resolved,
                rapid_candidates=rapid_candidates,
                rapid_resolved=rapid_resolved,
            )
            or final_row.fee_status not in {"paid", "waived"}
            or not cls._complete_review_output(final_row)
            or trace.denial_reasons
            or cls._primary_authoritative_decision(primary_outcome)
            or primary_resolved.unresolved_linkage
            or primary_resolved.contested_fields
            or (
                rapid_resolved is not None
                and (
                    rapid_resolved.case_id != primary_resolved.case_id
                    or rapid_resolved.unresolved_linkage
                    or rapid_resolved.contested_fields
                    or cls._has_authoritative_rapid_decision(
                        case_id=primary_resolved.case_id,
                        primary_resolved=primary_resolved,
                        rapid_resolved=rapid_resolved,
                        rapid_candidates=rapid_candidates,
                    )
                )
            )
        ):
            return final_row

        facts = frozenset(trace.approval_facts)
        reasons = frozenset(trace.review_reasons)
        if not {
            "application_date_current_or_exempt",
            "sponsor_present_and_not_publicly_barred",
        }.issubset(facts):
            return final_row
        if final_row.fee_status == "paid":
            if (
                reasons
                != {
                    "required_output_unknown:risk_flags",
                    "risk_flags_unknown",
                }
                or "fee_paid" not in facts
            ):
                return final_row
        elif reasons != {"unsupported_fee_waiver"}:
            return final_row

        if any(
            RAPID_BAD_CUES.intersection(candidate.visual_cues)
            or candidate.evidence_type in AUTHORITATIVE_RAPID_TYPES
            for candidate in all_candidates
        ):
            return final_row
        if any(
            cls._clean_multisource_candidate(
                candidate,
                source_sha256=source_sha256,
                linked_applicant=active_applicant,
            )
            and candidate.field_name == RAPID_RISK_FIELD
            and " ".join(str(candidate.value).strip().split()).casefold()
            not in {"", "none", "unknown", "null"}
            for candidate in all_candidates
        ):
            return final_row

        expected = (
            (
                "visa_class",
                final_row.visa_class,
                EvidenceType.SPONSOR_ATTESTATION,
            ),
            (
                "home_world",
                final_row.home_world,
                EvidenceType.REGISTRY_EXTRACT,
            ),
            (
                "arrival_date",
                final_row.arrival_date,
                EvidenceType.REGISTRY_EXTRACT,
            ),
        )
        if cls._multisource_conflict(
            case_id=primary_resolved.case_id,
            source_sha256=source_sha256,
            active_applicant=active_applicant,
            expected=expected,
            candidates=all_candidates,
        ):
            return final_row
        if not all(
            cls._same_page_source_fact(
                case_id=primary_resolved.case_id,
                source_sha256=source_sha256,
                active_applicant=active_applicant,
                field_name=field_name,
                expected_value=expected_value,
                evidence_type=evidence_type,
                candidates=primary_candidates,
            )
            for field_name, expected_value, evidence_type in expected
        ):
            return final_row

        payload = final_row.to_dict()
        payload["adjudication"] = "APPROVED"
        payload["confidence"] = XW1_MULTISOURCE_REVIEW_APPROVAL_CONFIDENCE
        return PredictionRow.from_mapping(
            payload,
            fallback_case_id=final_row.case_id,
        )

    @classmethod
    def _apply_review_approval_heads(
        cls,
        *,
        final_row: PredictionRow,
        source_sha256: str,
        primary_candidates: Iterable[CandidateEvidence],
        primary_outcome: AdjudicationOutcome,
        primary_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence] = (),
        rapid_resolved: ResolvedCase | None = None,
    ) -> PredictionRow:
        """Run the conservative audited rule before the frozen broad head."""

        primary_candidates = tuple(primary_candidates)
        rapid_candidates = tuple(rapid_candidates)
        recovered = cls._xw1_multisource_complete_review_recovery(
            final_row=final_row,
            source_sha256=source_sha256,
            primary_candidates=primary_candidates,
            primary_outcome=primary_outcome,
            primary_resolved=primary_resolved,
            rapid_candidates=rapid_candidates,
            rapid_resolved=rapid_resolved,
        )
        return cls._review_approval_head(
            final_row=recovered,
            source_sha256=source_sha256,
            primary_candidates=primary_candidates,
            primary_outcome=primary_outcome,
            primary_resolved=primary_resolved,
            rapid_candidates=rapid_candidates,
            rapid_resolved=rapid_resolved,
        )

    @staticmethod
    def _repair_biometric_applicant(
        *,
        case_id: str,
        primary_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
    ) -> tuple[PredictionRow, CandidateEvidence | None]:
        """Prefer one stronger exact-case biometric name over one intake name.

        This is deliberately an output-only repair over evidence already read
        by the primary extractor.  It neither relinks the packet nor reruns
        resolution, policy, or calibration.  Any scope ambiguity abstains.
        """

        scoped = tuple(
            candidate
            for candidate in primary_candidates
            if isinstance(candidate, CandidateEvidence)
            and candidate.field_name == "applicant_name"
            and candidate.value is not None
            and candidate.evidence_type
            in {EvidenceType.BIOMETRIC_SLIP, EvidenceType.INTAKE_FORM}
            and candidate.source == "visible_ocr"
        )
        if any(
            candidate.case_id_hint not in {None, case_id}
            or RAPID_BAD_CUES.intersection(candidate.visual_cues)
            for candidate in scoped
        ):
            return primary_row, None

        relevant = tuple(
            candidate
            for candidate in scoped
            if candidate.legible and not candidate.superseded
        )

        exact_case = tuple(
            candidate
            for candidate in relevant
            if candidate.case_id_hint == case_id
        )
        biometrics = tuple(
            candidate
            for candidate in exact_case
            if candidate.evidence_type is EvidenceType.BIOMETRIC_SLIP
            and candidate.ocr_confidence
            >= BIOMETRIC_APPLICANT_MINIMUM_CONFIDENCE
        )
        intakes = tuple(
            candidate
            for candidate in exact_case
            if candidate.evidence_type is EvidenceType.INTAKE_FORM
        )
        biometric_values = {candidate.value for candidate in biometrics}
        intake_values = {candidate.value for candidate in intakes}
        if len(biometric_values) != 1 or len(intake_values) != 1:
            return primary_row, None

        biometric_value = next(iter(biometric_values))
        intake_value = next(iter(intake_values))
        if biometric_value == intake_value:
            return primary_row, None
        if max(candidate.ocr_confidence for candidate in biometrics) < max(
            candidate.ocr_confidence for candidate in intakes
        ):
            return primary_row, None

        payload = primary_row.to_dict()
        payload["applicant_name"] = biometric_value
        return (
            PredictionRow.from_mapping(payload, fallback_case_id=case_id),
            RapidOutputRecoveryProcessor._best_candidate(biometrics),
        )

    @staticmethod
    def _best_candidate(
        candidates: Iterable[CandidateEvidence],
    ) -> CandidateEvidence:
        """Select one equal-value support deterministically for the audit."""

        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("at least one candidate is required")
        return min(
            candidates,
            key=lambda candidate: (
                -candidate.ocr_confidence,
                candidate.page_index,
                candidate.box.left,
                candidate.box.bottom,
                candidate.box.right,
                candidate.box.top,
                candidate.evidence_type.value,
                candidate.value or "",
            ),
        )

    @staticmethod
    def _safe_primary_candidate(candidate: object) -> bool:
        """Return whether one candidate is usable by output-only repairs."""

        return bool(
            isinstance(candidate, CandidateEvidence)
            and candidate.value is not None
            and candidate.legible
            and not candidate.superseded
            and candidate.source == "visible_ocr"
            and not RAPID_BAD_CUES.intersection(candidate.visual_cues)
        )

    @classmethod
    def _page_has_active_applicant(
        cls,
        *,
        case_id: str,
        page_index: int,
        active_applicant: str | None,
        candidates: Iterable[CandidateEvidence],
    ) -> bool:
        """Require one clean same-page applicant anchor when case ID is absent."""

        if active_applicant is None:
            return False
        return any(
            cls._safe_primary_candidate(candidate)
            and candidate.field_name == "applicant_name"
            and candidate.page_index == page_index
            and candidate.value == active_applicant
            and candidate.case_id_hint in {None, case_id}
            and candidate.applicant_hint in {None, active_applicant}
            for candidate in candidates
        )

    @classmethod
    def _repair_source_priority_fields(
        cls,
        *,
        case_id: str,
        primary_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
        primary_resolved: ResolvedCase,
    ) -> tuple[PredictionRow, dict[str, CandidateEvidence]]:
        """Repair three serialized fields without changing policy state.

        The frozen gates cover values redundantly visible on a sponsor or
        registry page when a noisier intake read won the binding precedence
        hierarchy.  This method changes only JSON output fields: resolution,
        adjudication, trace, and calibrated confidence stay untouched.
        """

        candidates = tuple(
            candidate
            for candidate in primary_candidates
            if isinstance(candidate, CandidateEvidence)
        )
        active_applicant = primary_resolved.active_applicant
        payload = primary_row.to_dict()
        repaired: dict[str, CandidateEvidence] = {}

        def intake_winner(field_name: str) -> CandidateEvidence | None:
            field = primary_resolved.fields.get(field_name)
            if (
                field is None
                or field.state is not FieldState.RESOLVED
                or field.value is None
                or payload.get(field_name) != field.value
                or not cls._safe_primary_candidate(field.winning_evidence)
                or field.winning_evidence.evidence_type
                is not EvidenceType.INTAKE_FORM
            ):
                return None
            return field.winning_evidence

        def unique_value(
            field_name: str,
            evidence_type: EvidenceType,
            allowed_cues: frozenset[str],
            scope: Callable[[CandidateEvidence], bool],
        ) -> tuple[str, tuple[CandidateEvidence, ...]] | None:
            eligible = tuple(
                candidate
                for candidate in candidates
                if cls._safe_primary_candidate(candidate)
                and candidate.field_name == field_name
                and candidate.evidence_type is evidence_type
                and candidate.ocr_confidence
                >= SOURCE_PRIORITY_MINIMUM_CONFIDENCE
                and set(candidate.visual_cues) <= allowed_cues
                and scope(candidate)
            )
            values = {candidate.value for candidate in eligible}
            if len(values) != 1:
                return None
            return next(iter(values)), eligible

        visa_winner = intake_winner("visa_class")
        if visa_winner is not None:
            visa = unique_value(
                "visa_class",
                EvidenceType.SPONSOR_ATTESTATION,
                frozenset({"structured_sponsor_narrative"}),
                lambda candidate: (
                    candidate.case_id_hint in {None, case_id}
                    and candidate.applicant_hint in {None, active_applicant}
                    and (
                        candidate.case_id_hint == case_id
                        or cls._page_has_active_applicant(
                            case_id=case_id,
                            page_index=candidate.page_index,
                            active_applicant=active_applicant,
                            candidates=candidates,
                        )
                    )
                ),
            )
            if visa is not None and visa[0] != payload["visa_class"]:
                payload["visa_class"] = visa[0]
                repaired["visa_class"] = cls._best_candidate(visa[1])

        sponsor_winner = intake_winner("sponsor_id")
        if sponsor_winner is not None and active_applicant is not None:
            sponsor = unique_value(
                "sponsor_id",
                EvidenceType.SPONSOR_ATTESTATION,
                frozenset({"structured_sponsor_narrative"}),
                lambda candidate: (
                    candidate.case_id_hint in {None, case_id}
                    and candidate.applicant_hint in {None, active_applicant}
                    and cls._page_has_active_applicant(
                        case_id=case_id,
                        page_index=candidate.page_index,
                        active_applicant=active_applicant,
                        candidates=candidates,
                    )
                ),
            )
            if (
                sponsor is not None
                and sponsor[0] != payload["sponsor_id"]
                and max(
                    candidate.ocr_confidence for candidate in sponsor[1]
                )
                > sponsor_winner.ocr_confidence
            ):
                payload["sponsor_id"] = sponsor[0]
                repaired["sponsor_id"] = cls._best_candidate(sponsor[1])

        arrival_winner = intake_winner("arrival_date")
        if arrival_winner is not None and active_applicant is not None:
            arrival = unique_value(
                "arrival_date",
                EvidenceType.REGISTRY_EXTRACT,
                frozenset(),
                lambda candidate: (
                    candidate.case_id_hint == case_id
                    and candidate.applicant_hint == active_applicant
                ),
            )
            if (
                arrival is not None
                and arrival[0] != payload["arrival_date"]
                and max(
                    candidate.ocr_confidence for candidate in arrival[1]
                )
                > arrival_winner.ocr_confidence
            ):
                payload["arrival_date"] = arrival[0]
                repaired["arrival_date"] = cls._best_candidate(arrival[1])

        if not repaired:
            return primary_row, {}
        return (
            PredictionRow.from_mapping(payload, fallback_case_id=case_id),
            repaired,
        )

    def _recover(
        self,
        *,
        rendered: RenderedCase,
        primary_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
        primary_resolved: ResolvedCase,
        primary_outcome: AdjudicationOutcome,
        unknown_fields: frozenset[str],
        recover_risk: bool,
        pre_recovery_audits: dict[str, RecoveryFieldAudit],
    ) -> VisibleRecoveryResult:
        rapid_candidates = tuple(self._rapid_extractor().extract(rendered))
        rapid_linked = self._linker.link(rendered.case_id, rapid_candidates)
        rapid_resolved = self._resolver.resolve(rapid_linked)
        payload = primary_row.to_dict()
        recovered_audits = dict(pre_recovery_audits)

        # Overlay only fields whose primary state is truly UNKNOWN and whose
        # Rapid winner retains a complete physical observation.  Literal
        # ``unknown`` and ``none`` are visible values when a winning candidate
        # supports them; they must not be confused with serialization defaults.
        recovery_targets = set(unknown_fields)
        if recover_risk:
            recovery_targets.add(RAPID_RISK_FIELD)
        for field_name in sorted(recovery_targets):
            recovered_field = self._recoverable_rapid_field(
                case_id=primary_resolved.case_id,
                source_sha256=rendered.source_sha256,
                resolved=rapid_resolved,
                field_name=field_name,
            )
            if (
                recovered_field is None
                or recovered_field.winning_evidence is None
                or recovered_field.value is None
            ):
                continue
            candidate = recovered_field.winning_evidence
            scope_contract = self._recovery_scope_contract(
                field_name=field_name,
                candidate=candidate,
                primary_resolved=primary_resolved,
                rapid_resolved=rapid_resolved,
            )
            if scope_contract is None:
                continue
            expected_scope, linked_recovery_scope = scope_contract
            audit, _validation = recovered_field_audit(
                primary=self._audit_primary_field(
                    primary_resolved,
                    field_name,
                ),
                serialization_before=getattr(primary_row, field_name),
                serialization_after=recovered_field.value,
                candidate=candidate,
                recovery_source=RAPID_RECOVERY_ROUTE_ID,
                linked_recovery_scope=linked_recovery_scope,
                validation_policy=CandidateValidationPolicy(
                    expected_field_name=field_name,
                    expected_source_sha256=rendered.source_sha256,
                    expected_page_index=candidate.page_index,
                    expected_applicant_scope=expected_scope,
                    minimum_confidence=0.0,
                ),
            )
            if audit is not None:
                payload[field_name] = recovered_field.value
                recovered_audits[field_name] = audit

        # Ordinary output recovery and the signed-decision override preserve
        # primary identity and calibration. The frozen semantic denial head
        # below is the only calibrated exception.
        payload["case_id"] = primary_row.case_id
        payload["confidence"] = primary_row.confidence

        decision = self._authoritative_rapid_decision(
            case_id=primary_resolved.case_id,
            source_sha256=rendered.source_sha256,
            primary_resolved=primary_resolved,
            primary_outcome=primary_outcome,
            rapid_candidates=rapid_candidates,
        )
        if decision is not None:
            payload["adjudication"] = decision
        elif self._semantic_denial_rules(
            case_id=primary_resolved.case_id,
            source_sha256=rendered.source_sha256,
            payload=payload,
            primary_candidates=primary_candidates,
            rapid_candidates=rapid_candidates,
            primary_resolved=primary_resolved,
            rapid_resolved=rapid_resolved,
            primary_outcome=primary_outcome,
            unknown_fields=unknown_fields,
            recover_risk=recover_risk,
        ):
            payload["adjudication"] = "DENIED"
            payload["confidence"] = SEMANTIC_DENIAL_CONFIDENCE
        final_row = PredictionRow.from_mapping(
            payload,
            fallback_case_id=primary_row.case_id,
        )
        final_row = self._apply_review_approval_heads(
            final_row=final_row,
            source_sha256=rendered.source_sha256,
            primary_candidates=primary_candidates,
            primary_outcome=primary_outcome,
            primary_resolved=primary_resolved,
            rapid_candidates=rapid_candidates,
            rapid_resolved=rapid_resolved,
        )
        return self._visible_recovery_result(
            row=final_row,
            primary_resolved=primary_resolved,
            recovered_audits=recovered_audits,
            linked_recovery_scope=(
                primary_resolved.active_applicant
                or rapid_resolved.active_applicant
            ),
        )

    @classmethod
    def _fused_recovery_audit(
        cls,
        *,
        field_name: str,
        rendered: RenderedCase,
        primary_row: PredictionRow,
        fused_row: PredictionRow,
        primary_resolved: ResolvedCase,
        fused_resolved: ResolvedCase,
    ) -> RecoveryFieldAudit | None:
        """Audit one fused visible change against its physical observation."""

        fused_field = fused_resolved.fields.get(field_name)
        if (
            fused_field is None
            or fused_field.state is not FieldState.RESOLVED
            or fused_field.value is None
            or fused_field.winning_evidence is None
            or getattr(fused_row, field_name) != fused_field.value
        ):
            return None
        candidate = fused_field.winning_evidence
        if candidate.case_id_hint != rendered.case_id:
            return None

        observed_scopes = observed_applicant_scopes(candidate)
        linked_scope = fused_resolved.active_applicant
        if observed_scopes:
            if linked_scope is None or any(
                cls._scope_key(scope) != cls._scope_key(linked_scope)
                for scope in observed_scopes
            ):
                return None
            expected_scope = linked_scope
        else:
            expected_scope = None

        route_ids = tuple(
            sorted(
                {
                    provenance.route_id
                    for provenance in candidate.ocr_provenance
                    if (
                        provenance.view_box == candidate.box
                        or provenance.observation.box == candidate.box
                    )
                }
            )
        )
        if not route_ids:
            return None
        recovery_source = (
            RAPID_RECOVERY_ROUTE_ID
            if RAPID_RECOVERY_ROUTE_ID in route_ids
            else route_ids[0]
        )
        audit_builder = (
            recovered_field_audit
            if cls._audit_primary_field(primary_resolved, field_name).state
            is FieldState.UNKNOWN
            else visible_repair_field_audit
        )
        audit, _validation = audit_builder(
            primary=cls._audit_primary_field(primary_resolved, field_name),
            serialization_before=getattr(primary_row, field_name),
            serialization_after=getattr(fused_row, field_name),
            candidate=candidate,
            recovery_source=recovery_source,
            linked_recovery_scope=linked_scope,
            validation_policy=CandidateValidationPolicy(
                expected_field_name=field_name,
                expected_source_sha256=rendered.source_sha256,
                expected_page_index=candidate.page_index,
                expected_applicant_scope=expected_scope,
                minimum_confidence=0.0,
            ),
        )
        return audit

    @staticmethod
    def _fusion_rapid_candidates(
        candidates: Iterable[CandidateEvidence],
        *,
        recover_risk: bool,
    ) -> tuple[CandidateEvidence, ...]:
        """Keep Rapid evidence inside the audited output-field boundary.

        A Rapid decision or policy-only marker could otherwise influence the
        adjudicator without a corresponding field in ``RecoveryAuditOverlay``.
        Those candidates therefore remain diagnostic-only in fusion mode.
        """

        allowed_fields = set(RAPID_OUTPUT_FIELDS)
        if recover_risk:
            allowed_fields.add(RAPID_RISK_FIELD)
        return tuple(
            candidate
            for candidate in candidates
            if isinstance(candidate, CandidateEvidence)
            and candidate.field_name in allowed_fields
        )

    @staticmethod
    def _fused_field_changed(
        primary_field: ResolvedField | None,
        fused_field: ResolvedField | None,
    ) -> bool:
        """Return whether fusion changed evidence that policy can consume."""

        if primary_field is None or fused_field is None:
            return primary_field is not fused_field
        return (
            primary_field.state is not fused_field.state
            or primary_field.value != fused_field.value
            or primary_field.winning_evidence
            != fused_field.winning_evidence
        )

    def _recover_with_fusion(
        self,
        *,
        rendered: RenderedCase,
        primary_candidates: tuple[CandidateEvidence, ...],
        primary_resolved: ResolvedCase,
        primary_outcome: AdjudicationOutcome,
        unknown_fields: frozenset[str],
        recover_risk: bool,
    ) -> VisibleRecoveryResult:
        """Link and resolve primary plus Rapid evidence as one coherent case."""

        primary_row = primary_outcome.row
        if not unknown_fields and not recover_risk:
            final_row = self._apply_review_approval_heads(
                final_row=primary_row,
                source_sha256=rendered.source_sha256,
                primary_candidates=primary_candidates,
                primary_outcome=primary_outcome,
                primary_resolved=primary_resolved,
            )
            return self._visible_recovery_result(
                row=final_row,
                primary_resolved=primary_resolved,
                linked_recovery_scope=primary_resolved.active_applicant,
            )

        try:
            rapid_candidates = self._fusion_rapid_candidates(
                self._rapid_extractor().extract(rendered),
                recover_risk=recover_risk,
            )
            fused_linked = self._linker.link(
                rendered.case_id,
                (*primary_candidates, *rapid_candidates),
            )
            fused_resolved = self._resolver.resolve(fused_linked)
            fused_outcome = self._adjudicator.adjudicate_case(fused_resolved)
            fused_row = fused_outcome.row

            recovered_audits: dict[str, RecoveryFieldAudit] = {}
            for field_name in _COMPLETE_REVIEW_OUTPUT_FIELDS:
                evidence_changed = self._fused_field_changed(
                    primary_resolved.fields.get(field_name),
                    fused_resolved.fields.get(field_name),
                )
                serialization_changed = getattr(
                    fused_row,
                    field_name,
                ) != getattr(primary_row, field_name)
                if not evidence_changed and not serialization_changed:
                    continue
                if not evidence_changed:
                    raise ValueError(
                        "fused serialization changed without visible "
                        f"evidence: {field_name}"
                    )
                audit = self._fused_recovery_audit(
                    field_name=field_name,
                    rendered=rendered,
                    primary_row=primary_row,
                    fused_row=fused_row,
                    primary_resolved=primary_resolved,
                    fused_resolved=fused_resolved,
                )
                if audit is None:
                    # A field-level change without a complete exact-source,
                    # exact-applicant physical observation is not recoverable.
                    raise ValueError(
                        f"unauditable fused recovery: {field_name}"
                    )
                recovered_audits[field_name] = audit

            # Policy-only facts and authoritative decisions have no slot in
            # the recovery audit.  They must therefore remain exactly primary
            # in WO16; a later policy work order may add a dedicated contract.
            output_fields = set(_COMPLETE_REVIEW_OUTPUT_FIELDS)
            for field_name in set(primary_resolved.fields) | set(
                fused_resolved.fields
            ):
                if field_name in output_fields or field_name == "case_id":
                    continue
                if self._fused_field_changed(
                    primary_resolved.fields.get(field_name),
                    fused_resolved.fields.get(field_name),
                ):
                    raise ValueError(
                        f"unaudited fused policy change: {field_name}"
                    )

            if (
                fused_resolved.case_id != primary_resolved.case_id
                or fused_resolved.rescinded_decision
                != primary_resolved.rescinded_decision
            ):
                raise ValueError("fusion changed unaudited case policy state")
            linkage_changed = (
                fused_resolved.active_applicant
                != primary_resolved.active_applicant
                or fused_resolved.unresolved_linkage
                != primary_resolved.unresolved_linkage
                or fused_resolved.unresolved_reasons
                != primary_resolved.unresolved_reasons
            )
            if linkage_changed and "applicant_name" not in recovered_audits:
                raise ValueError("fusion changed unaudited applicant linkage")
            if (
                fused_outcome.trace != primary_outcome.trace
                or fused_row.adjudication != primary_row.adjudication
                or fused_row.confidence != primary_row.confidence
            ) and not recovered_audits:
                raise ValueError(
                    "fusion changed policy without an audited visible field"
                )

            return self._visible_recovery_result(
                row=fused_row,
                primary_resolved=primary_resolved,
                recovered_audits=recovered_audits,
                linked_recovery_scope=fused_resolved.active_applicant,
                final_fusion_audit_counts=(
                    fused_resolved.fusion_audit_counts
                ),
            )
        except Exception:
            # Fusion and RapidOCR are optional recovery.  A malformed,
            # ambiguous, or unauditable fused result cannot replace the
            # coherent primary adjudication.
            final_row = self._apply_review_approval_heads(
                final_row=primary_row,
                source_sha256=rendered.source_sha256,
                primary_candidates=primary_candidates,
                primary_outcome=primary_outcome,
                primary_resolved=primary_resolved,
            )
            return self._visible_recovery_result(
                row=final_row,
                primary_resolved=primary_resolved,
                linked_recovery_scope=primary_resolved.active_applicant,
            )

    def process_case_with_audit(self, pdf_path: Path) -> VisibleRecoveryResult:
        """Process one case and retain explicit evidence/default distinctions."""

        rendered = self._renderer.render(pdf_path)
        primary_candidates = tuple(self._primary_extractor.extract(rendered))
        primary_linked = self._linker.link(rendered.case_id, primary_candidates)
        primary_resolved = self._resolver.resolve(primary_linked)
        primary_outcome = self._adjudicator.adjudicate_case(primary_resolved)

        if self._fusion_enabled():
            return self._recover_with_fusion(
                rendered=rendered,
                primary_candidates=primary_candidates,
                primary_resolved=primary_resolved,
                primary_outcome=primary_outcome,
                unknown_fields=self._unknown_output_fields(primary_resolved),
                recover_risk=self._recover_non_none_risk(
                    primary_resolved,
                    self._unknown_output_fields(primary_resolved),
                ),
            )

        base_primary_row = primary_outcome.row
        primary_row, repaired_applicant = self._repair_biometric_applicant(
            case_id=rendered.case_id,
            primary_row=base_primary_row,
            primary_candidates=primary_candidates,
        )
        pre_recovery_audits: dict[str, RecoveryFieldAudit] = {}
        if repaired_applicant is not None:
            applicant_audit = self._primary_visible_repair_audit(
                field_name="applicant_name",
                candidate=repaired_applicant,
                source_sha256=rendered.source_sha256,
                serialization_before=base_primary_row.applicant_name,
                serialization_after=primary_row.applicant_name,
                primary_resolved=primary_resolved,
            )
            if applicant_audit is None:
                primary_row = base_primary_row
                repaired_applicant = None
            else:
                pre_recovery_audits["applicant_name"] = applicant_audit

        before_source_repairs = primary_row
        primary_row, source_repair_candidates = self._repair_source_priority_fields(
            case_id=rendered.case_id,
            primary_row=primary_row,
            primary_candidates=primary_candidates,
            primary_resolved=primary_resolved,
        )
        accepted_source_fields: set[str] = set()
        repaired_payload = primary_row.to_dict()
        for field_name, candidate in sorted(source_repair_candidates.items()):
            repair_audit = self._primary_visible_repair_audit(
                field_name=field_name,
                candidate=candidate,
                source_sha256=rendered.source_sha256,
                serialization_before=getattr(before_source_repairs, field_name),
                serialization_after=getattr(primary_row, field_name),
                primary_resolved=primary_resolved,
            )
            if repair_audit is None:
                repaired_payload[field_name] = getattr(
                    before_source_repairs,
                    field_name,
                )
                continue
            accepted_source_fields.add(field_name)
            pre_recovery_audits[field_name] = repair_audit
        if accepted_source_fields != set(source_repair_candidates):
            primary_row = PredictionRow.from_mapping(
                repaired_payload,
                fallback_case_id=rendered.case_id,
            )

        unknown_fields = self._unknown_output_fields(primary_resolved)
        if repaired_applicant:
            # The exact-case biometric fact is already frozen into the output;
            # an independent OCR pass must not replace it again.
            unknown_fields = unknown_fields - {"applicant_name"}
        unknown_fields = unknown_fields - accepted_source_fields
        recover_risk = self._recover_non_none_risk(
            primary_resolved,
            unknown_fields,
        )
        if not unknown_fields and not recover_risk:
            final_row = self._apply_review_approval_heads(
                final_row=primary_row,
                source_sha256=rendered.source_sha256,
                primary_candidates=primary_candidates,
                primary_outcome=primary_outcome,
                primary_resolved=primary_resolved,
            )
            return self._visible_recovery_result(
                row=final_row,
                primary_resolved=primary_resolved,
                recovered_audits=pre_recovery_audits,
                linked_recovery_scope=primary_resolved.active_applicant,
            )

        try:
            return self._recover(
                rendered=rendered,
                primary_row=primary_row,
                primary_candidates=primary_candidates,
                primary_resolved=primary_resolved,
                primary_outcome=primary_outcome,
                unknown_fields=unknown_fields,
                recover_risk=recover_risk,
                pre_recovery_audits=pre_recovery_audits,
            )
        except Exception:
            # RapidOCR is optional recovery, never a reason to lose a primary
            # prediction or abort the batch.
            final_row = self._apply_review_approval_heads(
                final_row=primary_row,
                source_sha256=rendered.source_sha256,
                primary_candidates=primary_candidates,
                primary_outcome=primary_outcome,
                primary_resolved=primary_resolved,
            )
            return self._visible_recovery_result(
                row=final_row,
                primary_resolved=primary_resolved,
                recovered_audits=pre_recovery_audits,
                linked_recovery_scope=primary_resolved.active_applicant,
            )

    def process_case(self, pdf_path: Path) -> PredictionRow:
        """Return the schema row while keeping the audit API opt-in."""

        result = self.process_case_with_audit(pdf_path)
        observer = getattr(self, "_fusion_audit_observer", None)
        if observer is not None:
            if not callable(observer):
                raise TypeError("fusion audit observer must be callable")
            observer(result.fusion_audit_counts)
        return result.row
