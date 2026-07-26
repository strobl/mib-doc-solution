"""Development-only, one-variable OCR ablation measurement.

The submitted runtime never imports this module.  It builds explicit
development variants, records label-blind runtime observations, and then uses
the repository's official evaluator in a separate reporting step.  That
separation prevents truth labels from entering the OCR execution path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "mib_ocr_ablation_v1"
REPORT_VERSION = "mib_ocr_ablation_report_v3"
MINIMUM_DETERMINISM_REPEATS = 2
PRIORITY_FIELDS = (
    "risk_flags",
    "fee_status",
    "applicant_name",
    "sponsor_id",
    "arrival_date",
    "visa_class",
    "species_code",
    "home_world",
    "declared_purpose",
)
CHECKBOX_ACTIVITY_COUNTERS = frozenset(
    {
        "pages_scanned",
        "complete_groups",
        "checked_groups",
        "candidates_added",
        "ambiguous_groups",
    }
)

BASELINE_CONFIG: Mapping[str, Mapping[str, bool]] = {
    "primary": {
        "psm6_refinement": True,
        "cross_view_consensus": True,
        "fee_threshold_consensus": True,
        "sparse_intake_crop_consensus": True,
        "orientation_retry": True,
        "trusted_scope_repair": True,
        "risk_geometry_retry": True,
        "renderer_deskew": True,
        "visible_cue_interpretation": True,
    },
    "secondary": {
        "rapid_uncertain_fields": True,
    },
    "candidate": {
        "bounded_contrast": False,
        "checkbox_state_recovery": False,
        "template_registration": False,
    },
}

_ALLOWED_VARIABLES = frozenset(
    {
        "primary.psm6_refinement",
        "primary.cross_view_consensus",
        "primary.fee_threshold_consensus",
        "primary.sparse_intake_crop_consensus",
        "primary.orientation_retry",
        "primary.trusted_scope_repair",
        "primary.risk_geometry_retry",
        "primary.renderer_deskew",
        "primary.visible_cue_interpretation",
        "secondary.rapid_uncertain_fields",
        "candidate.checkbox_state_recovery",
        "candidate.bounded_contrast",
        "candidate.template_registration",
    }
)


class AblationConfigurationError(ValueError):
    """An ablation plan or observation is not comparable to the baseline."""


@dataclass(frozen=True)
class AblationVariant:
    """One bounded technique comparison against the frozen production baseline."""

    variant_id: str
    family: str
    technique: str
    changed_variable: str
    config: Mapping[str, Mapping[str, bool]]
    target_fields: tuple[str, ...]
    technique_enabled_in: str = "baseline"
    hypothesis: str = ""

    def __post_init__(self) -> None:
        if self.technique_enabled_in not in {"baseline", "variant"}:
            raise AblationConfigurationError(
                "technique_enabled_in must be baseline or variant"
            )
        if not self.variant_id or self.variant_id == "baseline":
            raise AblationConfigurationError(
                "variant_id must be non-empty and cannot be baseline"
            )
        unknown_fields = sorted(set(self.target_fields) - set(PRIORITY_FIELDS))
        if unknown_fields:
            raise AblationConfigurationError(
                f"unknown target fields: {', '.join(unknown_fields)}"
            )
        validate_one_variable_config(
            self.config,
            changed_variable=self.changed_variable,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "family": self.family,
            "technique": self.technique,
            "changed_variable": self.changed_variable,
            "config": _plain_config(self.config),
            "target_fields": list(self.target_fields),
            "technique_enabled_in": self.technique_enabled_in,
            "hypothesis": self.hypothesis,
        }


def _plain_config(
    config: Mapping[str, Mapping[str, bool]],
) -> dict[str, dict[str, bool]]:
    return {
        section: {name: bool(value) for name, value in sorted(values.items())}
        for section, values in sorted(config.items())
    }


def _variant_config(changed_variable: str, value: bool) -> dict[str, dict[str, bool]]:
    config = _plain_config(BASELINE_CONFIG)
    section, name = changed_variable.split(".", 1)
    config[section][name] = value
    return config


def _flatten_config(
    config: Mapping[str, Mapping[str, bool]],
) -> dict[str, bool]:
    if not isinstance(config, Mapping):
        raise AblationConfigurationError("config must be a mapping")
    flattened: dict[str, bool] = {}
    for section, values in config.items():
        if not isinstance(section, str) or not isinstance(values, Mapping):
            raise AblationConfigurationError(
                "config sections must map string names to booleans"
            )
        for name, value in values.items():
            if not isinstance(name, str) or not isinstance(value, bool):
                raise AblationConfigurationError(
                    "every ablation setting must be a boolean"
                )
            flattened[f"{section}.{name}"] = value
    return flattened


def validate_one_variable_config(
    config: Mapping[str, Mapping[str, bool]],
    *,
    changed_variable: str,
) -> None:
    """Require the candidate to differ from baseline in exactly one setting."""

    baseline = _flatten_config(BASELINE_CONFIG)
    candidate = _flatten_config(config)
    if set(candidate) != set(baseline):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise AblationConfigurationError(
            f"config key mismatch; missing={missing}, extra={extra}"
        )
    differences = sorted(
        key for key in baseline if baseline[key] != candidate[key]
    )
    if differences != [changed_variable]:
        raise AblationConfigurationError(
            "variant must differ from baseline in exactly its declared variable; "
            f"declared={changed_variable!r}, actual={differences}"
        )
    if changed_variable not in _ALLOWED_VARIABLES:
        raise AblationConfigurationError(
            f"unsupported or unbounded ablation variable: {changed_variable}"
        )


def registered_variants() -> tuple[AblationVariant, ...]:
    """Return the bounded marginal-contribution experiments in stable order."""

    return (
        AblationVariant(
            variant_id="without_psm6_refinement",
            family="page_segmentation",
            technique="Selective PSM 6 refinement of uncertain visible lines",
            changed_variable="primary.psm6_refinement",
            config=_variant_config("primary.psm6_refinement", False),
            target_fields=PRIORITY_FIELDS,
            hypothesis=(
                "PSM 6 should improve uncertain structured rows without adding "
                "an unconditional full-page secondary pass."
            ),
        ),
        AblationVariant(
            variant_id="without_cross_view_consensus",
            family="cross_view_consensus",
            technique="Independent PSM 3/4 agreement for unresolved fields",
            changed_variable="primary.cross_view_consensus",
            config=_variant_config("primary.cross_view_consensus", False),
            target_fields=PRIORITY_FIELDS,
            hypothesis=(
                "Independent OCR-view agreement should recover fields while "
                "abstaining on conflicting readings."
            ),
        ),
        AblationVariant(
            variant_id="without_fee_threshold_consensus",
            family="normalized_crop_threshold",
            technique="Bounded fee-row crop with four threshold views",
            changed_variable="primary.fee_threshold_consensus",
            config=_variant_config("primary.fee_threshold_consensus", False),
            target_fields=("fee_status",),
            hypothesis=(
                "A template-relative receipt crop and threshold consensus should "
                "recover fee status without blanket high-DPI OCR."
            ),
        ),
        AblationVariant(
            variant_id="without_sparse_intake_crop_consensus",
            family="normalized_crop_layout",
            technique="Template-relative intake crop with PSM 6/3 agreement",
            changed_variable="primary.sparse_intake_crop_consensus",
            config=_variant_config(
                "primary.sparse_intake_crop_consensus",
                False,
            ),
            target_fields=(
                "applicant_name",
                "sponsor_id",
                "arrival_date",
                "visa_class",
                "species_code",
                "home_world",
                "declared_purpose",
            ),
            hypothesis=(
                "Normalized intake crops should recover sparse identity, date, "
                "and closed-vocabulary rows with bounded OCR."
            ),
        ),
        AblationVariant(
            variant_id="without_orientation_retry",
            family="orientation",
            technique="Bounded orientation retry on candidate-free pages",
            changed_variable="primary.orientation_retry",
            config=_variant_config("primary.orientation_retry", False),
            target_fields=PRIORITY_FIELDS,
            hypothesis=(
                "At most two candidate-free pages should benefit from bounded "
                "orientation correction without rotating the whole corpus."
            ),
        ),
        AblationVariant(
            variant_id="without_trusted_scope_repair",
            family="active_applicant_scope",
            technique="Visible exact-case applicant-scope repair",
            changed_variable="primary.trusted_scope_repair",
            config=_variant_config("primary.trusted_scope_repair", False),
            target_fields=(
                "applicant_name",
                "sponsor_id",
                "arrival_date",
                "visa_class",
                "species_code",
                "home_world",
                "declared_purpose",
            ),
            hypothesis=(
                "Visible applicant and case anchors should repair scope without "
                "identity tables or filename-conditioned predictions."
            ),
        ),
        AblationVariant(
            variant_id="without_risk_geometry_retry",
            family="line_cell_geometry",
            technique="Cropped risk-row geometry consensus",
            changed_variable="primary.risk_geometry_retry",
            config=_variant_config("primary.risk_geometry_retry", False),
            target_fields=("risk_flags",),
            hypothesis=(
                "Line and cell geometry in the visible risk block should recover "
                "risk wording while preserving strike and correction vetoes."
            ),
        ),
        AblationVariant(
            variant_id="without_renderer_deskew",
            family="bounded_deskew",
            technique="Bounded render-time deskew over plus or minus three degrees",
            changed_variable="primary.renderer_deskew",
            config=_variant_config("primary.renderer_deskew", False),
            target_fields=PRIORITY_FIELDS,
            hypothesis=(
                "Bounded render-time deskew should recover OCR evidence on "
                "slightly rotated pages without changing render resolution."
            ),
        ),
        AblationVariant(
            variant_id="without_visible_cue_interpretation",
            family="visible_status_cues",
            technique=(
                "Visible stamp, correction, watermark, and strikethrough "
                "interpretation"
            ),
            changed_variable="primary.visible_cue_interpretation",
            config=_variant_config("primary.visible_cue_interpretation", False),
            target_fields=PRIORITY_FIELDS,
            hypothesis=(
                "Visible status cues should prevent superseded or decorative "
                "readings from entering evidence resolution."
            ),
        ),
        AblationVariant(
            variant_id="without_targeted_rapidocr",
            family="secondary_ocr",
            technique="RapidOCR routed only to unresolved output fields",
            changed_variable="secondary.rapid_uncertain_fields",
            config=_variant_config("secondary.rapid_uncertain_fields", False),
            target_fields=PRIORITY_FIELDS,
            hypothesis=(
                "A secondary engine should add value only after the primary path "
                "abstains; unconditional full-page RapidOCR remains out of scope."
            ),
        ),
        AblationVariant(
            variant_id="with_checked_fee_option_recovery",
            family="visible_checkbox_pixels",
            technique=(
                "Exact three-option fee group with one pixel-confirmed check "
                "and two pixel-confirmed empty boxes"
            ),
            changed_variable="candidate.checkbox_state_recovery",
            config=_variant_config("candidate.checkbox_state_recovery", True),
            target_fields=("fee_status",),
            technique_enabled_in="variant",
            hypothesis=(
                "A fail-closed pixel check should recover a fee state only from "
                "one complete, aligned, uncorrected option group."
            ),
        ),
        AblationVariant(
            variant_id="with_bounded_template_registration",
            family="template_registration",
            technique=(
                "Bounded content-frame translation to canonical page "
                "coordinates"
            ),
            changed_variable="candidate.template_registration",
            config=_variant_config("candidate.template_registration", True),
            target_fields=PRIORITY_FIELDS,
            technique_enabled_in="variant",
            hypothesis=(
                "A label-blind, translation-only registration should expose "
                "shifted template fields without rescaling or rotating pages."
            ),
        ),
        AblationVariant(
            variant_id="with_bounded_contrast",
            family="bounded_contrast",
            technique=(
                "Autocontrast on at most two visibly low-contrast pages per case"
            ),
            changed_variable="candidate.bounded_contrast",
            config=_variant_config("candidate.bounded_contrast", True),
            target_fields=PRIORITY_FIELDS,
            technique_enabled_in="variant",
            hypothesis=(
                "A visible-pixel contrast gate should recover faint fields "
                "without changing already high-contrast pages."
            ),
        ),
    )


def variant_by_id(variant_id: str) -> AblationVariant | None:
    if variant_id == "baseline":
        return None
    variants = {variant.variant_id: variant for variant in registered_variants()}
    try:
        return variants[variant_id]
    except KeyError as exc:
        raise AblationConfigurationError(f"unknown variant: {variant_id}") from exc


def config_sha256(config: Mapping[str, Mapping[str, bool]]) -> str:
    payload = json.dumps(
        _plain_config(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _disabled_rapid_extractor() -> Any:
    raise RuntimeError("RapidOCR disabled by development ablation")


class _VisibleCuesDisabled:
    """Return no visible cues while preserving the extractor interface."""

    @staticmethod
    def prepare_page(grayscale: Any) -> Any:
        return grayscale

    @staticmethod
    def cues_for_line(line: Any, page_pixels: Any) -> tuple[str, ...]:
        del line, page_pixels
        return ()


def build_ablation_processor(variant_id: str) -> Any:
    """Build a processor with exactly one bounded development setting changed.

    ``baseline`` delegates to the production composition root.  Every other
    processor repeats the same composition while changing one declared OCR
    setting.  Additive candidates remain under ``devtools`` and cannot become
    a submission dependency through this harness.
    """

    from mib_pipeline import (
        AdjudicationEngine,
        CaseLinker,
        ConfidenceCalibrator,
        DocumentRenderer,
        EvidencePrecedenceResolver,
        GeneralizablePolicyExceptionStore,
        OutputConfidenceRecalibrationProcessor,
        OutputConfidenceRecalibrator,
        RapidOutputRecoveryProcessor,
        ReviewDenialRecoveryAdjudicator,
        TesseractOcrEngine,
        VisibleEvidenceExtractor,
        build_production_processor,
    )

    variant = variant_by_id(variant_id)
    if variant is None:
        return build_production_processor()
    config = _flatten_config(variant.config)
    rapid_factory = (
        None
        if config["secondary.rapid_uncertain_fields"]
        else _disabled_rapid_extractor
    )
    rapid_arguments: dict[str, Any] = {}
    if rapid_factory is not None:
        rapid_arguments["rapid_extractor_factory"] = rapid_factory
    renderer: Any
    if config["primary.renderer_deskew"]:
        renderer = DocumentRenderer()
    else:
        class DeskewDisabledDocumentRenderer(DocumentRenderer):
            @staticmethod
            def _estimate_skew(
                image: Any,
                image_module: Any,
                numpy_module: Any,
            ) -> float:
                del image, image_module, numpy_module
                return 0.0

        renderer = DeskewDisabledDocumentRenderer()
    if config["candidate.template_registration"]:
        from devtools.render_candidates import (
            BoundedTemplateRegistrationRenderer,
        )

        renderer = BoundedTemplateRegistrationRenderer(renderer)
    if config["candidate.bounded_contrast"]:
        from devtools.render_candidates import BoundedContrastRenderer

        renderer = BoundedContrastRenderer(renderer)
    cue_detector = (
        None
        if config["primary.visible_cue_interpretation"]
        else _VisibleCuesDisabled()
    )
    recording_ocr: Any | None = None
    primary_ocr_arguments: dict[str, Any] = {}
    if config["candidate.checkbox_state_recovery"]:
        from devtools.checked_box_candidate import RecordingOcrEngine

        recording_ocr = RecordingOcrEngine(TesseractOcrEngine())
        primary_ocr_arguments["ocr_engine"] = recording_ocr
    primary_extractor: Any = VisibleEvidenceExtractor(
        cue_detector=cue_detector,
        packet_page_type_markers=True,
        psm6_refinement=config["primary.psm6_refinement"],
        consensus_retry=config["primary.cross_view_consensus"],
        fee_receipt_retry=config["primary.fee_threshold_consensus"],
        sparse_intake_retry=config["primary.sparse_intake_crop_consensus"],
        orientation_retry=config["primary.orientation_retry"],
        trusted_scope_repair=config["primary.trusted_scope_repair"],
        risk_flag_retry=config["primary.risk_geometry_retry"],
        **primary_ocr_arguments,
    )
    if recording_ocr is not None:
        from devtools.checked_box_candidate import CheckedFeeOptionExtractor

        primary_extractor = CheckedFeeOptionExtractor(
            delegate=primary_extractor,
            recording_ocr=recording_ocr,
        )
    processor = RapidOutputRecoveryProcessor(
        renderer=renderer,
        primary_extractor=primary_extractor,
        linker=CaseLinker(),
        resolver=EvidencePrecedenceResolver(),
        adjudicator=ReviewDenialRecoveryAdjudicator(
            AdjudicationEngine(
                calibrator=ConfidenceCalibrator.from_pinned_artifact(),
                exceptions=GeneralizablePolicyExceptionStore.from_pinned_artifact(),
            )
        ),
        **rapid_arguments,
    )
    return OutputConfidenceRecalibrationProcessor(
        processor=processor,
        recalibrator=OutputConfidenceRecalibrator.from_pinned_artifact(),
    )


def _collect_ablation_activity(processor: Any) -> dict[str, int]:
    """Find one development component's aggregate counters without case data."""

    pending = [processor]
    visited: set[int] = set()
    while pending:
        component = pending.pop()
        identity = id(component)
        if identity in visited:
            continue
        visited.add(identity)
        activity_method = getattr(component, "ablation_activity", None)
        if callable(activity_method):
            activity = activity_method()
            if not isinstance(activity, Mapping):
                raise RuntimeError("ablation activity must be a mapping")
            normalized: dict[str, int] = {}
            for name, value in activity.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise RuntimeError(
                        "ablation activity requires names and non-negative integers"
                    )
                normalized[name] = value
            return normalized
        for attribute in (
            "processor",
            "_processor",
            "_primary_extractor",
            "_renderer",
            "_delegate",
        ):
            child = getattr(component, attribute, None)
            if child is not None:
                pending.append(child)
    return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_tree_sha256(input_dir: Path) -> tuple[str, int]:
    pdfs = tuple(
        sorted(
            (
                path
                for path in Path(input_dir).iterdir()
                if path.is_file() and path.suffix.casefold() == ".pdf"
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )
    digest = hashlib.sha256()
    for path in pdfs:
        name_bytes = path.name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest(), len(pdfs)


def _cpu_seconds() -> float:
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        self_usage.ru_utime
        + self_usage.ru_stime
        + child_usage.ru_utime
        + child_usage.ru_stime
    )


def _peak_memory_mib() -> float:
    usages = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    # Linux reports KiB; macOS reports bytes.
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return max(float(value) / divisor for value in usages)


def _tool_versions() -> dict[str, str]:
    version = "unavailable"
    try:
        completed = subprocess.run(
            ["tesseract", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
        first_line = completed.stdout.splitlines()
        if completed.returncode == 0 and first_line:
            version = first_line[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tesseract": version,
    }


def run_variant(
    *,
    variant_id: str,
    benchmark_id: str,
    source_revision: str,
    repeat_index: int,
    input_dir: Path,
    predictions_path: Path,
    observation_path: Path,
    max_workers: int = 4,
    processor_factory: Callable[[str], Any] = build_ablation_processor,
) -> dict[str, Any]:
    """Execute one label-blind ablation repetition and record aggregate metrics."""

    if not benchmark_id.strip() or not source_revision.strip():
        raise AblationConfigurationError(
            "benchmark_id and source_revision must be non-empty"
        )
    if repeat_index < 1:
        raise AblationConfigurationError("repeat_index must be positive")
    if not 1 <= max_workers <= 4:
        raise AblationConfigurationError("max_workers must be between 1 and 4")
    variant = variant_by_id(variant_id)
    config = BASELINE_CONFIG if variant is None else variant.config
    input_tree_sha256, input_pdf_count = _input_tree_sha256(input_dir)
    predictions_path = Path(predictions_path)
    observation_path = Path(observation_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    observation_path.parent.mkdir(parents=True, exist_ok=True)

    from mib_pipeline import BatchRunner

    cpu_before = _cpu_seconds()
    wall_before = time.monotonic()
    processor = processor_factory(variant_id)
    batch = BatchRunner(
        processor,
        max_workers=max_workers,
    ).run(input_dir, predictions_path)
    wall_seconds = time.monotonic() - wall_before
    cpu_seconds = _cpu_seconds() - cpu_before
    if cpu_seconds <= 0.0 or wall_seconds <= 0.0:
        raise RuntimeError("runtime clock did not advance")
    observation = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "variant_id": variant_id,
        "repeat_index": repeat_index,
        "source_revision": source_revision,
        "config_sha256": config_sha256(config),
        "input_tree_sha256": input_tree_sha256,
        "input_pdf_count": input_pdf_count,
        "max_workers": max_workers,
        "predictions_path": str(predictions_path.resolve()),
        "predictions_sha256": _sha256_file(predictions_path),
        "attempted": batch.attempted,
        "answered": batch.answered,
        "omitted": batch.omitted,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "peak_memory_mib": _peak_memory_mib(),
        "metrics_source": (
            "fresh_process_rusage_self_plus_waited_children_and_monotonic_wall"
        ),
        "tool_versions": _tool_versions(),
        "activity_counts": _collect_ablation_activity(processor),
    }
    observation_path.write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return observation


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationConfigurationError(f"cannot read JSON {path}: {exc}") from exc


def _validate_observation(path: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise AblationConfigurationError(f"{path}: unsupported observation schema")
    required = {
        "benchmark_id",
        "variant_id",
        "repeat_index",
        "source_revision",
        "config_sha256",
        "input_tree_sha256",
        "input_pdf_count",
        "max_workers",
        "predictions_path",
        "predictions_sha256",
        "attempted",
        "answered",
        "omitted",
        "cpu_seconds",
        "wall_seconds",
        "metrics_source",
    }
    missing = sorted(required - set(value))
    if missing:
        raise AblationConfigurationError(
            f"{path}: observation is missing {', '.join(missing)}"
        )
    variant = variant_by_id(str(value["variant_id"]))
    expected_config = BASELINE_CONFIG if variant is None else variant.config
    if value["config_sha256"] != config_sha256(expected_config):
        raise AblationConfigurationError(f"{path}: config hash does not match variant")
    prediction_path = Path(str(value["predictions_path"]))
    if not prediction_path.is_file():
        raise AblationConfigurationError(
            f"{path}: predictions file is unavailable: {prediction_path}"
        )
    if value["predictions_sha256"] != _sha256_file(prediction_path):
        raise AblationConfigurationError(f"{path}: predictions hash mismatch")
    numeric_positive = ("cpu_seconds", "wall_seconds")
    for key in numeric_positive:
        value_number = value[key]
        if (
            isinstance(value_number, bool)
            or not isinstance(value_number, (int, float))
            or float(value_number) <= 0.0
        ):
            raise AblationConfigurationError(f"{path}: {key} must be positive")
    activity_counts = value.get("activity_counts", {})
    if not isinstance(activity_counts, dict):
        raise AblationConfigurationError(
            f"{path}: activity_counts must be a mapping"
        )
    for name, count in activity_counts.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise AblationConfigurationError(
                f"{path}: activity counters require names and non-negative integers"
            )
    if (
        str(value["variant_id"]) == "with_checked_fee_option_recovery"
        and set(activity_counts) != set(CHECKBOX_ACTIVITY_COUNTERS)
    ):
        raise AblationConfigurationError(
            f"{path}: checkbox observation requires the fixed activity counters"
        )
    return dict(value)


def _evaluate_submission(
    *,
    repo_root: Path,
    truth_path: Path,
    submission_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluator = Path(repo_root) / "scripts" / "evaluate.py"
    if not evaluator.is_file():
        raise AblationConfigurationError(f"official evaluator not found: {evaluator}")
    with tempfile.TemporaryDirectory(prefix="mib-ocr-ablation-score-") as directory:
        aggregate_path = Path(directory) / "aggregate.json"
        cases_path = Path(directory) / "cases.jsonl"
        completed = subprocess.run(
            [
                sys.executable,
                str(evaluator),
                "--truth",
                str(Path(truth_path).resolve()),
                "--submission",
                str(Path(submission_path).resolve()),
                "--output-json",
                str(aggregate_path),
                "--case-scores-jsonl",
                str(cases_path),
            ],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode not in {0, 2} or not aggregate_path.is_file():
            raise AblationConfigurationError(
                "official evaluator failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        aggregate = _read_json(aggregate_path)
        cases = [
            json.loads(line)
            for line in cases_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return aggregate, cases


def _field_totals(cases: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    totals = {
        field: {"points": 0.0, "max_points": 0.0, "matched": 0, "scorable": 0}
        for field in PRIORITY_FIELDS
    }
    for case in cases:
        results = case.get("field_results", {})
        for field in PRIORITY_FIELDS:
            value = results.get(field, {})
            maximum = float(value.get("max_points", 0.0))
            points = float(value.get("points", 0.0))
            totals[field]["points"] += points
            totals[field]["max_points"] += maximum
            if maximum > 0.0:
                totals[field]["scorable"] += 1
                totals[field]["matched"] += int(value.get("status") == "matched")
    return totals


def _invalid_count(aggregate: Mapping[str, Any]) -> int:
    counts = aggregate["counts"]
    return sum(
        int(counts.get(name, 0))
        for name in (
            "extra_cases",
            "duplicate_case_ids",
            "blank_case_rows",
            "invalid_adjudication_records",
            "invalid_confidence_records",
            "invalid_fee_status_records",
        )
    )


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(tuple(float(value) for value in values)))


def _group_observations(
    observation_paths: Sequence[Path],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for path in observation_paths:
        observation = _validate_observation(path, _read_json(path))
        identity = (
            str(observation["variant_id"]),
            int(observation["repeat_index"]),
        )
        if identity in seen:
            raise AblationConfigurationError(
                f"duplicate variant/repeat observation: {identity}"
            )
        seen.add(identity)
        grouped.setdefault(identity[0], []).append(observation)
    for observations in grouped.values():
        observations.sort(key=lambda item: int(item["repeat_index"]))
    return grouped


def _group_consistency(
    observations: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> bool:
    return all(
        len({json.dumps(item.get(key), sort_keys=True) for item in observations}) == 1
        for key in keys
    )


def build_report(
    *,
    repo_root: Path,
    truth_path: Path,
    observation_paths: Sequence[Path],
) -> dict[str, Any]:
    """Score observations, prove determinism, and rank only safe techniques."""

    grouped = _group_observations(observation_paths)
    if "baseline" not in grouped:
        raise AblationConfigurationError("at least one baseline observation is required")
    all_observations = [item for group in grouped.values() for item in group]
    consistency_keys = (
        "benchmark_id",
        "source_revision",
        "input_tree_sha256",
        "input_pdf_count",
        "max_workers",
        "metrics_source",
    )
    if not _group_consistency(all_observations, consistency_keys):
        raise AblationConfigurationError(
            "all observations must share benchmark, revision, input, workers, and metrics source"
        )

    scored: dict[str, dict[str, Any]] = {}
    for variant_id, observations in grouped.items():
        aggregates: list[dict[str, Any]] = []
        field_totals: list[dict[str, dict[str, float]]] = []
        for observation in observations:
            aggregate, cases = _evaluate_submission(
                repo_root=repo_root,
                truth_path=truth_path,
                submission_path=Path(str(observation["predictions_path"])),
            )
            aggregates.append(aggregate)
            field_totals.append(_field_totals(cases))
        deterministic = (
            len(observations) >= MINIMUM_DETERMINISM_REPEATS
            and len({item["predictions_sha256"] for item in observations}) == 1
            and len(
                {
                    json.dumps(
                        item.get("activity_counts", {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for item in observations
                }
            )
            == 1
            and len(
                {
                    json.dumps(aggregate, sort_keys=True)
                    for aggregate in aggregates
                }
            )
            == 1
        )
        aggregate = aggregates[0]
        counts = aggregate["counts"]
        complete = bool(
            all(int(item["omitted"]) == 0 for item in observations)
            and int(counts["missing_cases"]) == 0
            and int(counts["scored_predictions"]) == int(counts["truth_cases"])
            and _invalid_count(aggregate) == 0
        )
        scored[variant_id] = {
            "observations": observations,
            "aggregates": aggregates,
            "aggregate": aggregate,
            "field_totals": field_totals[0],
            "repeat_count": len(observations),
            "deterministic": deterministic,
            "complete": complete,
            "cpu_seconds_median": _median(
                item["cpu_seconds"] for item in observations
            ),
            "wall_seconds_median": _median(
                item["wall_seconds"] for item in observations
            ),
            "peak_memory_mib_max": max(
                float(item.get("peak_memory_mib", 0.0)) for item in observations
            ),
            "activity_counts": dict(observations[0].get("activity_counts", {})),
        }

    baseline = scored["baseline"]
    baseline_score = float(baseline["aggregate"]["scores"]["total_score"])
    baseline_catastrophic = int(
        baseline["aggregate"]["raw"]["catastrophic_false_approvals"]
    )
    variants_by_id = {
        variant.variant_id: variant for variant in registered_variants()
    }
    entries: list[dict[str, Any]] = []
    for variant in registered_variants():
        measured = scored.get(variant.variant_id)
        if measured is None:
            entries.append(
                {
                    **variant.to_dict(),
                    "evidence_status": "not_measured",
                    "recommendation_eligible": False,
                }
            )
            continue
        variant_score = float(measured["aggregate"]["scores"]["total_score"])
        raw_variant_gain = variant_score - baseline_score
        if variant.technique_enabled_in == "baseline":
            technique_gain = -raw_variant_gain
            enabled = baseline
            disabled = measured
        else:
            technique_gain = raw_variant_gain
            enabled = measured
            disabled = baseline
        enabled_cpu = float(enabled["cpu_seconds_median"])
        disabled_cpu = float(disabled["cpu_seconds_median"])
        incremental_cpu = enabled_cpu - disabled_cpu
        enabled_catastrophic = int(
            enabled["aggregate"]["raw"]["catastrophic_false_approvals"]
        )
        disabled_catastrophic = int(
            disabled["aggregate"]["raw"]["catastrophic_false_approvals"]
        )
        safety_pass = bool(
            enabled["complete"]
            and disabled["complete"]
            and enabled_catastrophic <= disabled_catastrophic
        )
        deterministic = bool(enabled["deterministic"] and disabled["deterministic"])
        field_deltas: dict[str, float] = {}
        for field in variant.target_fields:
            enabled_points = float(enabled["field_totals"][field]["points"])
            disabled_points = float(disabled["field_totals"][field]["points"])
            field_deltas[field] = enabled_points - disabled_points
        evidence_status = (
            "measured"
            if deterministic and enabled["complete"] and disabled["complete"]
            else "insufficient_evidence"
        )
        eligible = bool(
            evidence_status == "measured"
            and safety_pass
            and technique_gain > 0.0
        )
        entries.append(
            {
                **variant.to_dict(),
                "evidence_status": evidence_status,
                "repeat_count": measured["repeat_count"],
                "deterministic": deterministic,
                "complete": bool(enabled["complete"] and disabled["complete"]),
                "safety_pass": safety_pass,
                "baseline_score": baseline_score,
                "variant_score": variant_score,
                "raw_variant_score_gain": raw_variant_gain,
                "technique_score_gain": technique_gain,
                "technique_cpu_seconds": enabled_cpu,
                "incremental_cpu_seconds": incremental_cpu,
                "score_gain_per_cpu_second": technique_gain / enabled_cpu,
                "incremental_score_gain_per_cpu_second": (
                    technique_gain / incremental_cpu
                    if incremental_cpu > 0.0
                    else None
                ),
                "baseline_cpu_seconds": baseline["cpu_seconds_median"],
                "variant_cpu_seconds": measured["cpu_seconds_median"],
                "baseline_wall_seconds": baseline["wall_seconds_median"],
                "variant_wall_seconds": measured["wall_seconds_median"],
                "baseline_peak_memory_mib": baseline["peak_memory_mib_max"],
                "variant_peak_memory_mib": measured["peak_memory_mib_max"],
                "enabled_catastrophic_false_approvals": enabled_catastrophic,
                "disabled_catastrophic_false_approvals": disabled_catastrophic,
                "target_field_raw_point_deltas": field_deltas,
                "enabled_activity_counts": enabled["activity_counts"],
                "disabled_activity_counts": disabled["activity_counts"],
                "recommendation_eligible": eligible,
            }
        )

    ranked = sorted(
        (
            entry
            for entry in entries
            if entry.get("recommendation_eligible")
        ),
        key=lambda entry: (
            -float(entry["score_gain_per_cpu_second"]),
            -float(entry["technique_score_gain"]),
            entry["variant_id"],
        ),
    )
    rank_by_id = {
        entry["variant_id"]: rank for rank, entry in enumerate(ranked, start=1)
    }
    for entry in entries:
        entry["recommendation_rank"] = rank_by_id.get(entry["variant_id"])

    baseline_aggregate = baseline["aggregate"]
    return {
        "report_version": REPORT_VERSION,
        "benchmark_context": (
            "public_development_ablation_not_unseen_not_official_leaderboard"
        ),
        "measurement_definition": {
            "raw_variant_score_gain": "variant total score minus baseline total score",
            "technique_score_gain": (
                "score with the technique enabled minus score with it disabled"
            ),
            "score_gain_per_cpu_second": (
                "technique_score_gain divided by median CPU seconds of the "
                "technique-enabled full run"
            ),
            "incremental_score_gain_per_cpu_second": (
                "technique_score_gain divided by positive median incremental CPU "
                "cost; null when runtime noise makes the denominator non-positive"
            ),
        },
        "benchmark_id": all_observations[0]["benchmark_id"],
        "source_revision": all_observations[0]["source_revision"],
        "truth_sha256": _sha256_file(truth_path),
        "input_tree_sha256": all_observations[0]["input_tree_sha256"],
        "input_pdf_count": all_observations[0]["input_pdf_count"],
        "max_workers": all_observations[0]["max_workers"],
        "baseline": {
            "repeat_count": baseline["repeat_count"],
            "deterministic": baseline["deterministic"],
            "complete": baseline["complete"],
            "total_score": baseline_score,
            "cpu_seconds_median": baseline["cpu_seconds_median"],
            "wall_seconds_median": baseline["wall_seconds_median"],
            "peak_memory_mib_max": baseline["peak_memory_mib_max"],
            "catastrophic_false_approvals": baseline_catastrophic,
            "counts": baseline_aggregate["counts"],
        },
        "variants": entries,
        "ranked_recommendations": [
            {
                "rank": rank_by_id[entry["variant_id"]],
                "variant_id": entry["variant_id"],
                "technique": entry["technique"],
                "target_fields": entry["target_fields"],
                "technique_score_gain": entry["technique_score_gain"],
                "score_gain_per_cpu_second": entry[
                    "score_gain_per_cpu_second"
                ],
                "incremental_cpu_seconds": entry["incremental_cpu_seconds"],
                "safety_pass": entry["safety_pass"],
            }
            for entry in ranked
        ],
        "limitations": [
            (
                "The report measures a public development partition and makes no "
                "unseen/private/leaderboard claim."
            ),
            (
                "Removal ablations estimate the marginal contribution of an "
                "existing technique; interactions between techniques are not "
                "identified."
            ),
            (
                "Only deterministic, complete, non-catastrophic techniques with "
                "positive measured gain are recommendation-eligible."
            ),
            (
                "This development harness does not integrate a technique into "
                "the submitted production runtime."
            ),
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render an aggregate-only, reviewable ablation report."""

    baseline = report["baseline"]
    lines = [
        "# Bounded OCR ablation report",
        "",
        (
            f"Benchmark `{report['benchmark_id']}` at source revision "
            f"`{report['source_revision']}`."
        ),
        "",
        (
            "**Evidence class:** public development ablation; this is neither an "
            "unseen/private measurement nor an official leaderboard result."
        ),
        "",
        "## Baseline",
        "",
        (
            f"- Score: {float(baseline['total_score']):.6f}/150; "
            f"CPU: {float(baseline['cpu_seconds_median']):.3f}s median; "
            f"wall: {float(baseline['wall_seconds_median']):.3f}s median."
        ),
        (
            f"- Repetitions: {baseline['repeat_count']}; deterministic: "
            f"{str(bool(baseline['deterministic'])).lower()}; complete: "
            f"{str(bool(baseline['complete'])).lower()}; catastrophic false "
            f"approvals: {baseline['catastrophic_false_approvals']}."
        ),
        (
            f"- Input PDFs: {report['input_pdf_count']}; input tree SHA-256: "
            f"`{report['input_tree_sha256']}`; truth SHA-256: "
            f"`{report['truth_sha256']}`."
        ),
        "",
        "## Ranked measured techniques",
        "",
        (
            "| Rank | Technique | Target fields | Score gain | CPU seconds | "
            "Gain / CPU second | Incremental CPU | Safety |"
        ),
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    ranked = report.get("ranked_recommendations", [])
    if not ranked:
        lines.append(
            "| — | No technique has sufficient positive evidence yet | — | — | — | — | — | — |"
        )
    else:
        entries = {
            entry["variant_id"]: entry for entry in report.get("variants", [])
        }
        for recommendation in ranked:
            entry = entries[recommendation["variant_id"]]
            incremental = entry.get("incremental_cpu_seconds")
            incremental_text = (
                f"{float(incremental):.3f}s"
                if incremental is not None
                else "n/a"
            )
            lines.append(
                "| {rank} | {technique} | {fields} | {gain:.6f} | {cpu:.3f}s | "
                "{efficiency:.9f} | {incremental} | pass |".format(
                    rank=recommendation["rank"],
                    technique=recommendation["technique"],
                    fields=", ".join(recommendation["target_fields"]),
                    gain=float(recommendation["technique_score_gain"]),
                    cpu=float(entry["technique_cpu_seconds"]),
                    efficiency=float(
                        recommendation["score_gain_per_cpu_second"]
                    ),
                    incremental=incremental_text,
                )
            )

    lines.extend(
        [
            "",
            "## Complete variant ledger",
            "",
            "| Variant | One changed variable | Evidence | Deterministic | Score effect | Safety |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for entry in report.get("variants", []):
        gain = entry.get("technique_score_gain")
        gain_text = f"{float(gain):.6f}" if gain is not None else "not measured"
        lines.append(
            "| `{variant}` | `{variable}` | {status} | {deterministic} | "
            "{gain} | {safety} |".format(
                variant=entry["variant_id"],
                variable=entry["changed_variable"],
                status=entry["evidence_status"],
                deterministic=(
                    str(bool(entry.get("deterministic"))).lower()
                    if entry["evidence_status"] != "not_measured"
                    else "not measured"
                ),
                gain=gain_text,
                safety=(
                    "pass"
                    if entry.get("safety_pass")
                    else (
                        "fail"
                        if entry["evidence_status"] != "not_measured"
                        else "not measured"
                    )
                ),
            )
        )
    activity_entries = [
        entry
        for entry in report.get("variants", [])
        if entry.get("enabled_activity_counts")
    ]
    if activity_entries:
        lines.extend(
            [
                "",
                "## Aggregate route activity",
                "",
                "| Variant | Technique-enabled activity counters |",
                "|---|---|",
            ]
        )
        for entry in activity_entries:
            counters = ", ".join(
                f"`{name}={count}`"
                for name, count in sorted(
                    entry["enabled_activity_counts"].items()
                )
            )
            lines.append(f"| `{entry['variant_id']}` | {counters} |")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    if ranked:
        winner = ranked[0]
        lines.append(
            f"Carry `{winner['variant_id']}` into the recovery implementation "
            f"review first: its enabled technique gained "
            f"{float(winner['technique_score_gain']):.6f} points at "
            f"{float(winner['score_gain_per_cpu_second']):.9f} points per "
            "full-run CPU second without a catastrophic-false-approval increase."
        )
    else:
        lines.append(
            "No implementation recommendation is justified until at least one "
            "variant has two byte-identical, complete repetitions with positive "
            "score gain and no catastrophic-false-approval increase."
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(
        [
            "",
            "The JSON report is authoritative; this Markdown is a deterministic "
            "aggregate rendering and contains no per-case identifiers.",
            "",
        ]
    )
    return "\n".join(lines)
