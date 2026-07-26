import math
import unittest
from pathlib import Path

from mib_pipeline.extraction import (
    CandidateEvidence,
    EvidenceType,
    TesseractOcrEngine,
    VisibleEvidenceExtractor,
    stable_ocr_engine_id,
)
from mib_pipeline.ingestion import Rect, RenderedCase
from mib_pipeline.provenance import (
    CoordinateTransform,
    PhysicalObservation,
    make_ocr_provenance,
    merge_ocr_provenance,
)


SOURCE_SHA256 = "a" * 64


class ProvenanceTests(unittest.TestCase):
    def test_crop_translation_maps_view_box_to_physical_page(self):
        provenance = make_ocr_provenance(
            source_sha256=SOURCE_SHA256,
            page_index=2,
            view_box=Rect(10, 20, 30, 45),
            applicant_scope="  Ada Nova  ",
            route_id=" sparse_intake_retry ",
            engine_id=" fake:psm6 ",
            view_id=" sparse_intake_crop ",
            transform=CoordinateTransform.crop_translation(
                left=100,
                upper=75,
            ),
        )

        self.assertEqual(
            provenance.observation.box,
            Rect(110, 95, 130, 120),
        )
        self.assertEqual(provenance.observation.applicant_scope, "Ada Nova")
        self.assertEqual(provenance.route_id, "sparse_intake_retry")
        self.assertEqual(provenance.engine_id, "fake:psm6")
        self.assertEqual(provenance.view_id, "sparse_intake_crop")

    def test_inverse_quarter_turn_maps_rotated_views_to_source_page(self):
        ninety = CoordinateTransform.inverse_quarter_turn(
            angle_degrees=90,
            source_width=200,
            source_height=100,
        )
        two_seventy = CoordinateTransform.inverse_quarter_turn(
            angle_degrees=270,
            source_width=200,
            source_height=100,
        )

        self.assertEqual(
            ninety.apply_box(Rect(20, 30, 50, 70)),
            Rect(130, 20, 170, 50),
        )
        self.assertEqual(
            two_seventy.apply_box(Rect(20, 30, 50, 70)),
            Rect(30, 50, 70, 80),
        )
        self.assertEqual(
            ninety.apply_box(Rect(0, 0, 100, 200)),
            Rect(0, 0, 200, 100),
        )
        self.assertEqual(
            two_seventy.apply_box(Rect(0, 0, 100, 200)),
            Rect(0, 0, 200, 100),
        )

    def test_fingerprint_is_canonical_and_merge_is_order_independent(self):
        first = make_ocr_provenance(
            source_sha256=SOURCE_SHA256.upper(),
            page_index=0,
            view_box=Rect(1, 2, 3, 4),
            applicant_scope="Ada",
            route_id=" primary ",
            engine_id=" engine ",
            view_id=" page ",
        )
        equivalent = make_ocr_provenance(
            source_sha256=SOURCE_SHA256,
            page_index=0,
            view_box=Rect(1, 2, 3, 4),
            applicant_scope=" Ada ",
            route_id="primary",
            engine_id="engine",
            view_id="page",
        )
        other = make_ocr_provenance(
            source_sha256=SOURCE_SHA256,
            page_index=0,
            view_box=Rect(5, 6, 7, 8),
            applicant_scope="Ada",
            route_id="refinement",
            engine_id="engine-2",
            view_id="page",
        )

        self.assertEqual(first.fingerprint, equivalent.fingerprint)
        self.assertEqual(
            merge_ocr_provenance((other, first), (equivalent,)),
            merge_ocr_provenance((equivalent,), (first, other)),
        )
        self.assertEqual(
            len(merge_ocr_provenance((first, equivalent, other),)),
            2,
        )

    def test_non_finite_boxes_and_control_character_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            PhysicalObservation(
                source_sha256=SOURCE_SHA256,
                page_index=0,
                box=Rect(0, 0, math.inf, 1),
            )
        with self.assertRaises(ValueError):
            make_ocr_provenance(
                source_sha256=SOURCE_SHA256,
                page_index=0,
                view_box=Rect(0, 0, 1, 1),
                applicant_scope=None,
                route_id="primary\nsecondary",
                engine_id="engine",
                view_id="page",
            )

    def test_candidate_rejects_cross_page_or_cross_document_provenance(self):
        page_zero = make_ocr_provenance(
            source_sha256=SOURCE_SHA256,
            page_index=0,
            view_box=Rect(0, 0, 1, 1),
            applicant_scope=None,
            route_id="primary",
            engine_id="engine",
            view_id="page",
        )
        page_one = make_ocr_provenance(
            source_sha256=SOURCE_SHA256,
            page_index=1,
            view_box=Rect(0, 0, 1, 1),
            applicant_scope=None,
            route_id="primary",
            engine_id="engine",
            view_id="page",
        )
        other_source = make_ocr_provenance(
            source_sha256="b" * 64,
            page_index=0,
            view_box=Rect(0, 0, 1, 1),
            applicant_scope=None,
            route_id="primary",
            engine_id="engine",
            view_id="page",
        )

        with self.assertRaises(ValueError):
            self._candidate((page_one,))
        with self.assertRaises(ValueError):
            self._candidate((page_zero, other_source))

    def test_extractor_adds_real_provenance_but_not_to_non_ocr_sources(self):
        extractor = VisibleEvidenceExtractor(
            ocr_engine=FakeEngine(),
            psm6_refinement=False,
            consensus_retry=False,
            fee_receipt_retry=False,
            sparse_intake_retry=False,
            orientation_retry=False,
            trusted_scope_repair=False,
            risk_flag_retry=False,
        )
        rendered_case = RenderedCase(
            source_path=Path("ignored.pdf"),
            source_sha256=SOURCE_SHA256,
            case_id="MIB-000001",
            pages=(),
            text_layer=(),
        )
        candidate = self._candidate(())

        attached = extractor._attach_ocr_provenance(
            candidate,
            rendered_case,
        )
        non_ocr = extractor._attach_ocr_provenance(
            self._candidate((), source="text_layer"),
            rendered_case,
        )

        self.assertEqual(len(attached.ocr_provenance), 1)
        self.assertEqual(
            attached.ocr_provenance[0].observation.source_sha256,
            SOURCE_SHA256,
        )
        self.assertEqual(attached.ocr_provenance[0].engine_id, "fake:engine")
        self.assertEqual(non_ocr.ocr_provenance, ())

    def test_engine_identity_never_uses_instance_repr(self):
        self.assertEqual(stable_ocr_engine_id(FakeEngine()), "fake:engine")
        first = stable_ocr_engine_id(UndeclaredEngine())
        second = stable_ocr_engine_id(UndeclaredEngine())
        self.assertEqual(first, second)
        self.assertNotIn("0x", first)
        self.assertNotEqual(
            stable_ocr_engine_id(TesseractOcrEngine(binary="tesseract")),
            stable_ocr_engine_id(
                TesseractOcrEngine(binary="/opt/legacy/tesseract-legacy")
            ),
        )

    @staticmethod
    def _candidate(
        provenance,
        *,
        source="visible_ocr",
    ):
        return CandidateEvidence(
            field_name="visa_class",
            value="XW-1",
            evidence_type=EvidenceType.INTAKE_FORM,
            page_index=0,
            box=Rect(10, 20, 30, 40),
            legible=True,
            superseded=False,
            ocr_confidence=0.9,
            source=source,
            ocr_provenance=provenance,
        )


class FakeEngine:
    provenance_id = "fake:engine"

    def read_page(self, _page):
        return ()


class UndeclaredEngine:
    pass


if __name__ == "__main__":
    unittest.main()
