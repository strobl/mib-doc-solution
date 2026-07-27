from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devtools import layout_manifest_freezer as freezer


class _FakeRaster:
    def __init__(self, ink_pixels, events, label):
        self._ink_pixels = ink_pixels
        self._events = events
        self._label = label

    def copy(self):
        self._events.append(f"{self._label}.copy")
        return _FakeRaster(self._ink_pixels, self._events, "rendered")

    def convert(self, mode):
        self._events.append(f"{self._label}.convert:{mode}")
        return _FakeRaster(self._ink_pixels, self._events, "converted")

    def resize(self, size, resampling):
        self._events.append(
            f"{self._label}.resize:{size[0]}x{size[1]}:{resampling}"
        )
        return _FakeRaster(self._ink_pixels, self._events, "resized")

    def histogram(self):
        histogram = [0] * 256
        histogram[0] = self._ink_pixels
        histogram[255] = (
            freezer.RENDER_WIDTH * freezer.RENDER_HEIGHT - self._ink_pixels
        )
        return histogram

    def close(self):
        self._events.append(f"{self._label}.close")


class _FakeBitmap:
    def __init__(self, events, ink_pixels):
        self._events = events
        self._ink_pixels = ink_pixels

    def to_pil(self):
        self._events.append("bitmap.to_pil")
        return _FakeRaster(self._ink_pixels, self._events, "borrowed")

    def close(self):
        self._events.append("bitmap.close")


class _FakePage:
    def __init__(self, events, ink_pixels):
        self._events = events
        self._ink_pixels = ink_pixels

    def render(self, *, scale):
        self._events.append(f"page.render:{scale}")
        return _FakeBitmap(self._events, self._ink_pixels)

    def close(self):
        self._events.append("page.close")


class _FakeDocument:
    def __init__(self, events, page_count, ink_pixels):
        self._events = events
        self._page_count = page_count
        self._page = _FakePage(events, ink_pixels)

    def __len__(self):
        return self._page_count

    def __getitem__(self, index):
        self._events.append(f"document.getitem:{index}")
        if index != 0:
            raise AssertionError("only the first page may be rendered")
        return self._page

    def close(self):
        self._events.append("document.close")


class _FakePdfium:
    PYPDFIUM_INFO = "fake-pypdfium"
    PDFIUM_INFO = "fake-pdfium"

    def __init__(self, events, *, page_count=2, ink_pixels=3400):
        self._events = events
        self._page_count = page_count
        self._ink_pixels = ink_pixels

    def PdfDocument(self, path):
        self._events.append(f"document.open:{Path(path).suffix}")
        return _FakeDocument(
            self._events,
            self._page_count,
            self._ink_pixels,
        )


class _FakeImageModule:
    class Resampling:
        LANCZOS = "lanczos"


class LayoutManifestFreezerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_cases(self, case_numbers):
        paths = []
        for case_number in case_numbers:
            path = self.pdf_dir / f"MIB-{case_number:06d}.pdf"
            path.write_bytes(b"%PDF-test-fixture")
            paths.append(path)
        return tuple(paths)

    @staticmethod
    def _signature(pdf_path):
        number = int(pdf_path.stem[len("MIB-") :])
        return number, number

    @staticmethod
    def _versions():
        return {
            "pillow_version": "test-pillow",
            "pypdfium2_version": "test-pypdfium",
            "pdfium_version": "test-pdfium",
        }

    def _build(self, *, expected_count=5, split_seed="test-seed-v1"):
        with mock.patch.object(
            freezer,
            "layout_signature",
            side_effect=self._signature,
        ), mock.patch.object(
            freezer,
            "_rendering_version_metadata",
            side_effect=self._versions,
        ):
            return freezer.build_layout_manifest(
                self.pdf_dir,
                expected_count=expected_count,
                split_seed=split_seed,
            )

    def test_layout_signature_uses_only_page_count_and_first_page_pixels(self):
        events = []
        dependencies = (
            _FakeImageModule,
            _FakePdfium(events, page_count=2, ink_pixels=3400),
        )
        with mock.patch.object(
            freezer,
            "_rendering_dependencies",
            return_value=dependencies,
        ):
            page_count, ink_bucket = freezer.layout_signature(
                self.pdf_dir / "MIB-000001.pdf"
            )

        self.assertEqual(page_count, 2)
        self.assertEqual(ink_bucket, 5)
        self.assertEqual(
            events,
            [
                "document.open:.pdf",
                "document.getitem:0",
                "page.render:1",
                "bitmap.to_pil",
                "borrowed.copy",
                "borrowed.close",
                "bitmap.close",
                "page.close",
                "document.close",
                "rendered.convert:L",
                "converted.resize:128x166:lanczos",
                "resized.close",
                "converted.close",
                "rendered.close",
            ],
        )

    def test_layout_signature_rejects_pageless_pdf_and_closes_document(self):
        events = []
        dependencies = (
            _FakeImageModule,
            _FakePdfium(events, page_count=0),
        )
        with mock.patch.object(
            freezer,
            "_rendering_dependencies",
            return_value=dependencies,
        ):
            with self.assertRaisesRegex(
                freezer.LayoutManifestFreezeError,
                "no pages",
            ):
                freezer.layout_signature(
                    self.pdf_dir / "MIB-000001.pdf"
                )
        self.assertEqual(
            events,
            ["document.open:.pdf", "document.close"],
        )

    def test_discovery_requires_an_existing_directory(self):
        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "existing directory",
        ):
            freezer.discover_pdfs(
                self.root / "missing",
                expected_count=5,
            )
        regular_file = self.root / "regular-file"
        regular_file.write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "existing directory",
        ):
            freezer.discover_pdfs(regular_file, expected_count=5)

    def test_discovery_requires_exact_count(self):
        self._write_cases(range(1, 6))
        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "expected exactly 6 PDFs, found 5",
        ):
            freezer.discover_pdfs(self.pdf_dir, expected_count=6)

    def test_discovery_rejects_non_pdf_entries_and_subdirectories(self):
        self._write_cases(range(1, 6))
        (self.pdf_dir / "labels.csv").write_text(
            "case_id,label\n",
            encoding="utf-8",
        )
        (self.pdf_dir / "nested").mkdir()
        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "found 2 other entries",
        ):
            freezer.discover_pdfs(self.pdf_dir, expected_count=5)

    def test_discovery_rejects_noncanonical_case_ids(self):
        self._write_cases(range(1, 5))
        (self.pdf_dir / "mib-000005.pdf").write_bytes(b"%PDF")
        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "without a canonical case ID",
        ):
            freezer.discover_pdfs(self.pdf_dir, expected_count=5)

    def test_discovery_rejects_unicode_digit_lookalike_case_ids(self):
        self._write_cases(range(1, 5))
        (self.pdf_dir / "MIB-０００００５.pdf").write_bytes(b"%PDF")
        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "without a canonical case ID",
        ):
            freezer.discover_pdfs(self.pdf_dir, expected_count=5)

    def test_discovery_rejects_duplicate_case_ids_on_every_filesystem(self):
        entries = []
        for stem, suffix in (
            ("MIB-000001", ".pdf"),
            ("MIB-000001", ".PDF"),
            ("MIB-000002", ".pdf"),
            ("MIB-000003", ".pdf"),
            ("MIB-000004", ".pdf"),
        ):
            entry = mock.Mock(spec=Path)
            entry.is_symlink.return_value = False
            entry.is_file.return_value = True
            entry.suffix = suffix
            entry.stem = stem
            entries.append(entry)
        with mock.patch.object(
            Path,
            "iterdir",
            return_value=iter(entries),
        ):
            with self.assertRaisesRegex(
                freezer.LayoutManifestFreezeError,
                "case IDs must be unique",
            ):
                freezer.discover_pdfs(self.pdf_dir, expected_count=5)

    def test_discovery_sorts_by_canonical_case_id(self):
        self._write_cases((5, 2, 4, 1, 3))
        discovered = freezer.discover_pdfs(
            self.pdf_dir,
            expected_count=5,
        )
        self.assertEqual(
            tuple(path.stem for path in discovered),
            tuple(f"MIB-{number:06d}" for number in range(1, 6)),
        )

    def test_expected_count_must_be_an_integer_large_enough_for_folds(self):
        for value in (True, 4, 0, -1, 5.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    freezer.LayoutManifestFreezeError,
                    "integer of at least 5",
                ):
                    freezer.discover_pdfs(
                        self.pdf_dir,
                        expected_count=value,
                    )

    def test_manifest_is_fixed_shape_and_byte_deterministic(self):
        self._write_cases((5, 2, 4, 1, 3))
        first, first_groups = self._build()
        second, second_groups = self._build()

        self.assertEqual(first, second)
        self.assertEqual(first_groups, second_groups)
        self.assertEqual(first["schema"], freezer.SCHEMA)
        self.assertEqual(first["schema"], "mib-wo12-layout-groups/v2")
        self.assertEqual(first["repeats"], 3)
        self.assertEqual(first["folds"], 5)
        self.assertIs(first["label_blind_construction"], True)
        self.assertNotIn("frozen_before_scoring", first)
        self.assertEqual(first["split_seed"], "test-seed-v1")
        self.assertEqual(
            tuple(row["case_id"] for row in first["cases"]),
            tuple(f"MIB-{number:06d}" for number in range(1, 6)),
        )
        self.assertEqual(
            first["layout_signature"],
            {
                "first_page_grayscale_ink_bucket_width": 0.03,
                "first_page_grayscale_ink_pixel_threshold_exclusive": 210,
                "first_page_render_height": 166,
                "first_page_render_width": 128,
                "inputs": [
                    "pdf_page_count",
                    "first_page_rendered_pixels",
                ],
                "pillow_version": "test-pillow",
                "pdfium_version": "test-pdfium",
                "pypdfium2_version": "test-pypdfium",
                "version": "page-count-plus-first-page-ink-v1",
            },
        )
        self.assertEqual(
            freezer.canonical_json(first),
            freezer.canonical_json(second),
        )

    def test_manifest_requires_supplied_normalized_seed(self):
        self._write_cases(range(1, 6))
        for seed in ("", "   ", " leading", "trailing "):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(
                    freezer.LayoutManifestFreezeError,
                    "split seed",
                ):
                    self._build(split_seed=seed)

    def test_manifest_requires_at_least_five_groups(self):
        self._write_cases(range(1, 6))
        with mock.patch.object(
            freezer,
            "layout_signature",
            return_value=(1, 1),
        ):
            with self.assertRaisesRegex(
                freezer.LayoutManifestFreezeError,
                "at least 5 layout groups, found 1",
            ):
                freezer.build_layout_manifest(
                    self.pdf_dir,
                    expected_count=5,
                    split_seed="test-seed",
                )

    def test_render_failure_reports_only_canonical_position(self):
        self._write_cases(range(1, 6))
        with mock.patch.object(
            freezer,
            "layout_signature",
            side_effect=ValueError("MIB-999999 should not escape"),
        ):
            with self.assertRaises(
                freezer.LayoutManifestFreezeError,
            ) as captured:
                freezer.build_layout_manifest(
                    self.pdf_dir,
                    expected_count=5,
                    split_seed="test-seed",
                )
        message = str(captured.exception)
        self.assertIn("canonical position 1", message)
        self.assertNotIn("MIB-", message)

    def test_freeze_writes_canonical_json_newline_and_aggregate_summary(self):
        self._write_cases((5, 2, 4, 1, 3))
        output = self.root / "external" / "layout-manifest.json"
        output.parent.mkdir()
        with mock.patch.object(
            freezer,
            "layout_signature",
            side_effect=self._signature,
        ), mock.patch.object(
            freezer,
            "_rendering_version_metadata",
            side_effect=self._versions,
        ):
            summary = freezer.freeze_layout_manifest(
                self.pdf_dir,
                output,
                expected_count=5,
                split_seed="test-seed-v1",
            )

        content = output.read_bytes()
        self.assertTrue(content.endswith(b"\n"))
        self.assertFalse(content.endswith(b"\n\n"))
        parsed = json.loads(content)
        self.assertEqual(
            content,
            (freezer.canonical_json(parsed) + "\n").encode("utf-8"),
        )
        self.assertEqual(summary["case_count"], 5)
        self.assertEqual(summary["group_count"], 5)
        self.assertEqual(summary["repeats"], 3)
        self.assertEqual(summary["folds"], 5)
        self.assertEqual(
            summary["sha256"],
            hashlib.sha256(content).hexdigest(),
        )
        rendered_summary = freezer.canonical_json(summary)
        self.assertNotIn("MIB-", rendered_summary)
        self.assertNotIn("page-count-", rendered_summary)
        self.assertNotIn("ink-bucket-", rendered_summary)
        self.assertNotIn('"cases"', rendered_summary)
        self.assertNotIn('"case_id"', rendered_summary)

    def test_freeze_refuses_to_overwrite_or_contaminate_pdf_input(self):
        self._write_cases(range(1, 6))
        targets = (
            self.pdf_dir / "MIB-000001.pdf",
            self.pdf_dir / "layout-manifest.json",
            self.pdf_dir / "nested" / "layout-manifest.json",
        )
        for output in targets:
            with self.subTest(output=output):
                with self.assertRaisesRegex(
                    freezer.LayoutManifestFreezeError,
                    "outside the PDF input directory",
                ):
                    freezer.freeze_layout_manifest(
                        self.pdf_dir,
                        output,
                        expected_count=5,
                        split_seed="test-seed-v1",
                    )
        self.assertEqual(
            (self.pdf_dir / "MIB-000001.pdf").read_bytes(),
            b"%PDF-test-fixture",
        )

    def test_freeze_requires_identity_manifest_outside_repository(self):
        self._write_cases(range(1, 6))
        output = (
            freezer.REPO_ROOT
            / "MIB-999999-forbidden-layout-manifest.json"
        )
        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "outside the repository",
        ):
            freezer.freeze_layout_manifest(
                self.pdf_dir,
                output,
                expected_count=5,
                split_seed="test-seed-v1",
            )
        self.assertFalse(output.exists())

    def test_freeze_rejects_symlinked_input_root_before_it_can_swap(self):
        self._write_cases(range(1, 6))
        alias = self.root / "pdf-alias"
        try:
            alias.symlink_to(self.pdf_dir, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")
        external = self.root / "external"
        external.mkdir()
        output = external / "layout-manifest.json"

        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "stable non-symlink directory",
        ), mock.patch.object(
            freezer,
            "build_layout_manifest",
        ) as build:
            freezer.freeze_layout_manifest(
                alias,
                output,
                expected_count=5,
                split_seed="test-seed-v1",
            )

        build.assert_not_called()
        self.assertFalse(output.exists())

    def test_freeze_rejects_symlinked_output_parent(self):
        self._write_cases(range(1, 6))
        real_parent = self.root / "real-external"
        real_parent.mkdir()
        alias_parent = self.root / "external-alias"
        try:
            alias_parent.symlink_to(
                real_parent,
                target_is_directory=True,
            )
        except OSError:
            self.skipTest("symlinks unavailable")
        output = alias_parent / "layout-manifest.json"
        with mock.patch.object(
            freezer,
            "layout_signature",
            side_effect=self._signature,
        ), mock.patch.object(
            freezer,
            "_rendering_version_metadata",
            side_effect=self._versions,
        ), self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "must not contain symlinks",
        ):
            freezer.freeze_layout_manifest(
                self.pdf_dir,
                output,
                expected_count=5,
                split_seed="test-seed-v1",
            )
        self.assertFalse((real_parent / output.name).exists())

    def test_freeze_is_idempotent_and_rejects_a_changed_manifest(self):
        self._write_cases(range(1, 6))
        output = self.root / "external" / "layout-manifest.json"
        output.parent.mkdir()
        with mock.patch.object(
            freezer,
            "layout_signature",
            side_effect=self._signature,
        ), mock.patch.object(
            freezer,
            "_rendering_version_metadata",
            side_effect=self._versions,
        ):
            first_summary = freezer.freeze_layout_manifest(
                self.pdf_dir,
                output,
                expected_count=5,
                split_seed="frozen-seed-v1",
            )
            frozen_bytes = output.read_bytes()
            frozen_stat = output.stat()
            second_summary = freezer.freeze_layout_manifest(
                self.pdf_dir,
                output,
                expected_count=5,
                split_seed="frozen-seed-v1",
            )
            with self.assertRaisesRegex(
                freezer.LayoutManifestFreezeError,
                "already exists with different bytes",
            ):
                freezer.freeze_layout_manifest(
                    self.pdf_dir,
                    output,
                    expected_count=5,
                    split_seed="changed-seed-v2",
                )

        self.assertEqual(second_summary, first_summary)
        self.assertEqual(output.read_bytes(), frozen_bytes)
        current_stat = output.stat()
        self.assertEqual(current_stat.st_ino, frozen_stat.st_ino)
        self.assertEqual(current_stat.st_mtime_ns, frozen_stat.st_mtime_ns)

    def test_atomic_write_is_idempotent_without_replacing_existing_inode(self):
        output = self.root / "manifest.json"
        content = b'{"frozen":true}\n'
        output.write_bytes(content)
        original_stat = output.stat()

        freezer._atomic_write(output, content)

        current_stat = output.stat()
        self.assertEqual(output.read_bytes(), content)
        self.assertEqual(current_stat.st_ino, original_stat.st_ino)
        self.assertEqual(current_stat.st_mtime_ns, original_stat.st_mtime_ns)
        self.assertEqual(
            tuple(self.root.glob(".manifest.json.*.tmp")),
            (),
        )

    def test_atomic_write_rejects_conflicting_existing_bytes(self):
        output = self.root / "manifest.json"
        output.write_bytes(b"original\n")
        original_stat = output.stat()

        with self.assertRaisesRegex(
            freezer.LayoutManifestFreezeError,
            "already exists with different bytes",
        ):
            freezer._atomic_write(output, b"replacement\n")

        current_stat = output.stat()
        self.assertEqual(output.read_bytes(), b"original\n")
        self.assertEqual(current_stat.st_ino, original_stat.st_ino)
        self.assertEqual(current_stat.st_mtime_ns, original_stat.st_mtime_ns)
        self.assertEqual(
            tuple(self.root.glob(".manifest.json.*.tmp")),
            (),
        )

    def test_atomic_write_cleans_temporary_file_if_atomic_create_fails(self):
        output = self.root / "manifest.json"
        with mock.patch.object(
            freezer.os,
            "link",
            side_effect=OSError("simulated link failure"),
        ):
            with self.assertRaisesRegex(OSError, "simulated"):
                freezer._atomic_write(output, b"replacement\n")
        self.assertFalse(output.exists())
        self.assertEqual(
            tuple(self.root.glob(".manifest.json.*.tmp")),
            (),
        )

    def test_open_output_parent_closes_descriptor_on_metadata_race(self):
        output = self.root / "manifest.json"
        real_open = freezer.os.open
        opened_descriptors = []

        def record_open(*args, **kwargs):
            descriptor = real_open(*args, **kwargs)
            opened_descriptors.append(descriptor)
            return descriptor

        with mock.patch.object(
            freezer.os,
            "open",
            side_effect=record_open,
        ), mock.patch.object(
            freezer.os,
            "fstat",
            side_effect=OSError("simulated metadata race"),
        ), self.assertRaises(
            freezer.LayoutManifestFreezeError
        ):
            freezer._open_output_parent(output)

        self.assertEqual(len(opened_descriptors), 1)
        with self.assertRaises(OSError):
            freezer.os.fstat(opened_descriptors[0])

    def test_atomic_write_rolls_back_link_after_parent_is_moved(self):
        output_parent = self.root / "safe-output"
        output_parent.mkdir()
        output = output_parent / "manifest.json"
        relocated = self.pdf_dir / "relocated-output"
        real_link = freezer.os.link

        def link_then_move(*args, **kwargs):
            result = real_link(*args, **kwargs)
            output_parent.rename(relocated)
            return result

        with mock.patch.object(
            freezer.os,
            "link",
            side_effect=link_then_move,
        ), self.assertRaises(
            freezer.LayoutManifestFreezeError
        ):
            freezer._atomic_write(output, b'{"private":"manifest"}\n')

        self.assertTrue(relocated.is_dir())
        self.assertFalse((relocated / output.name).exists())
        self.assertEqual(tuple(relocated.iterdir()), ())

    def test_atomic_write_never_rolls_back_preexisting_exact_output(self):
        output_parent = self.root / "safe-output"
        output_parent.mkdir()
        output = output_parent / "manifest.json"
        content = b'{"existing":true}\n'
        output.write_bytes(content)
        relocated = self.pdf_dir / "relocated-output"
        original_verify = freezer._verify_parent_binding
        verification_count = 0

        def move_before_second_verification(descriptor, parent):
            nonlocal verification_count
            verification_count += 1
            if verification_count == 2:
                output_parent.rename(relocated)
            return original_verify(descriptor, parent)

        with mock.patch.object(
            freezer,
            "_verify_parent_binding",
            side_effect=move_before_second_verification,
        ), self.assertRaises(
            freezer.LayoutManifestFreezeError
        ):
            freezer._atomic_write(output, content)

        self.assertEqual(verification_count, 2)
        self.assertEqual((relocated / output.name).read_bytes(), content)
        self.assertEqual(
            tuple(relocated.glob(f".{output.name}.*.tmp")),
            (),
        )

    def test_cli_requires_expected_count_and_seed(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as captured:
                freezer.main(
                    [
                        "--pdf-dir",
                        str(self.pdf_dir),
                        "--output",
                        str(self.root / "manifest.json"),
                        "--expected-count",
                        "4",
                        "--split-seed",
                        "seed",
                    ]
                )
        self.assertEqual(captured.exception.code, 2)
        self.assertIn("must be at least 5", stderr.getvalue())

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as captured:
                freezer.main(
                    [
                        "--pdf-dir",
                        str(self.pdf_dir),
                        "--output",
                        str(self.root / "manifest.json"),
                        "--expected-count",
                        "5",
                    ]
                )
        self.assertEqual(captured.exception.code, 2)
        self.assertIn("--split-seed", stderr.getvalue())

    def test_cli_prints_one_canonical_aggregate_only_line(self):
        # Even a caller-controlled identity-bearing output path must not be
        # reflected into the retained stdout summary.
        output = self.root / "MIB-999999-layout-manifest.json"
        expected_summary = {
            "case_count": 1000,
            "folds": 5,
            "group_count": 9,
            "largest_group_count": 500,
            "repeats": 3,
            "schema": freezer.SCHEMA,
            "sha256": "a" * 64,
            "singleton_group_count": 0,
            "smallest_group_count": 1,
        }
        stdout = io.StringIO()
        with mock.patch.object(
            freezer,
            "freeze_layout_manifest",
            return_value=expected_summary,
        ) as freeze, contextlib.redirect_stdout(stdout):
            result = freezer.main(
                [
                    "--pdf-dir",
                    str(self.pdf_dir),
                    "--output",
                    str(output),
                    "--expected-count",
                    "1000",
                    "--split-seed",
                    "wo12-test-v1",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            stdout.getvalue(),
            freezer.canonical_json(expected_summary) + "\n",
        )
        freeze.assert_called_once_with(
            self.pdf_dir,
            output,
            expected_count=1000,
            split_seed="wo12-test-v1",
        )
        self.assertNotIn("MIB-", stdout.getvalue())
        self.assertNotIn("case_id", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
