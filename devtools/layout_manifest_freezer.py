#!/usr/bin/env python3
"""Freeze an external, label-blind layout manifest for grouped evaluation.

This is development-only WO-12 control infrastructure.  It deliberately reads
only PDF page count and first-page rendered pixels.  It never opens truth,
prediction, score, OCR, or extracted-text artifacts, and it is not imported by
the submitted runtime.

The identity-bearing manifest belongs in an external evaluation directory.
Only the aggregate summary emitted on stdout is suitable for retained evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


# WO-12 records label-blind construction, not a historical timing claim.
# The older WO-15 schema and loader remain in grouped_recovery_evidence.py.
SCHEMA = "mib-wo12-layout-groups/v2"
REPEATS = 3
FOLDS = 5
CASE_ID_PATTERN = re.compile(r"MIB-[0-9]{6}")
RENDER_WIDTH = 128
RENDER_HEIGHT = 166
INK_THRESHOLD = 210
INK_BUCKET_WIDTH = 0.03
SIGNATURE_VERSION = "page-count-plus-first-page-ink-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


class LayoutManifestFreezeError(RuntimeError):
    """The requested manifest could not be frozen without ambiguity."""


def canonical_json(value: object) -> str:
    """Return the repository's canonical, single-line JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rendering_dependencies() -> tuple[Any, Any]:
    try:
        from PIL import Image
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise LayoutManifestFreezeError(
            "Pillow and pypdfium2 are required to freeze layout manifests"
        ) from exc
    return Image, pdfium


def _rendering_version_metadata() -> Mapping[str, str]:
    _Image, pdfium = _rendering_dependencies()
    try:
        from PIL import __version__ as pillow_version
    except ImportError as exc:
        raise LayoutManifestFreezeError(
            "Pillow version metadata is unavailable"
        ) from exc
    return {
        "pillow_version": str(pillow_version),
        "pypdfium2_version": str(pdfium.PYPDFIUM_INFO),
        "pdfium_version": str(pdfium.PDFIUM_INFO),
    }


def layout_signature(pdf_path: Path) -> tuple[int, int]:
    """Derive a coarse signature from page count and first-page pixels only."""

    Image, pdfium = _rendering_dependencies()
    document = pdfium.PdfDocument(str(pdf_path))
    rendered = None
    try:
        page_count = len(document)
        if page_count < 1:
            raise LayoutManifestFreezeError("a PDF has no pages")
        page = document[0]
        try:
            bitmap = page.render(scale=1)
            try:
                # Copy before closing the PDFium bitmap so PIL owns its pixels.
                borrowed = bitmap.to_pil()
                try:
                    rendered = borrowed.copy()
                finally:
                    borrowed.close()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()

    converted = None
    resized = None
    try:
        converted = rendered.convert("L")
        resampling = getattr(Image, "Resampling", Image)
        resized = converted.resize(
            (RENDER_WIDTH, RENDER_HEIGHT),
            resampling.LANCZOS,
        )
        histogram = resized.histogram()
        pixel_count = RENDER_WIDTH * RENDER_HEIGHT
        ink_pixel_count = sum(histogram[:INK_THRESHOLD])
        ink_fraction = ink_pixel_count / pixel_count
        ink_bucket = math.floor(ink_fraction / INK_BUCKET_WIDTH)
        return page_count, ink_bucket
    finally:
        if resized is not None:
            resized.close()
        if converted is not None:
            converted.close()
        if rendered is not None:
            rendered.close()


def discover_pdfs(pdf_dir: Path, *, expected_count: int) -> tuple[Path, ...]:
    """Validate and return one canonical, PDF-only cohort."""

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise LayoutManifestFreezeError("PDF input must be an existing directory")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < FOLDS
    ):
        raise LayoutManifestFreezeError(
            f"expected PDF count must be an integer of at least {FOLDS}"
        )

    try:
        entries = tuple(pdf_dir.iterdir())
    except OSError as exc:
        raise LayoutManifestFreezeError(
            "PDF input directory could not be enumerated"
        ) from exc

    pdfs: list[Path] = []
    non_pdf_entry_count = 0
    for entry in entries:
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.suffix.casefold() != ".pdf"
        ):
            non_pdf_entry_count += 1
        else:
            pdfs.append(entry)
    if non_pdf_entry_count:
        raise LayoutManifestFreezeError(
            "PDF input directory must contain PDF files only; "
            f"found {non_pdf_entry_count} other entries"
        )
    if len(pdfs) != expected_count:
        raise LayoutManifestFreezeError(
            f"expected exactly {expected_count} PDFs, found {len(pdfs)}"
        )

    case_ids = tuple(pdf_path.stem for pdf_path in pdfs)
    invalid_count = sum(
        CASE_ID_PATTERN.fullmatch(case_id) is None for case_id in case_ids
    )
    if invalid_count:
        raise LayoutManifestFreezeError(
            f"found {invalid_count} PDFs without a canonical case ID"
        )
    if len(set(case_ids)) != len(case_ids):
        raise LayoutManifestFreezeError("PDF case IDs must be unique")

    return tuple(sorted(pdfs, key=lambda path: path.stem))


def _external_temporary_root() -> Path:
    for candidate in (Path("/private/tmp"), Path("/tmp")):
        try:
            resolved = candidate.resolve()
            resolved.relative_to(REPO_ROOT)
        except ValueError:
            if resolved.is_dir():
                return resolved
        except (OSError, RuntimeError, UnboundLocalError):
            continue
    raise LayoutManifestFreezeError(
        "an external temporary directory is required"
    )


def _copy_pdf_snapshot(source: Path, target: Path) -> None:
    """Stream one no-follow regular file into the controlled snapshot."""

    source_descriptor = -1
    target_descriptor = -1
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(
            str(source),
            os.O_RDONLY | nofollow,
        )
        initial = os.fstat(source_descriptor)
        entry = os.stat(source, follow_symlinks=False)
        if (
            not stat.S_ISREG(initial.st_mode)
            or not stat.S_ISREG(entry.st_mode)
            or (initial.st_dev, initial.st_ino)
            != (entry.st_dev, entry.st_ino)
        ):
            raise LayoutManifestFreezeError(
                "a PDF input changed during snapshot creation"
            )
        target_descriptor = os.open(
            str(target),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o400,
        )
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(target_descriptor, view)
                view = view[written:]
        os.fsync(target_descriptor)
        final = os.fstat(source_descriptor)
        final_entry = os.stat(source, follow_symlinks=False)
        initial_binding = (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
            initial.st_ctime_ns,
        )
        if initial_binding != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) or (final.st_dev, final.st_ino) != (
            final_entry.st_dev,
            final_entry.st_ino,
        ):
            raise LayoutManifestFreezeError(
                "a PDF input changed during snapshot creation"
            )
    except LayoutManifestFreezeError:
        raise
    except OSError as exc:
        raise LayoutManifestFreezeError(
            "a PDF input could not be snapshotted safely"
        ) from exc
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)


def build_layout_manifest(
    pdf_dir: Path,
    *,
    expected_count: int,
    split_seed: str,
) -> tuple[dict[str, object], Counter[str]]:
    """Build a deterministic manifest without serializing it."""

    if not isinstance(split_seed, str) or not split_seed.strip():
        raise LayoutManifestFreezeError("split seed must be non-empty")
    if split_seed != split_seed.strip():
        raise LayoutManifestFreezeError(
            "split seed must not have leading or trailing whitespace"
        )

    pdfs = discover_pdfs(pdf_dir, expected_count=expected_count)
    cases: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(
        prefix="mib-wo12-layout-snapshot-",
        dir=str(_external_temporary_root()),
    ) as temporary_name:
        snapshot_root = Path(temporary_name)
        for index, pdf_path in enumerate(pdfs):
            snapshot_path = snapshot_root / pdf_path.name
            try:
                _copy_pdf_snapshot(pdf_path, snapshot_path)
                page_count, ink_bucket = layout_signature(snapshot_path)
            except LayoutManifestFreezeError:
                raise
            except Exception as exc:
                # Report only an ordinal.  A case identity must not enter logs.
                raise LayoutManifestFreezeError(
                    f"could not render PDF at canonical position {index + 1}"
                ) from exc
            finally:
                try:
                    snapshot_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise LayoutManifestFreezeError(
                        "temporary PDF snapshot could not be removed"
                    ) from exc
            cases.append(
                {
                    "case_id": pdf_path.stem,
                    "layout_group": (
                        f"page-count-{page_count:02d}__"
                        f"ink-bucket-{ink_bucket:02d}"
                    ),
                }
            )

    group_sizes = Counter(row["layout_group"] for row in cases)
    if len(group_sizes) < FOLDS:
        raise LayoutManifestFreezeError(
            f"expected at least {FOLDS} layout groups, found {len(group_sizes)}"
        )

    try:
        versions = _rendering_version_metadata()
    except LayoutManifestFreezeError:
        raise
    except Exception as exc:
        raise LayoutManifestFreezeError(
            "rendering engine version metadata is unavailable"
        ) from exc
    manifest: dict[str, object] = {
        "cases": cases,
        "folds": FOLDS,
        "label_blind_construction": True,
        "layout_signature": {
            "first_page_grayscale_ink_bucket_width": INK_BUCKET_WIDTH,
            "first_page_grayscale_ink_pixel_threshold_exclusive": INK_THRESHOLD,
            "first_page_render_height": RENDER_HEIGHT,
            "first_page_render_width": RENDER_WIDTH,
            "inputs": ["pdf_page_count", "first_page_rendered_pixels"],
            "pillow_version": versions["pillow_version"],
            "pdfium_version": versions["pdfium_version"],
            "pypdfium2_version": versions["pypdfium2_version"],
            "version": SIGNATURE_VERSION,
        },
        "repeats": REPEATS,
        "schema": SCHEMA,
        "split_seed": split_seed,
    }
    return manifest, group_sizes


def _open_output_parent(path: Path) -> tuple[int, Path, str]:
    path = Path(path)
    parent_descriptor = -1
    try:
        resolved_parent = path.parent.resolve(strict=True)
        if path.parent.is_symlink():
            raise LayoutManifestFreezeError(
                "manifest output path must not contain symlinks"
            )
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(str(resolved_parent), flags)
        opened = os.fstat(parent_descriptor)
        current = os.stat(resolved_parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise LayoutManifestFreezeError(
                "manifest output directory binding changed"
            )
    except LayoutManifestFreezeError:
        if parent_descriptor >= 0:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass
        raise
    except (OSError, RuntimeError) as exc:
        if parent_descriptor >= 0:
            try:
                os.close(parent_descriptor)
            except OSError:
                pass
        raise LayoutManifestFreezeError(
            "manifest output directory must already exist safely"
        ) from exc
    return parent_descriptor, resolved_parent, path.name


def _verify_parent_binding(descriptor: int, parent: Path) -> None:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(parent, follow_symlinks=False)
    except OSError as exc:
        raise LayoutManifestFreezeError(
            "manifest output directory binding could not be reverified"
        ) from exc
    if (opened.st_dev, opened.st_ino) != (
        current.st_dev,
        current.st_ino,
    ):
        raise LayoutManifestFreezeError(
            "manifest output directory binding changed"
        )


def _read_output_at(
    parent_descriptor: int,
    name: str,
) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LayoutManifestFreezeError(
                "existing manifest output must be a regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    except LayoutManifestFreezeError:
        raise
    except OSError as exc:
        raise LayoutManifestFreezeError(
            "existing manifest output could not be verified"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_write(path: Path, content: bytes) -> bool:
    parent_descriptor, resolved_parent, name = _open_output_parent(
        Path(path)
    )
    temporary_name = (
        f".{name}.{secrets.token_hex(12)}.tmp"
    )
    descriptor = -1
    created_final_binding: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        temporary_metadata = os.fstat(descriptor)
        os.close(descriptor)
        descriptor = -1
        _verify_parent_binding(parent_descriptor, resolved_parent)
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            existing = _read_output_at(parent_descriptor, name)
            if existing != content:
                raise LayoutManifestFreezeError(
                    "manifest output already exists with different bytes"
                )
            created = False
        else:
            created_final_binding = (
                temporary_metadata.st_dev,
                temporary_metadata.st_ino,
            )
            os.fsync(parent_descriptor)
            created = True
        _verify_parent_binding(parent_descriptor, resolved_parent)
        return created
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created_final_binding is not None:
            try:
                final_metadata = os.stat(
                    name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    final_metadata.st_dev,
                    final_metadata.st_ino,
                ) == created_final_binding:
                    os.unlink(name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent_descriptor)


def _validate_manifest_output_location(
    *,
    pdf_root: Path,
    output: Path,
) -> Path:
    if output.is_symlink() or output.parent.is_symlink():
        raise LayoutManifestFreezeError(
            "manifest output path must not contain symlinks"
        )
    try:
        resolved_output = output.resolve()
    except (OSError, RuntimeError) as exc:
        raise LayoutManifestFreezeError(
            "manifest paths could not be resolved safely"
        ) from exc
    try:
        resolved_output.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise LayoutManifestFreezeError(
            "identity-bearing manifest output must be outside the repository"
        )
    try:
        resolved_output.relative_to(pdf_root)
    except ValueError:
        pass
    else:
        raise LayoutManifestFreezeError(
            "manifest output must be outside the PDF input directory"
        )
    return resolved_output


def _validate_pdf_root_binding(requested: Path, expected: Path) -> None:
    try:
        current = requested.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LayoutManifestFreezeError(
            "PDF input path could not be resolved safely"
        ) from exc
    if (
        requested.is_symlink()
        or current != expected
        or not current.is_dir()
    ):
        raise LayoutManifestFreezeError(
            "PDF input path must be one stable non-symlink directory"
        )


def _read_manifest_output(path: Path) -> bytes:
    parent_descriptor, resolved_parent, name = _open_output_parent(path)
    try:
        _verify_parent_binding(parent_descriptor, resolved_parent)
        content = _read_output_at(parent_descriptor, name)
        _verify_parent_binding(parent_descriptor, resolved_parent)
        return content
    finally:
        os.close(parent_descriptor)


def freeze_layout_manifest(
    pdf_dir: Path,
    output: Path,
    *,
    expected_count: int,
    split_seed: str,
) -> dict[str, object]:
    """Build and atomically write a manifest, returning aggregate evidence."""

    output = Path(output)
    requested_pdf_root = Path(pdf_dir)
    try:
        pdf_root = requested_pdf_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LayoutManifestFreezeError(
            "manifest paths could not be resolved safely"
        ) from exc
    _validate_pdf_root_binding(requested_pdf_root, pdf_root)
    resolved_output = _validate_manifest_output_location(
        pdf_root=pdf_root,
        output=output,
    )

    manifest, group_sizes = build_layout_manifest(
        pdf_root,
        expected_count=expected_count,
        split_seed=split_seed,
    )
    _validate_pdf_root_binding(requested_pdf_root, pdf_root)
    content = (canonical_json(manifest) + "\n").encode("utf-8")
    if (
        _validate_manifest_output_location(
            pdf_root=pdf_root,
            output=output,
        )
        != resolved_output
    ):
        raise LayoutManifestFreezeError(
            "manifest output binding changed during the freeze"
        )
    try:
        _atomic_write(resolved_output, content)
    except LayoutManifestFreezeError:
        raise
    except OSError as exc:
        raise LayoutManifestFreezeError(
            "manifest output could not be written atomically"
        ) from exc
    try:
        _validate_pdf_root_binding(requested_pdf_root, pdf_root)
        if (
            _validate_manifest_output_location(
                pdf_root=pdf_root,
                output=output,
            )
            != resolved_output
        ):
            raise LayoutManifestFreezeError(
                "manifest output binding changed during the freeze"
            )
        written = _read_manifest_output(resolved_output)
    except LayoutManifestFreezeError:
        raise
    except OSError as exc:
        raise LayoutManifestFreezeError(
            "written manifest could not be verified"
        ) from exc
    if written != content:
        raise LayoutManifestFreezeError(
            "written manifest does not match canonical bytes"
        )

    return {
        "case_count": expected_count,
        "folds": FOLDS,
        "group_count": len(group_sizes),
        "largest_group_count": max(group_sizes.values()),
        "repeats": REPEATS,
        "schema": SCHEMA,
        "sha256": hashlib.sha256(written).hexdigest(),
        "singleton_group_count": sum(
            size == 1 for size in group_sizes.values()
        ),
        "smallest_group_count": min(group_sizes.values()),
    }


def _expected_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < FOLDS:
        raise argparse.ArgumentTypeError(f"must be at least {FOLDS}")
    return parsed


def _split_seed(value: str) -> str:
    if not value or value != value.strip():
        raise argparse.ArgumentTypeError(
            "must be non-empty with no surrounding whitespace"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an external label-blind layout manifest and print only "
            "aggregate evidence."
        )
    )
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-count",
        type=_expected_count,
        required=True,
        help="exact number of canonical PDF cases expected in --pdf-dir",
    )
    parser.add_argument(
        "--split-seed",
        type=_split_seed,
        required=True,
        help="non-empty seed frozen into the manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        summary = freeze_layout_manifest(
            arguments.pdf_dir,
            arguments.output,
            expected_count=arguments.expected_count,
            split_seed=arguments.split_seed,
        )
    except LayoutManifestFreezeError as exc:
        parser.error(str(exc))
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
