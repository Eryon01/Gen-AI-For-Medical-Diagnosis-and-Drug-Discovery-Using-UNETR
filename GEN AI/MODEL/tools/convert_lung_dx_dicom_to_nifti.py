"""
Convert Lung_Dx DICOM directory trees into NIfTI volumes for segmentation.

Lung_Dx dataset structure:
  - lung_pet_ct_dx/<CASE_ID>/<STUDY_UID>/<SERIES_UID>/*.dcm

This script converts to:
  - lung_dx_nifti/images/<case_id>.nii.gz
  - lung_dx_nifti/labels/ (placeholder labels for now)

Usage:
  python tools/convert_lung_dx_dicom_to_nifti.py ^
    --dicom-root E:\\aryan_v2\\lung_pet_ct_dx ^
    --out-root E:\\aryan_v2\\lung_dx_nifti ^
    --min-slices 50
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import SimpleITK as sitk
import pydicom
from tqdm import tqdm


SKIP_DESC_SUBSTRINGS = (
    "LOCALIZER",
    "SCOUT",
    "TOPOGRAM",
    "SURVIEW",
    "LABEL",
)


def _safe_slug(s: str) -> str:
    """Slugify string for safe filename."""
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s[:120] if len(s) > 120 else s


@dataclass(frozen=True)
class SeriesInfo:
    case_id: str
    study_uid: str
    series_uid: str
    modality: str
    series_description: str
    body_part: str
    num_files: int
    series_dir: Path


def _read_series_info(series_dir: Path) -> Optional[SeriesInfo]:
    """Extract metadata from first DICOM file in series."""
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return None

    try:
        # Read only the header to be fast.
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)

        modality = str(getattr(ds, "Modality", "") or "")
        study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
        series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        series_desc = str(getattr(ds, "SeriesDescription", "") or "")
        body_part = str(getattr(ds, "BodyPartExamined", "") or "")

        # Get case ID from directory structure
        # Expected directory structure:
        #   lung_pet_ct_dx/<CASE_ID>/<STUDY_UID>/<SERIES_UID>/*.dcm
        # From series_dir perspective:
        #   series_dir.parents[0] = <STUDY_UID> dir
        #   series_dir.parents[1] = <CASE_ID> dir (e.g., Lung_Dx-A0001)
        #   series_dir.parents[2] = lung_pet_ct_dx dir
        case_id = series_dir.parents[1].name

        return SeriesInfo(
            case_id=case_id,
            study_uid=study_uid,
            series_uid=series_uid,
            modality=modality,
            series_description=series_desc,
            body_part=body_part,
            num_files=len(dcm_files),
            series_dir=series_dir,
        )
    except Exception as e:
        print(f"Warning: Could not read metadata from {series_dir}: {e}")
        return None


def _iter_series_dirs(dicom_root: Path) -> Iterable[Path]:
    """Iterate through all series directories."""
    # Expected: dicom_root/<CASE_ID>/<STUDY_UID>/<SERIES_UID>/*.dcm
    for case_dir in sorted([p for p in dicom_root.iterdir() if p.is_dir()]):
        if case_dir.name.startswith('.'):
            continue
        for study_dir in sorted([p for p in case_dir.iterdir() if p.is_dir()]):
            if study_dir.name.startswith('.'):
                continue
            for series_dir in sorted([p for p in study_dir.iterdir() if p.is_dir()]):
                if series_dir.name.startswith('.'):
                    continue
                yield series_dir


def _looks_like_diagnostic_ct(info: SeriesInfo, min_slices: int) -> bool:
    """Check if series is likely a diagnostic CT scan."""
    if info.modality.upper() != "CT":
        return False
    if info.num_files < min_slices:
        return False
    desc = (info.series_description or "").upper()
    if any(s in desc for s in SKIP_DESC_SUBSTRINGS):
        return False
    return True


def _read_dicom_series_to_image(series_dir: Path) -> sitk.Image:
    """Read DICOM series and convert to 3D image."""
    # Use GDCM to get files in correct slice order.
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not file_names:
        raise RuntimeError(f"No readable DICOM series found in: {series_dir}")
    reader.SetFileNames(file_names)
    return reader.Execute()


def _write_nifti(img: sitk.Image, out_path: Path) -> None:
    """Write SimpleITK image to NIfTI file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def _create_placeholder_label(case_id: str, output_label_dir: Path, num_slices: int) -> None:
    """Create a placeholder label (all zeros) for the given case."""
    import numpy as np
    
    # Create an all-zeros label image
    label_array = np.zeros((512, 512, num_slices), dtype=np.uint8)
    
    # Convert to SimpleITK image
    label_img = sitk.GetImageFromArray(label_array)
    
    # Save to NIfTI
    label_path = output_label_dir / f"{case_id}.nii.gz"
    _write_nifti(label_img, label_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Lung_Dx DICOM to NIfTI")
    ap.add_argument("--dicom-root", type=Path, required=True, help="Path to lung_pet_ct_dx directory")
    ap.add_argument("--out-root", type=Path, required=True, help="Output directory for NIfTI files")
    ap.add_argument("--min-slices", type=int, default=50, help="Minimum slices for valid series")
    ap.add_argument("--keep-all-ct", action="store_true", help="Do not skip LOCALIZER/SCOUT etc.")
    ap.add_argument("--create-labels", action="store_true", help="Create placeholder labels"),
    ap.add_argument("--limit", type=int, default=0, help="For testing: max number of series to convert (0 = no limit).")
    args = ap.parse_args()

    dicom_root: Path = args.dicom_root
    out_root: Path = args.out_root
    min_slices: int = args.min_slices
    keep_all_ct: bool = bool(args.keep_all_ct)
    create_labels: bool = bool(args.create_labels)
    limit: int = int(args.limit)

    if not dicom_root.exists():
        raise SystemExit(f"DICOM root not found: {dicom_root}")

    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "metadata.csv"
    error_path = out_root / "errors.txt"

    converted = 0
    skipped = 0
    errors = 0
    cases_processed = set()

    print(f"Converting DICOM files from {dicom_root}")
    print(f"Output directory: {out_root}")
    print(f"Min slices: {min_slices}")
    print(f"Create placeholder labels: {create_labels}")

    series_dirs = list(_iter_series_dirs(dicom_root))
    print(f"Found {len(series_dirs)} series directories")

    with meta_path.open("w", newline="", encoding="utf-8") as fmeta, error_path.open("w", encoding="utf-8") as ferr:
        writer = csv.DictWriter(
            fmeta,
            fieldnames=[
                "case_id",
                "study_uid",
                "series_uid",
                "modality",
                "series_description",
                "body_part",
                "num_slices",
                "nifti_path",
                "source_series_dir",
            ],
        )
        writer.writeheader()

        for series_dir in tqdm(series_dirs, desc="Converting series"):
            info = None
            try:
                info = _read_series_info(series_dir)
                if info is None:
                    skipped += 1
                    continue

                if info.modality.upper() != "CT":
                    skipped += 1
                    continue

                if (not keep_all_ct) and (not _looks_like_diagnostic_ct(info, min_slices=min_slices)):
                    skipped += 1
                    continue

                # Use case ID as primary identifier
                case_id = info.case_id
                out_name = f"{case_id}.nii.gz"
                out_path = images_dir / out_name

                if out_path.exists():
                    # Idempotent: keep existing output
                    writer.writerow(
                        {
                            "case_id": case_id,
                            "study_uid": info.study_uid,
                            "series_uid": info.series_uid,
                            "modality": info.modality,
                            "series_description": info.series_description,
                            "body_part": info.body_part,
                            "num_slices": info.num_files,
                            "nifti_path": str(out_path),
                            "source_series_dir": str(series_dir),
                        }
                    )
                    converted += 1
                    cases_processed.add(case_id)
                    if limit and converted >= limit:
                        break
                    continue

                # Read and convert DICOM series
                img = _read_dicom_series_to_image(series_dir)
                _write_nifti(img, out_path)

                writer.writerow(
                    {
                        "case_id": case_id,
                        "study_uid": info.study_uid,
                        "series_uid": info.series_uid,
                        "modality": info.modality,
                        "series_description": info.series_description,
                        "body_part": info.body_part,
                        "num_slices": info.num_files,
                        "nifti_path": str(out_path),
                        "source_series_dir": str(series_dir),
                    }
                )
                converted += 1
                cases_processed.add(case_id)
                
                # Create placeholder label if requested
                if create_labels:
                    _create_placeholder_label(case_id, labels_dir, info.num_files)

                if limit and converted >= limit:
                    break

            except Exception as e:  # noqa: BLE001
                errors += 1
                ferr.write(f"[ERROR] series_dir={series_dir}\n")
                if info is not None:
                    ferr.write(f"  case_id={info.case_id} modality={info.modality} files={info.num_files} desc={info.series_description}\n")
                ferr.write(f"  exception={type(e).__name__}: {e}\n\n")
                continue

    # Print summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Converted: {converted} series from {len(cases_processed)} unique cases")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Output images: {images_dir}")
    print(f"Output labels: {labels_dir if create_labels else 'Not created'}")
    print(f"Metadata CSV:  {meta_path}")
    print(f"Errors log:    {error_path}")
    print("="*70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
