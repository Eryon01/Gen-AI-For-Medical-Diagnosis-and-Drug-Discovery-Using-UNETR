"""
Convert TCGA-style DICOM directory trees (patient/study/series/*.dcm) into NIfTI volumes.

This is intended for 3D segmentation pipelines (e.g., MONAI UNETR) which expect:
  - one 3D volume per case (NIfTI .nii.gz)
  - an optional 3D label mask per case (NIfTI), if you have annotations

TCGA-LUAD imaging downloads often include multiple series per patient, including LOCALIZER/scout.
This script filters to CT series and skips obvious non-diagnostic series by description.

Example:
  E:\\aryan_v2\\.venv\\Scripts\\python.exe tools\\convert_tcga_dicom_to_nifti.py ^
    --dicom-root E:\\aryan_v2\\tcga_luad ^
    --out-root E:\\aryan_v2\\tcga_luad_nifti ^
    --min-slices 64
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
)


def _safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s[:120] if len(s) > 120 else s


@dataclass(frozen=True)
class SeriesInfo:
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    series_description: str
    body_part: str
    num_files: int
    series_dir: Path


def _read_series_info(series_dir: Path) -> Optional[SeriesInfo]:
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return None

    # Read only the header to be fast.
    ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)

    modality = str(getattr(ds, "Modality", "") or "")
    patient_id = str(getattr(ds, "PatientID", "") or "")
    study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
    series_desc = str(getattr(ds, "SeriesDescription", "") or "")
    body_part = str(getattr(ds, "BodyPartExamined", "") or "")

    return SeriesInfo(
        patient_id=patient_id,
        study_uid=study_uid,
        series_uid=series_uid,
        modality=modality,
        series_description=series_desc,
        body_part=body_part,
        num_files=len(dcm_files),
        series_dir=series_dir,
    )


def _iter_series_dirs(dicom_root: Path) -> Iterable[Path]:
    # Expected: dicom_root/<PATIENT>/<STUDY_UID>/<SERIES_UID>/*.dcm
    for patient_dir in sorted([p for p in dicom_root.iterdir() if p.is_dir()]):
        for study_dir in sorted([p for p in patient_dir.iterdir() if p.is_dir()]):
            for series_dir in sorted([p for p in study_dir.iterdir() if p.is_dir()]):
                yield series_dir


def _looks_like_diagnostic_ct(info: SeriesInfo, min_slices: int) -> bool:
    if info.modality.upper() != "CT":
        return False
    if info.num_files < min_slices:
        return False
    desc = (info.series_description or "").upper()
    if any(s in desc for s in SKIP_DESC_SUBSTRINGS):
        return False
    return True


def _read_dicom_series_to_image(series_dir: Path) -> sitk.Image:
    # Use GDCM to get files in correct slice order.
    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not file_names:
        raise RuntimeError(f"No readable DICOM series found in: {series_dir}")
    reader.SetFileNames(file_names)
    return reader.Execute()


def _write_nifti(img: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--min-slices", type=int, default=64)
    ap.add_argument("--keep-all-ct", action="store_true", help="Do not skip LOCALIZER/SCOUT etc.")
    ap.add_argument("--limit", type=int, default=0, help="For testing: max number of series to convert (0 = no limit).")
    args = ap.parse_args()

    dicom_root: Path = args.dicom_root
    out_root: Path = args.out_root
    min_slices: int = args.min_slices
    keep_all_ct: bool = bool(args.keep_all_ct)
    limit: int = int(args.limit)

    if not dicom_root.exists():
        raise SystemExit(f"DICOM root not found: {dicom_root}")

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "metadata.csv"
    error_path = out_root / "errors.txt"

    converted = 0
    skipped = 0
    errors = 0

    with meta_path.open("w", newline="", encoding="utf-8") as fmeta, error_path.open("w", encoding="utf-8") as ferr:
        writer = csv.DictWriter(
            fmeta,
            fieldnames=[
                "patient_id",
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

        for series_dir in tqdm(list(_iter_series_dirs(dicom_root)), desc="Scanning series"):
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

                patient = _safe_slug(info.patient_id or series_dir.parents[2].name)
                study_tail = info.study_uid[-12:] if info.study_uid else "nostudy"
                series_tail = info.series_uid[-12:] if info.series_uid else series_dir.name[-12:]
                desc_slug = _safe_slug(info.series_description or "CT")

                out_name = f"{patient}__{study_tail}__{series_tail}__{desc_slug}.nii.gz"
                out_path = images_dir / out_name

                if out_path.exists():
                    # Idempotent runs: keep existing output and just record it.
                    writer.writerow(
                        {
                            "patient_id": info.patient_id,
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
                    if limit and converted >= limit:
                        break
                    continue

                img = _read_dicom_series_to_image(series_dir)
                _write_nifti(img, out_path)

                writer.writerow(
                    {
                        "patient_id": info.patient_id,
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
                if limit and converted >= limit:
                    break

            except Exception as e:  # noqa: BLE001
                errors += 1
                ferr.write(f"[ERROR] series_dir={series_dir}\n")
                if info is not None:
                    ferr.write(f"  patient_id={info.patient_id} modality={info.modality} files={info.num_files} desc={info.series_description}\n")
                ferr.write(f"  exception={type(e).__name__}: {e}\n\n")
                continue

    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Output images: {images_dir}")
    print(f"Metadata CSV:  {meta_path}")
    print(f"Errors log:    {error_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

