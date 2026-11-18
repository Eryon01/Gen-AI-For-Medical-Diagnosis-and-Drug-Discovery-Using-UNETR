import glob
import itertools
import multiprocessing as mp
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder
from tqdm import tqdm

NSCLC_ROOT = Path(r"e:\aryan\nsclc_radiomics")
OUT_IMAGES = Path(r"e:\aryan\nifti_data\images")
OUT_LABELS = Path(r"e:\aryan\nifti_data\labels")
MAX_WORKERS = 4

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_LABELS.mkdir(parents=True, exist_ok=True)


def find_ct_and_rtstruct(patient_dir):
    import pydicom

    ct_dir = None
    rt_file = None

    dcm_files = glob.glob(os.path.join(patient_dir, "**\\*.dcm"), recursive=True)
    if not dcm_files:
        return None, None

    for file_path in dcm_files:
        try:
            dataset = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            if getattr(dataset, "Modality", None) == "RTSTRUCT":
                rt_file = file_path
                break
        except Exception:
            continue

    if rt_file:
        rt_dir = os.path.dirname(rt_file)
        dir_counts = {}
        for file_path in dcm_files:
            directory = os.path.dirname(file_path)
            if directory != rt_dir:
                dir_counts[directory] = dir_counts.get(directory, 0) + 1

        if dir_counts:
            best_dir = max(dir_counts.items(), key=lambda item: item[1])[0]
            test_files = glob.glob(os.path.join(best_dir, "*.dcm"))
            if test_files:
                try:
                    dataset = pydicom.dcmread(test_files[0], force=True, stop_before_pixels=True)
                    if getattr(dataset, "Modality", None) == "CT":
                        ct_dir = best_dir
                except Exception:
                    pass

    return ct_dir, rt_file


def align_mask_to_ct(mask_3d, ct_shape):
    if mask_3d.shape == ct_shape:
        return mask_3d

    for axes in itertools.permutations(range(mask_3d.ndim)):
        candidate = np.transpose(mask_3d, axes)
        if candidate.shape == ct_shape:
            return candidate

    raise ValueError(f"Mask shape {mask_3d.shape} could not be aligned to CT shape {ct_shape}.")


def cleanup_paths(*paths):
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def process_patient(patient_id):
    patient_dir = NSCLC_ROOT / patient_id
    temp_image_path = OUT_IMAGES / f"{patient_id}.tmp.nii.gz"
    temp_label_path = OUT_LABELS / f"{patient_id}.tmp.nii.gz"
    final_image_path = OUT_IMAGES / f"{patient_id}.nii.gz"
    final_label_path = OUT_LABELS / f"{patient_id}.nii.gz"

    try:
        ct_dir, rt_file = find_ct_and_rtstruct(str(patient_dir))
        if not ct_dir or not rt_file:
            cleanup_paths(temp_image_path, temp_label_path)
            return f"Missing CT or RTSTRUCT for {patient_id}"

        if final_image_path.exists() and final_label_path.exists():
            cleanup_paths(temp_image_path, temp_label_path)
            return f"Skipped {patient_id}"

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ct_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, str(temp_image_path))

        rtstruct = RTStructBuilder.create_from(dicom_series_path=ct_dir, rt_struct_path=rt_file)
        roi_names = rtstruct.get_roi_names()
        target_roi = None
        for roi_name in roi_names:
            roi_name_lower = roi_name.lower()
            if "gtv" in roi_name_lower or "tumor" in roi_name_lower:
                target_roi = roi_name
                break

        if target_roi is None:
            if roi_names:
                target_roi = roi_names[0]
            else:
                cleanup_paths(temp_image_path, temp_label_path)
                return f"No ROIs found in {patient_id} RTSTRUCT"

        mask_3d = rtstruct.get_roi_mask_by_name(target_roi)
        ct_nifti = nib.load(str(temp_image_path))
        mask_3d = align_mask_to_ct(mask_3d, ct_nifti.shape)

        label_image = nib.Nifti1Image(mask_3d.astype(np.uint8), ct_nifti.affine)
        nib.save(label_image, str(temp_label_path))

        os.replace(temp_image_path, final_image_path)
        os.replace(temp_label_path, final_label_path)
        return f"Successfully processed {patient_id}"
    except Exception as exc:
        cleanup_paths(temp_image_path, temp_label_path)
        return f"Error on {patient_id}: {exc}"


def main():
    patients = sorted(
        patient_dir.name
        for patient_dir in NSCLC_ROOT.iterdir()
        if patient_dir.is_dir() and patient_dir.name.startswith("LUNG1-")
    )
    if not patients:
        raise FileNotFoundError(f"No patient folders were found under {NSCLC_ROOT}.")

    worker_count = min(MAX_WORKERS, max(2, (os.cpu_count() or 8) // 8))
    print(f"Found {len(patients)} patients.")
    print(f"Using {worker_count} worker processes to keep the system responsive.")

    with mp.Pool(processes=worker_count, maxtasksperchild=1) as pool:
        results = list(tqdm(pool.imap(process_patient, patients), total=len(patients)))

    errors = [result for result in results if not result.startswith("Successfully") and not result.startswith("Skipped")]
    print("\nProcess complete!")
    print(f"Total errors encountered: {len(errors)}")
    for error in errors[:10]:
        print(error)


if __name__ == "__main__":
    mp.freeze_support()
    main()
