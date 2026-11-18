"""
Project paths and directory structure.
All paths are relative to the project root.
"""

from pathlib import Path
import sys
import os

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
ARYAN_V1 = Path("E:\\aryan")
ARYAN_V2 = PROJECT_ROOT

# Dataset selection (default to unified_lung1, can override with ARYAN_DATASET env var)
DATASET_NAME = os.getenv("ARYAN_DATASET", "unified_lung1")

# Available datasets configuration
DATASETS = {
    "tcga_luad_nifti": {
        "images_dir": ARYAN_V2 / "tcga_luad_nifti" / "images",
        "labels_dir": ARYAN_V2 / "tcga_luad_nifti" / "labels",
        "name": "TCGA LUAD"
    },
    "lung_dx_nifti": {
        "images_dir": ARYAN_V2 / "lung_dx_nifti" / "images",
        "labels_dir": ARYAN_V2 / "lung_dx_nifti" / "labels",
        "name": "Lung_Dx"
    },
    "unified_lung1": {
        "images_dir": ARYAN_V2 / "unified_dataset" / "LUNG1" / "images",
        "labels_dir": ARYAN_V2 / "unified_dataset" / "LUNG1" / "labels",
        "name": "LUNG1 (Unified 422 Cases)"
    }
}

# Set current dataset paths
if DATASET_NAME not in DATASETS:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}. Available: {list(DATASETS.keys())}")

CURRENT_DATASET = DATASETS[DATASET_NAME]
DATA_IMAGES_DIR = CURRENT_DATASET["images_dir"]
DATA_LABELS_DIR = CURRENT_DATASET["labels_dir"]

# Artifacts
ARTIFACTS_ROOT = ARYAN_V2 / "artifacts"
MANIFESTS_DIR = ARTIFACTS_ROOT / "manifests"
SPLITS_DIR = ARTIFACTS_ROOT / "splits"
RUNS_DIR = ARTIFACTS_ROOT / "runs"
REPORTS_DIR = ARTIFACTS_ROOT / "reports"

# Checkpoints
CHECKPOINTS_DIR = RUNS_DIR / "checkpoints"

# Config
CONFIG_DIR = ARYAN_V2 / "configs"

def ensure_directories():
    """Create all required directories."""
    for directory in [
        MANIFESTS_DIR, SPLITS_DIR, RUNS_DIR, REPORTS_DIR, 
        CHECKPOINTS_DIR, CONFIG_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()                      
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Current dataset: {DATASET_NAME} ({CURRENT_DATASET['name']})")
    print(f"Data images: {DATA_IMAGES_DIR}")
    print(f"Data labels: {DATA_LABELS_DIR}")
    print(f"Artifacts: {ARTIFACTS_ROOT}")
    print(f"\nAvailable datasets:")
    for ds_name, ds_config in DATASETS.items():
        marker = "✓" if ds_name == DATASET_NAME else " "
        print(f"  [{marker}] {ds_name} ({ds_config['name']})")
    print(f"\nTo switch datasets, set environment variable:")
    print(f"  set ARYAN_DATASET=lung_dx_nifti")
