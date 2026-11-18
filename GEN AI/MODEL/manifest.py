"""
Data manifest builder.
Records case id, image path, label path, shapes, metadata, and QA flags.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import nibabel as nib
import numpy as np
from datetime import datetime


@dataclass
class CaseMetadata:
    """Metadata for a single case."""
    case_id: str
    image_path: str
    label_path: str
    image_shape: List[int]
    label_shape: List[int]
    affine_image: List[List[float]]
    affine_label: List[List[float]]
    spacing_image: List[float]
    spacing_label: List[float]
    orientation: str
    label_foreground_voxel_count: int
    label_background_voxel_count: int
    label_foreground_ratio: float
    image_min: float
    image_max: float
    image_mean: float
    qa_flags: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.qa_flags is None:
            self.qa_flags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert numpy types to Python types
        return data
    
    def has_qa_issues(self) -> bool:
        """Check if case has any QA flags."""
        return len(self.qa_flags) > 0


class ManifestBuilder:
    """Build and manage dataset manifest."""
    
    EXCLUSION_LIST = {
        "LUNG1-035",  # Problematic case to exclude
    }
    
    def __init__(self, images_dir: Path, labels_dir: Path):
        """Initialize manifest builder."""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.manifest: Dict[str, CaseMetadata] = {}
        self.excluded_cases: List[str] = []
    
    def discover_cases(self) -> List[str]:
        """Discover all cases by matching image and label files."""
        image_files = sorted(self.images_dir.glob("*.nii.gz"))
        case_ids = []
        
        for img_file in image_files:
            case_id = img_file.stem.replace(".nii", "")
            label_file = self.labels_dir / f"{case_id}.nii.gz"
            
            # Check if case should be excluded
            if case_id in self.EXCLUSION_LIST:
                self.excluded_cases.append(case_id)
                print(f"Excluding case: {case_id}")
                continue
            
            # Check if label exists
            if not label_file.exists():
                print(f"Warning: Missing label for {case_id}")
                continue
            
            case_ids.append(case_id)
        
        return case_ids
    
    def build_case_metadata(self, case_id: str) -> Optional[CaseMetadata]:
        """Build metadata for a single case."""
        try:
            image_path = self.images_dir / f"{case_id}.nii.gz"
            label_path = self.labels_dir / f"{case_id}.nii.gz"
            
            # Load NIfTI files
            img_nib = nib.load(image_path)
            label_nib = nib.load(label_path)
            
            img_data = np.asarray(img_nib.dataobj)
            label_data = np.asarray(label_nib.dataobj)
            
            # Extract metadata
            image_shape = list(img_data.shape)
            label_shape = list(label_data.shape)
            
            # Get affine matrices (convert to list for JSON serialization)
            affine_image = img_nib.affine.tolist()
            affine_label = label_nib.affine.tolist()
            
            # Get voxel spacing
            spacing_image = list(np.diag(img_nib.affine)[:3])
            spacing_label = list(np.diag(label_nib.affine)[:3])
            
            # Get orientation
            orientation = nib.aff2axcodes(img_nib.affine)
            
            # Analyze label
            label_fg_count = int(np.sum(label_data > 0))
            label_bg_count = int(np.sum(label_data == 0))
            total = label_fg_count + label_bg_count
            label_fg_ratio = label_fg_count / total if total > 0 else 0.0
            
            # Image statistics
            img_min = float(np.min(img_data))
            img_max = float(np.max(img_data))
            img_mean = float(np.mean(img_data))
            
            # QA checks
            qa_flags = []
            if label_fg_count == 0:
                qa_flags.append("empty_label")
            if label_fg_ratio < 0.001:
                qa_flags.append("tiny_foreground")
            if image_shape != label_shape:
                qa_flags.append("shape_mismatch")
            
            metadata = CaseMetadata(
                case_id=case_id,
                image_path=str(image_path),
                label_path=str(label_path),
                image_shape=image_shape,
                label_shape=label_shape,
                affine_image=affine_image,
                affine_label=affine_label,
                spacing_image=spacing_image,
                spacing_label=spacing_label,
                orientation=orientation,
                label_foreground_voxel_count=label_fg_count,
                label_background_voxel_count=label_bg_count,
                label_foreground_ratio=label_fg_ratio,
                image_min=img_min,
                image_max=img_max,
                image_mean=img_mean,
                qa_flags=qa_flags,
            )
            
            return metadata
        
        except Exception as e:
            print(f"Error processing case {case_id}: {e}")
            return None
    
    def build_manifest(self) -> Dict[str, CaseMetadata]:
        """Build complete manifest for all discovered cases."""
        case_ids = self.discover_cases()
        print(f"Found {len(case_ids)} cases to process")
        
        count = 0
        for case_id in case_ids:
            metadata = self.build_case_metadata(case_id)
            if metadata is not None:
                self.manifest[case_id] = metadata
                count += 1
        
        print(f"Successfully processed {count}/{len(case_ids)} cases")
        return self.manifest
    
    def save_manifest(self, filepath: Path) -> None:
        """Save manifest to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        manifest_dict = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(self.manifest),
            "excluded_cases": self.excluded_cases,
            "cases": {case_id: metadata.to_dict() for case_id, metadata in self.manifest.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(manifest_dict, f, indent=2)
        
        print(f"Manifest saved to {filepath}")
    
    @staticmethod
    def load_manifest(filepath: Path) -> Dict[str, CaseMetadata]:
        """Load manifest from JSON file."""
        with open(filepath, 'r') as f:
            manifest_dict = json.load(f)
        
        manifest = {}
        for case_id, case_data in manifest_dict["cases"].items():
            # Reconstruct affine as list (already converted by to_dict)
            metadata = CaseMetadata(**case_data)
            manifest[case_id] = metadata
        
        return manifest
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the manifest."""
        if not self.manifest:
            return {}
        
        foreground_ratios = [m.label_foreground_ratio for m in self.manifest.values()]
        shapes = [m.image_shape for m in self.manifest.values()]
        
        stats = {
            "total_cases": len(self.manifest),
            "cases_with_qa_issues": sum(1 for m in self.manifest.values() if m.has_qa_issues()),
            "mean_foreground_ratio": float(np.mean(foreground_ratios)),
            "min_foreground_ratio": float(np.min(foreground_ratios)),
            "max_foreground_ratio": float(np.max(foreground_ratios)),
            "excluded_cases": self.excluded_cases,
        }
        
        return stats


if __name__ == "__main__":
    from project_v2.paths import DATA_IMAGES_DIR, DATA_LABELS_DIR, MANIFESTS_DIR
    
    builder = ManifestBuilder(DATA_IMAGES_DIR, DATA_LABELS_DIR)
    manifest = builder.build_manifest()
    
    print(f"Manifest built: {len(manifest)} cases")
    print(f"Statistics: {builder.get_statistics()}")
    
    # Save manifest
    manifest_path = MANIFESTS_DIR / "manifest_main.json"
    builder.save_manifest(manifest_path)
