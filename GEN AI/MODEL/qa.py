"""
Data QA and preflight checks.
Identifies missing pairs, empty masks, metadata mismatches, and suspicious ratios.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import nibabel as nib
from project_v2.manifest import ManifestBuilder, CaseMetadata


class DataQA:
    """Quality assurance checks for dataset."""
    
    def __init__(self, images_dir: Path, labels_dir: Path):
        """Initialize QA checker."""
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.issues: Dict[str, List[str]] = {}
        self.warnings: List[str] = []
    
    def check_missing_pairs(self) -> Dict[str, List[str]]:
        """Check for missing image-label pairs."""
        missing = {"orphan_images": [], "orphan_labels": []}
        
        image_files = set(f.stem.replace(".nii", "") for f in self.images_dir.glob("*.nii.gz"))
        label_files = set(f.stem.replace(".nii", "") for f in self.labels_dir.glob("*.nii.gz"))
        
        missing["orphan_images"] = list(image_files - label_files)
        missing["orphan_labels"] = list(label_files - image_files)
        
        return missing
    
    def check_empty_masks(self, manifest: Dict[str, CaseMetadata]) -> List[str]:
        """Check for empty or near-empty masks."""
        empty_cases = []
        threshold = 0.001  # 0.1% foreground ratio
        
        for case_id, metadata in manifest.items():
            if metadata.label_foreground_ratio < threshold:
                empty_cases.append(case_id)
        
        return empty_cases
    
    def check_metadata_mismatch(self, manifest: Dict[str, CaseMetadata]) -> List[str]:
        """Check for metadata mismatches between image and label."""
        mismatches = []
        
        for case_id, metadata in manifest.items():
            if metadata.image_shape != metadata.label_shape:
                mismatches.append(case_id)
                self.warnings.append(f"{case_id}: Shape mismatch - {metadata.image_shape} vs {metadata.label_shape}")
            
            # Check spacing
            spacing_diff = np.array(metadata.spacing_image) - np.array(metadata.spacing_label)
            if np.any(np.abs(spacing_diff) > 0.1):
                self.warnings.append(f"{case_id}: Spacing mismatch - {metadata.spacing_image} vs {metadata.spacing_label}")
        
        return mismatches
    
    def check_suspicious_foreground_ratios(self, manifest: Dict[str, CaseMetadata]) -> Dict[str, List[str]]:
        """Check for suspicious foreground ratios."""
        issues = {"very_small": [], "very_large": []}
        
        for case_id, metadata in manifest.items():
            ratio = metadata.label_foreground_ratio
            
            if ratio < 0.001:
                issues["very_small"].append(f"{case_id} ({ratio:.4f})")
            elif ratio > 0.95:
                issues["very_large"].append(f"{case_id} ({ratio:.4f})")
        
        return issues
    
    def check_transform_shape_drift(self, manifest: Dict[str, CaseMetadata]) -> List[str]:
        """Check for potential shape drift during transforms."""
        # This is a preflight check to identify cases that might have issues
        potential_issues = []
        
        for case_id, metadata in manifest.items():
            if metadata.image_shape != metadata.label_shape:
                potential_issues.append(case_id)
        
        return potential_issues
    
    def run_all_checks(self, manifest: Dict[str, CaseMetadata]) -> Dict[str, Any]:
        """Run all QA checks and return comprehensive report."""
        issues = {
            "missing_pairs": self.check_missing_pairs(),
            "empty_masks": self.check_empty_masks(manifest),
            "metadata_mismatch": self.check_metadata_mismatch(manifest),
            "suspicious_ratios": self.check_suspicious_foreground_ratios(manifest),
            "transform_drift": self.check_transform_shape_drift(manifest),
        }
        
        report = {
            "total_cases_checked": len(manifest),
            "issues": issues,
            "warnings": self.warnings,
            "summary": {
                "missing_image_label_pairs": len(issues["missing_pairs"]["orphan_images"]) + len(issues["missing_pairs"]["orphan_labels"]),
                "empty_or_tiny_masks": len(issues["empty_masks"]),
                "metadata_mismatches": len(issues["metadata_mismatch"]),
                "suspicious_ratios": len(issues["suspicious_ratios"]["very_small"]) + len(issues["suspicious_ratios"]["very_large"]),
                "potential_shape_drift": len(issues["transform_drift"]),
            }
        }
        
        return report
    
    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Print QA report in human-readable format."""
        print("\n" + "="*60)
        print("DATA QA REPORT")
        print("="*60)
        print(f"Total cases checked: {report['total_cases_checked']}")
        print("\nSUMMARY:")
        for key, value in report["summary"].items():
            print(f"  {key}: {value}")
        
        print("\nDETAILS:")
        if report["issues"]["empty_masks"]:
            print(f"\nEmpty/Tiny masks ({len(report['issues']['empty_masks'])}):")
            for case in report["issues"]["empty_masks"][:5]:  # Show first 5
                print(f"  - {case}")
            if len(report["issues"]["empty_masks"]) > 5:
                print(f"  ... and {len(report['issues']['empty_masks']) - 5} more")
        
        if report["issues"]["metadata_mismatch"]:
            print(f"\nMetadata mismatches ({len(report['issues']['metadata_mismatch'])}):")
            for case in report["issues"]["metadata_mismatch"][:5]:
                print(f"  - {case}")
            if len(report["issues"]["metadata_mismatch"]) > 5:
                print(f"  ... and {len(report['issues']['metadata_mismatch']) - 5} more")
        
        if report["warnings"]:
            print(f"\nWarnings ({len(report['warnings'])}):")
            for warning in report["warnings"][:5]:
                print(f"  - {warning}")
            if len(report["warnings"]) > 5:
                print(f"  ... and {len(report['warnings']) - 5} more")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    from project_v2.paths import DATA_IMAGES_DIR, DATA_LABELS_DIR
    
    # Build manifest
    builder = ManifestBuilder(DATA_IMAGES_DIR, DATA_LABELS_DIR)
    manifest = builder.build_manifest()
    
    # Run QA
    qa = DataQA(DATA_IMAGES_DIR, DATA_LABELS_DIR)
    report = qa.run_all_checks(manifest)
    
    # Print report
    DataQA.print_report(report)
