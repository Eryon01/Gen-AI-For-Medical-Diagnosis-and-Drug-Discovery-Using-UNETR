"""
Reporting and visualization utilities.
Generate per-case metrics, summaries, and overlay visualizations.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


class ReportGenerator:
    """Generate structured reports from results."""
    
    @staticmethod
    def save_case_metrics(
        case_id: str,
        metrics: Dict[str, float],
        output_path: Path,
    ) -> None:
        """Save per-case metrics as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "case_id": case_id,
            "metrics": metrics,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    @staticmethod
    def save_aggregate_report(
        all_metrics: Dict[str, Dict[str, float]],
        output_path: Path,
    ) -> None:
        """Save aggregate report across all cases."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Compute statistics
        stats = {}
        metric_names = set()
        for m in all_metrics.values():
            metric_names.update(m.keys())
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics.values() if metric_name in m]
            
            if metric_name == "empty_prediction":
                rate = sum(values) / len(values)
                stats[metric_name] = {
                    "mean": rate,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                }
            else:
                stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        
        # Find best and worst cases
        dice_scores = {cid: m.get("dice", 0) for cid, m in all_metrics.items()}
        best_case = max(dice_scores, key=dice_scores.get)
        worst_case = min(dice_scores, key=dice_scores.get)
        
        report = {
            "total_cases": len(all_metrics),
            "statistics": stats,
            "best_case": {
                "case_id": best_case,
                "dice": dice_scores[best_case],
            },
            "worst_case": {
                "case_id": worst_case,
                "dice": dice_scores[worst_case],
            },
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    @staticmethod
    def print_aggregate_report(report: Dict[str, Any]) -> None:
        """Print aggregate report."""
        print("\n" + "="*70)
        print("EVALUATION REPORT")
        print("="*70)
        print(f"Total cases: {report['total_cases']}")
        
        print("\nMetric Statistics:")
        print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-"*70)
        
        for metric_name, stats in sorted(report['statistics'].items()):
            print(f"{metric_name:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                  f"{stats['min']:>10.4f} {stats['max']:>10.4f}")
        
        print(f"\nBest case: {report['best_case']['case_id']} (Dice: {report['best_case']['dice']:.4f})")
        print(f"Worst case: {report['worst_case']['case_id']} (Dice: {report['worst_case']['dice']:.4f})")
        print("="*70)


class Visualizations:
    """Generate visualization overlays."""
    
    @staticmethod
    def create_overlay_image(
        image: np.ndarray,
        prediction: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        slice_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Create RGB overlay image for visualization."""
        if image.ndim == 3:
            # Auto-select middle slice if not specified
            if slice_idx is None:
                slice_idx = image.shape[2] // 2
            
            image_slice = image[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]
            gt_slice = ground_truth[:, :, slice_idx] if ground_truth is not None else None
        else:
            image_slice = image
            pred_slice = prediction
            gt_slice = ground_truth if ground_truth is not None else None
        
        # Normalize image
        if image_slice.max() > image_slice.min():
            image_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        else:
            image_normalized = image_slice
        
        # Create RGB image
        h, w = image_slice.shape
        rgb = np.zeros((h, w, 3))
        rgb[:] = image_normalized[:, :, np.newaxis]
        
        # Add prediction in red
        rgb[pred_slice > 0, 0] = 1.0
        rgb[pred_slice > 0, 1] = 0.0
        rgb[pred_slice > 0, 2] = 0.0
        
        # Add ground truth in green
        if gt_slice is not None:
            rgb[gt_slice > 0, 0] = 0.0
            rgb[gt_slice > 0, 1] = 1.0
            rgb[gt_slice > 0, 2] = 0.0
        
        return (rgb * 255).astype(np.uint8)
    
    @staticmethod
    def save_overlay_visualization(
        image_path: Path,
        pred_path: Path,
        output_path: Path,
        label_path: Optional[Path] = None,
    ) -> None:
        """Save overlay visualization as PNG."""
        # Load data
        image = np.asarray(nib.load(image_path).dataobj)
        pred = np.asarray(nib.load(pred_path).dataobj)
        
        label = None
        if label_path is not None:
            label = np.asarray(nib.load(label_path).dataobj)
        
        # Create overlay
        overlay = Visualizations.create_overlay_image(image, pred, label)
        
        # Save using matplotlib
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(overlay)
        ax.axis('off')
        
        if label is not None:
            red_patch = mpatches.Patch(color='red', label='Prediction')
            green_patch = mpatches.Patch(color='green', label='Ground Truth')
            ax.legend(handles=[red_patch, green_patch], loc='upper right')
        else:
            red_patch = mpatches.Patch(color='red', label='Prediction')
            ax.legend(handles=[red_patch], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("Reporting module loaded successfully")
