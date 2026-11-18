"""
Evaluation metrics and computation.
Includes Dice, IoU, precision, recall, specificity, volume similarity, Hausdorff distance.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from monai.metrics import DiceMetric, HausdorffDistanceMetric


class MetricComputer:
    """Compute segmentation metrics."""
    
    @staticmethod
    def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """Compute Dice similarity coefficient."""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        dice = (2 * intersection + smooth) / (union + smooth)
        return float(dice)
    
    @staticmethod
    def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """Compute Intersection over Union (Jaccard)."""
        intersection = np.sum(pred * target)
        union = np.sum((pred == 1) | (target == 1))
        iou = (intersection + smooth) / (union + smooth)
        return float(iou)
    
    @staticmethod
    def precision(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """Compute precision (true positives / predicted positives)."""
        tp = np.sum(pred * target)
        fp = np.sum(pred * (1 - target))
        precision = (tp + smooth) / (tp + fp + smooth)
        return float(precision)
    
    @staticmethod
    def recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """Compute recall (true positives / condition positives)."""
        tp = np.sum(pred * target)
        fn = np.sum((1 - pred) * target)
        recall = (tp + smooth) / (tp + fn + smooth)
        return float(recall)
    
    @staticmethod
    def specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """Compute specificity (true negatives / condition negatives)."""
        tn = np.sum((1 - pred) * (1 - target))
        fp = np.sum(pred * (1 - target))
        specificity = (tn + smooth) / (tn + fp + smooth)
        return float(specificity)
    
    @staticmethod
    def volume_similarity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
        """Compute volume similarity."""
        vol_pred = np.sum(pred)
        vol_target = np.sum(target)
        vs = 1.0 - abs(vol_pred - vol_target) / (vol_pred + vol_target + smooth)
        return float(vs)
    
    @staticmethod
    def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> Optional[float]:
        """Compute Hausdorff distance in voxels."""
        # Handle empty predictions or targets
        if np.sum(pred) == 0 or np.sum(target) == 0:
            return None
        
        try:
            # Get surface voxels using distance transform
            pred_dist = distance_transform_edt(1 - pred)
            target_dist = distance_transform_edt(1 - target)
            
            # Hausdorff distance: max of min distances
            hd = max(
                np.max(np.min(target_dist[pred > 0.5])),
                np.max(np.min(pred_dist[target > 0.5]))
            )
            return float(hd)
        except:
            return None
    
    @staticmethod
    def empty_prediction_rate(pred: np.ndarray) -> float:
        """Check if prediction is empty (0 foreground voxels)."""
        return float(np.sum(pred) == 0)
    
    @staticmethod
    def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute all metrics for a case."""
        metrics = {
            "dice": MetricComputer.dice_score(pred, target),
            "iou": MetricComputer.iou_score(pred, target),
            "precision": MetricComputer.precision(pred, target),
            "recall": MetricComputer.recall(pred, target),
            "specificity": MetricComputer.specificity(pred, target),
            "volume_similarity": MetricComputer.volume_similarity(pred, target),
            "empty_prediction": MetricComputer.empty_prediction_rate(pred),
        }
        
        # Hausdorff distance (might be None for empty cases)
        hd = MetricComputer.hausdorff_distance(pred, target)
        if hd is not None:
            metrics["hausdorff_distance"] = hd
        
        return metrics


class AggregateMetrics:
    """Aggregate metrics across multiple cases."""
    
    @staticmethod
    def compute_stats(metrics_list: list) -> Dict[str, Dict[str, float]]:
        """Compute statistics from a list of metric dicts."""
        if not metrics_list:
            return {}
        
        # Collect all metric names
        metric_names = set()
        for m in metrics_list:
            metric_names.update(m.keys())
        
        stats = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            
            if metric_name == "empty_prediction":
                # This is a binary flag
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
        
        return stats
    
    @staticmethod
    def print_statistics(stats: Dict[str, Dict[str, float]]) -> None:
        """Print aggregated statistics."""
        print("\n" + "="*70)
        print("AGGREGATE METRICS STATISTICS")
        print("="*70)
        print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-"*70)
        
        for metric_name in sorted(stats.keys()):
            data = stats[metric_name]
            print(f"{metric_name:<20} {data['mean']:>10.4f} {data['std']:>10.4f} "
                  f"{data['min']:>10.4f} {data['max']:>10.4f}")
        print("="*70)


if __name__ == "__main__":
    # Test metrics
    pred = np.random.rand(128, 128, 128) > 0.5
    target = np.random.rand(128, 128, 128) > 0.5
    
    metrics = MetricComputer.compute_all_metrics(pred, target)
    print(f"Computed metrics: {metrics}")
