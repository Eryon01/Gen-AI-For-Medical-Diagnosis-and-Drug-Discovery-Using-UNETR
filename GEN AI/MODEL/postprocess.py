"""
Post-processing of predictions.
Includes largest component filtering, small object removal, and hole filling.
"""

from typing import Tuple
import numpy as np
from scipy import ndimage


class PostProcessor:
    """Apply post-processing to segmentation predictions."""
    
    @staticmethod
    def largest_component(pred: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        if np.sum(pred) == 0:
            return pred
        
        labeled, num_features = ndimage.label(pred)
        
        if num_features == 0:
            return pred
        
        # Find the largest component
        sizes = ndimage.sum(pred, labeled, range(num_features + 1))
        largest_label = np.argmax(sizes)
        
        # Keep only largest component
        result = (labeled == largest_label).astype(np.uint8)
        return result
    
    @staticmethod
    def remove_small_objects(pred: np.ndarray, threshold: int = 50) -> np.ndarray:
        """Remove small connected components below size threshold."""
        if np.sum(pred) == 0:
            return pred
        
        labeled, num_features = ndimage.label(pred)
        
        if num_features == 0:
            return pred
        
        # Find sizes of each component
        sizes = ndimage.sum(pred, labeled, range(num_features + 1))
        
        # Create mask for components larger than threshold
        mask = sizes > threshold
        
        # Apply mask
        result = mask[labeled]
        return result.astype(np.uint8)
    
    @staticmethod
    def fill_holes(pred: np.ndarray) -> np.ndarray:
        """Fill holes in the prediction."""
        if np.sum(pred) == 0:
            return pred
        
        # Use binary_fill_holes for 2D-like operations on each slice
        result = ndimage.binary_fill_holes(pred).astype(np.uint8)
        return result
    
    @staticmethod
    def apply_postprocessing(
        pred: np.ndarray,
        largest_component: bool = True,
        remove_small_objects: bool = True,
        remove_small_objects_threshold: int = 50,
        fill_holes: bool = False,
    ) -> np.ndarray:
        """Apply complete post-processing pipeline."""
        result = pred.copy()
        
        if largest_component:
            result = PostProcessor.largest_component(result)
        
        if remove_small_objects:
            result = PostProcessor.remove_small_objects(
                result, 
                threshold=remove_small_objects_threshold
            )
        
        if fill_holes:
            result = PostProcessor.fill_holes(result)
        
        return result


if __name__ == "__main__":
    # Test post-processing
    pred = np.random.rand(128, 128, 128) > 0.7
    
    print(f"Original volume: {np.sum(pred)}")
    
    pp = PostProcessor()
    result = pp.apply_postprocessing(
        pred,
        largest_component=True,
        remove_small_objects=True,
        remove_small_objects_threshold=50,
        fill_holes=False,
    )
    print(f"Post-processed volume: {np.sum(result)}")
