"""
Inference pipeline.
Load models and run inference on new data.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from tqdm import tqdm

from project_v2.models import ModelFactory
from project_v2.postprocess import PostProcessor
from project_v2.metrics import MetricComputer
from project_v2.device_manager import DeviceManager


class Inference:
    """Run inference on test data."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        checkpoint_path: Optional[Path] = None,
    ):
        """Initialize inference engine."""
        self.model = model
        
        # Initialize device manager
        if device == "auto":
            self.device_manager = DeviceManager(primary_ratio=0.7)
            self.device = self.device_manager.primary_device
        else:
            self.device_manager = None
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def predict_volume(self, image: np.ndarray, roi_size: Tuple[int, ...]) -> np.ndarray:
        """Predict on a full volume using sliding window."""
        image_tensor = torch.from_numpy(image[np.newaxis, np.newaxis, ...]).float()
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            # Simple global prediction (for small volumes)
            if all(s <= r for s, r in zip(image.shape, roi_size)):
                output = self.model(image_tensor)
                pred = torch.softmax(output, dim=1).argmax(dim=1)
                return pred[0].cpu().numpy().astype(np.uint8)
            else:
                # For larger volumes, use overlapping sliding window
                return self._sliding_window_inference(image_tensor, roi_size)
    
    def _sliding_window_inference(self, image: torch.Tensor, roi_size: Tuple[int, ...]) -> np.ndarray:
        """Sliding window inference for large volumes."""
        batch_size, channels, d, h, w = image.shape
        
        # Create output tensor
        output_shape = (batch_size, 2, d, h, w)  # 2 classes
        output = torch.zeros(output_shape, device=self.device)
        count = torch.zeros(output_shape, device=self.device)
        
        # Sliding window parameters
        step_size = tuple(s // 2 for s in roi_size)  # 50% overlap
        
        # Iterate over volume
        with torch.no_grad():
            for d_start in range(0, d, step_size[0]):
                for h_start in range(0, h, step_size[1]):
                    for w_start in range(0, w, step_size[2]):
                        d_end = min(d_start + roi_size[0], d)
                        h_end = min(h_start + roi_size[1], h)
                        w_end = min(w_start + roi_size[2], w)
                        
                        # Pad if necessary
                        patch = image[:, :, 
                                     d_start:d_end,
                                     h_start:h_end,
                                     w_start:w_end]
                        
                        if patch.shape[2:] != roi_size:
                            pad_d = roi_size[0] - patch.shape[2]
                            pad_h = roi_size[1] - patch.shape[3]
                            pad_w = roi_size[2] - patch.shape[4]
                            
                            patch = torch.nn.functional.pad(
                                patch,
                                (0, pad_w, 0, pad_h, 0, pad_d),
                                mode='constant',
                                value=0
                            )
                        
                        # Predict
                        pred_patch = self.model(patch)
                        
                        # Unpad
                        output[:, :, 
                               d_start:d_end,
                               h_start:h_end,
                               w_start:w_end] += pred_patch[:, :, 
                                                            :d_end-d_start,
                                                            :h_end-h_start,
                                                            :w_end-w_start]
                        count[:, :, 
                              d_start:d_end,
                              h_start:h_end,
                              w_start:w_end] += 1
        
        # Average overlapping regions
        output = output / (count + 1e-6)
        pred = torch.softmax(output, dim=1).argmax(dim=1)
        
        return pred[0].cpu().numpy().astype(np.uint8)
    
    def run_inference_on_cases(
        self,
        image_dir: Path,
        output_dir: Path,
        roi_size: Tuple[int, ...] = (128, 128, 128),
        postprocess_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run inference on all cases in a directory."""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for image_file in sorted(image_dir.glob("*.nii.gz")):
            case_id = image_file.stem.replace(".nii", "")
            
            print(f"\nProcessing {case_id}...")
            
            try:
                # Load image
                img_nib = nib.load(image_file)
                image = np.asarray(img_nib.dataobj)
                
                # Predict
                pred = self.predict_volume(image, roi_size)
                
                # Post-process
                if postprocess_config:
                    pred = PostProcessor.apply_postprocessing(
                        pred.astype(np.float32),
                        **postprocess_config
                    )
                
                # Save prediction
                pred_nib = nib.Nifti1Image(pred, img_nib.affine, img_nib.header)
                pred_path = output_dir / f"{case_id}_pred.nii.gz"
                nib.save(pred_nib, pred_path)
                
                results[case_id] = {
                    "status": "success",
                    "prediction_path": str(pred_path),
                    "prediction_volume": int(np.sum(pred)),
                }
            
            except Exception as e:
                print(f"Error processing {case_id}: {e}")
                results[case_id] = {
                    "status": "error",
                    "error": str(e),
                }
        
        return results
    
    def evaluate_predictions(
        self,
        pred_dir: Path,
        label_dir: Path,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate predictions against ground truth labels."""
        pred_dir = Path(pred_dir)
        label_dir = Path(label_dir)
        
        all_metrics = {}
        
        for pred_file in sorted(pred_dir.glob("*_pred.nii.gz")):
            case_id = pred_file.stem.replace("_pred", "")
            label_file = label_dir / f"{case_id}.nii.gz"
            
            if not label_file.exists():
                continue
            
            # Load files
            pred_nib = nib.load(pred_file)
            label_nib = nib.load(label_file)
            
            pred = np.asarray(pred_nib.dataobj)
            label = np.asarray(label_nib.dataobj)
            
            # Compute metrics
            metrics = MetricComputer.compute_all_metrics(pred > 0, label > 0)
            all_metrics[case_id] = metrics
        
        return all_metrics


if __name__ == "__main__":
    print("Inference module loaded successfully")
