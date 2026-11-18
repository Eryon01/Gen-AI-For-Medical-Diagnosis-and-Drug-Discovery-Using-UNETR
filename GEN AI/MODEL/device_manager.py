"""
Device management for mixed GPU/CPU training.
Supports 70% CUDA / 30% CPU ratio when both are available.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import psutil
import GPUtil


class DeviceManager:
    """Manage device allocation and batch splitting."""
    
    def __init__(self, primary_ratio: float = 1.0):
        """
        Initialize device manager.
        
        Args:
            primary_ratio: Ratio of batch for primary device (0.0-1.0)
                          Default 1.0 = 100% to primary (RTX 5090 Elite)
        """
        self.primary_ratio = primary_ratio
        self.primary_device = self._detect_primary_device()
        self.secondary_device = self._detect_secondary_device()
        
        print(f"Primary device: {self.primary_device}")
        print(f"Secondary device: {self.secondary_device}")
        print(f"Batch allocation: {self.primary_ratio*100:.0f}% to primary, {(1-self.primary_ratio)*100:.0f}% to secondary")
        
        # Print GPU info if available
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    @staticmethod
    def _detect_primary_device() -> str:
        """Detect primary device (prefer CUDA if available)."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @staticmethod
    def _detect_secondary_device() -> Optional[str]:
        """Detect secondary device."""
        if torch.cuda.is_available():
            # If CUDA is primary, use CPU as secondary
            return "cpu"
        return None
    
    def print_utilization(self):
        """Print current GPU and CPU utilization."""
        print("\n--- Resource Utilization ---")
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_percent}%")
        
        # GPU utilization
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    print(f"GPU Usage: {gpu.load*100:.1f}%")
                    print(f"GPU Memory: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB ({gpu.memoryUtil*100:.1f}%)")
                    
                    # PyTorch GPU memory
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"PyTorch GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except:
                print("GPU monitoring unavailable (GPUtil not installed)")
        print("---------------------------\n")
    
    def split_batch(self, 
                   images: torch.Tensor, 
                   labels: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Split batch between primary and secondary devices.
        
        Returns:
            Tuple of (image_tensors, label_tensors) where each is a list of tensors on appropriate device
        """
        if self.secondary_device is None:
            # Only one device available, use primary for everything
            return (
                [images.to(self.primary_device)],
                [labels.to(self.primary_device)]
            )
        
        # Calculate split point
        batch_size = images.shape[0]
        split_idx = max(1, int(batch_size * self.primary_ratio))
        
        # Split batch
        images_primary = images[:split_idx].to(self.primary_device)
        labels_primary = labels[:split_idx].to(self.primary_device)
        
        images_secondary = images[split_idx:].to(self.secondary_device)
        labels_secondary = labels[split_idx:].to(self.secondary_device)
        
        return (
            [images_primary, images_secondary],
            [labels_primary, labels_secondary]
        )
    
    def move_model_to_primary(self, model: nn.Module) -> nn.Module:
        """Move model to primary device."""
        return model.to(self.primary_device)
    
    def combine_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine outputs from multiple devices.
        
        Args:
            outputs: List of tensors from different devices
            
        Returns:
            Combined tensor on primary device
        """
        if len(outputs) == 1:
            return outputs[0].to(self.primary_device)
        
        # Move all to primary device and concatenate
        combined = torch.cat([out.to(self.primary_device) for out in outputs], dim=0)
        return combined
    
    def combine_losses(self, losses: List[float], batch_counts: List[int]) -> float:
        """
        Combine losses from multiple devices with proper weighting.
        
        Args:
            losses: List of loss values from each device
            batch_counts: Number of samples processed on each device
            
        Returns:
            Weighted average loss
        """
        total_samples = sum(batch_counts)
        if total_samples == 0:
            return 0.0
        
        weighted_loss = sum(l * c for l, c in zip(losses, batch_counts)) / total_samples
        return weighted_loss


class MixedDeviceTrainer:
    """Wrapper for mixed device training support."""
    
    def __init__(self, device_manager: DeviceManager):
        """Initialize mixed device trainer."""
        self.device_manager = device_manager
    
    def forward_and_loss(self,
                        model: nn.Module,
                        images: torch.Tensor,
                        labels: torch.Tensor,
                        loss_fn: nn.Module) -> float:
        """
        Forward pass and loss computation across devices.
        
        Args:
            model: Model to use
            images: Batch of images
            labels: Batch of labels
            loss_fn: Loss function
            
        Returns:
            Combined loss value
        """
        # Split batch
        image_splits, label_splits = self.device_manager.split_batch(images, labels)
        
        total_loss = 0.0
        batch_counts = []
        
        # Process each split
        for img_split, label_split in zip(image_splits, label_splits):
            if img_split.shape[0] == 0:
                continue
            
            # Forward pass
            with torch.no_grad() if not model.training else torch.enable_grad():
                outputs = model(img_split)
                loss = loss_fn(outputs, label_split)
            
            batch_count = img_split.shape[0]
            batch_counts.append(batch_count)
            total_loss += loss.item() * batch_count
        
        # Average loss
        if batch_counts:
            total_loss = total_loss / sum(batch_counts)
        
        return total_loss
    
    def compute_dice(self,
                    model: nn.Module,
                    images: torch.Tensor,
                    labels: torch.Tensor) -> float:
        """
        Compute Dice score across devices.
        
        Args:
            model: Model to use
            images: Batch of images
            labels: Batch of labels
            
        Returns:
            Combined Dice score
        """
        # Split batch
        image_splits, label_splits = self.device_manager.split_batch(images, labels)
        
        dice_scores = []
        
        for img_split, label_split in zip(image_splits, label_splits):
            if img_split.shape[0] == 0:
                continue
            
            # Forward pass
            with torch.no_grad():
                outputs = model(img_split)
                pred = torch.softmax(outputs, dim=1).argmax(dim=1)
            
            # Compute Dice
            pred = (pred > 0).float()
            target = (label_split > 0).float()
            
            intersection = torch.sum(pred * target)
            union = torch.sum(pred) + torch.sum(target)
            dice = (2 * intersection) / (union + 1e-6)
            
            dice_scores.append(dice.item())
        
        # Average Dice
        if dice_scores:
            return sum(dice_scores) / len(dice_scores)
        return 0.0


if __name__ == "__main__":
    # Test device manager
    dm = DeviceManager(primary_ratio=0.7)
    
    # Test batch splitting
    images = torch.randn(4, 1, 128, 128, 128)
    labels = torch.randint(0, 2, (4, 1, 128, 128, 128))
    
    img_splits, label_splits = dm.split_batch(images, labels)
    print(f"Image splits: {[img.shape for img in img_splits]}")
    print(f"Label splits: {[lbl.shape for lbl in label_splits]}")
