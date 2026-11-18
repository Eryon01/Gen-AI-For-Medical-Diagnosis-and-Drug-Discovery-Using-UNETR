"""
Loss functions and loss factory.
Supports Dice, CrossEntropy, DiceCE, and FocalLoss.
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss, DiceCELoss, DiceFocalLoss


class ClassAdaptiveDiceLoss(nn.Module):
    """Dice loss that ignores patches with no foreground voxels."""
    
    def __init__(
        self,
        include_background: bool = False,
        to_onehot_y: bool = True,
        softmax: bool = True,
        smooth_nr: float = 1e-6,
        smooth_dr: float = 1e-6,
    ):
        super().__init__()
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass with class-adaptive behavior."""
        # Check if target has any foreground voxels (class 1+)
        # Standardize: Class 0 is ALWAYS background.
        if target.shape[1] > 1:
            # One-hot encoding: exclude channel 0
            has_foreground = (target[:, 1:] > 0).any(dim=tuple(range(1, target.ndim)))
        else:
            # Index encoding: 0 is background, >0 is foreground
            has_foreground = (target > 0).any(dim=tuple(range(1, target.ndim)))
        
        # If no foreground voxels in any sample in the batch, return zero loss
        # Note: In a multi-sample batch, if only some have foreground, 
        # DiceLoss itself handles the per-sample reduction. 
        # This check is a 'short circuit' for entirely background batches.
        if not has_foreground.any():
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        
        # Otherwise, compute Dice loss
        return self.dice_loss(input, target)


class EliteCompositeLoss(nn.Module):
    """Composite loss: 0.7 × DiceFocal + 0.3 × ClassAdaptiveDice."""
    
    def __init__(
        self,
        dice_focal_kwargs: dict = None,
        class_adaptive_kwargs: dict = None,
    ):
        super().__init__()
        if dice_focal_kwargs is None:
            dice_focal_kwargs = {}
        if class_adaptive_kwargs is None:
            class_adaptive_kwargs = {}
        
        self.dice_focal = LossFactory.create_dice_focal_loss(**dice_focal_kwargs)
        self.class_adaptive = ClassAdaptiveDiceLoss(**class_adaptive_kwargs)
        self.lambda_dice_focal = 0.8
        self.lambda_class_adaptive = 0.2
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        loss_dice_focal = self.dice_focal(input, target)
        loss_class_adaptive = self.class_adaptive(input, target)
        
        return self.lambda_dice_focal * loss_dice_focal + self.lambda_class_adaptive * loss_class_adaptive


class DeepSupervisionLossWrapper(nn.Module):
    """Wraps any loss function to support a list/tuple of outputs from Deep Supervision."""
    
    def __init__(self, loss_fn: nn.Module, weights: list = [1.0, 0.5, 0.25, 0.125]):
        super().__init__()
        self.loss_fn = loss_fn
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
    def forward(self, inputs, target) -> torch.Tensor:
        """Forward pass handling either a single output or a list of deep supervision outputs."""
        # Phase 8: Deep Supervision Alignment
        # Handle 6D stacked tensors [Batch, Scale, Channel, D, H, W]
        if isinstance(inputs, torch.Tensor) and inputs.dim() == 6:
            inputs = [inputs[:, i] for i in range(inputs.shape[1])]
            
        if not isinstance(inputs, (list, tuple)):
            # Fallback: single output
            if inputs.shape[2:] != target.shape[2:]:
                target_ds = torch.nn.functional.interpolate(
                    target.float(), size=inputs.shape[2:], mode="nearest"
                )
                return self.loss_fn(inputs, target_ds)
            return self.loss_fn(inputs, target)
        
        total_loss = 0.0
        for i, out in enumerate(inputs):
            if i >= len(self.weights):
                break
            
            # Triple-Check Audit: Target-Downsampling (The VRAM/SOTA Fix)
            # Instead of upsampling 8x8x8 -> 256x256x256, we downsample labels.
            target_ds = target
            if out.shape[2:] != target.shape[2:]:
                target_ds = torch.nn.functional.interpolate(
                    target.float(), size=out.shape[2:], mode="nearest"
                )
            
            total_loss += self.weights[i] * self.loss_fn(out, target_ds)
            
        return total_loss


class LossFactory:
    """Create and configure loss functions."""
    
    @staticmethod
    def create_dice_loss(
        include_background: bool = False,
        to_onehot_y: bool = True,
        softmax: bool = True,
        smooth_nr: float = 1e-6,
        smooth_dr: float = 1e-6,
    ) -> DiceLoss:
        """Create Dice loss."""
        return DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
    
    @staticmethod
    def create_cross_entropy_loss(
        weight: torch.Tensor = None,
        reduction: str = 'mean',
    ) -> nn.CrossEntropyLoss:
        """Create Cross Entropy loss."""
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    @staticmethod
    def create_dice_ce_loss(
        include_background: bool = True,
        to_onehot_y: bool = True,
        ce_weight: torch.Tensor = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> DiceCELoss:
        """Create DiceCE (combined Dice and Cross Entropy) loss."""
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    
    @staticmethod
    def create_focal_loss(
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ) -> FocalLoss:
        """Create Focal loss."""
        return FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction=reduction,
        )
    
    @staticmethod
    def create_dice_focal_loss(
        include_background: bool = True,
        to_onehot_y: bool = True,
        softmax: bool = True,
        sigmoid: bool = False,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> DiceCELoss:
        """Create DiceCE (combined Dice and Cross Entropy) loss.
        Replaces DiceFocal for improved numerical stability in MICCAI 2024 shootout.
        """
        from monai.losses import DiceCELoss
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            sigmoid=sigmoid,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    
    @staticmethod
    def create_class_adaptive_dice_loss(**kwargs) -> ClassAdaptiveDiceLoss:
        """Create Class-Adaptive Dice loss."""
        return ClassAdaptiveDiceLoss(**kwargs)
    
    @staticmethod
    def create_elite_composite_loss(**kwargs) -> EliteCompositeLoss:
        """Create Elite Composite loss."""
        return EliteCompositeLoss(**kwargs)
    
    @staticmethod
    def create_loss(loss_type: str = "DiceCE", **kwargs) -> nn.Module:
        """Create loss function from type name."""
        loss_type = loss_type.lower()
        
        if loss_type == "dice":
            base_loss = LossFactory.create_dice_loss(**kwargs)
        elif loss_type == "crossentropy" or loss_type == "ce":
            base_loss = LossFactory.create_cross_entropy_loss(**kwargs)
        elif loss_type == "dicece" or loss_type == "dice_ce":
            base_loss = LossFactory.create_dice_ce_loss(**kwargs)
        elif loss_type == "focal":
            base_loss = LossFactory.create_focal_loss(**kwargs)
        elif loss_type == "dicefocal" or loss_type == "dice_focal":
            base_loss = LossFactory.create_dice_focal_loss(**kwargs)
        elif loss_type == "class_adaptive_dice":
            base_loss = LossFactory.create_class_adaptive_dice_loss(**kwargs)
        elif loss_type == "elite_composite":
            base_loss = LossFactory.create_elite_composite_loss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        # Wrap all losses to support deep supervision gracefully
        # Elite Standard: 5 levels of Deep Supervision for UNetResEnc
        return DeepSupervisionLossWrapper(base_loss, weights=[1.0, 0.5, 0.25, 0.125, 0.0625])


if __name__ == "__main__":
    # Test loss creation
    dice_loss = LossFactory.create_loss("dice")
    print(f"Dice loss: {dice_loss}")
    
    dicece_loss = LossFactory.create_loss("dicece")
    print(f"DiceCE loss: {dicece_loss}")
    
    focal_loss = LossFactory.create_loss("focal")
    print(f"Focal loss: {focal_loss}")
