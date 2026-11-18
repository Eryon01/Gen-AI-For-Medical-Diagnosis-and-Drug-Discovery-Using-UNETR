"""
Model factory and architecture definitions.
Supports UNet and SegResNet.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from monai.networks.nets import UNet, SegResNet, SwinUNETR, DynUNet

try:
    from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1
    HAS_MEDNEXT = True
except ImportError:
    HAS_MEDNEXT = False


class ModelFactory:
    """Create and configure models."""
    
    @staticmethod
    def create_unet(
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: List[int] = None,
        strides: List[int] = None,
        num_res_units: int = 0,
    ) -> UNet:
        """Create UNet model."""
        if channels is None:
            channels = (16, 32, 64, 128, 256)
        if strides is None:
            strides = (2, 2, 2, 2)
        
        return UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm='instance',
            dropout=0.1,
        )
    
    @staticmethod
    def create_segresnet(
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        init_filters: int = 8,
        dropout_prob: float = 0.2,
    ) -> SegResNet:
        """Create SegResNet model."""
        return SegResNet(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )
    
    @staticmethod
    def create_swinunetr(
        img_size: List[int] = (256, 256, 256),
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 48,
    ) -> SwinUNETR:
        """Create SwinUNETR model."""
        return SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
        )

    @staticmethod
    def create_unet_resenc(
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
    ) -> nn.Module:
        """Create nnU-Net ResEnc (Elite MICCAI 2024 spec).
        Widened filters for RTX 5090 32GB VRAM utilization.
        """
        # DynUNet: 5 encoder stages, wide filters, native deep supervision
        # All strides=2 ensures clean power-of-2 spatial dim halving at each level
        return DynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3, 3, 3],        # 5 encoder stages
            strides=[2, 2, 2, 2, 2],             # All stride-2 (nnU-Net gold standard)
            upsample_kernel_size=[2, 2, 2, 2, 2],# Match strides for transposed conv
            filters=[64, 128, 256, 512, 512],     # Wide filters for 32GB
            res_block=True,
            deep_supervision=True,
            deep_supr_num=2, # Supervision at 2 decoder scales
        )
    
    @staticmethod
    def create_mednext(
        in_channels: int = 1,
        out_channels: int = 2,
        model_id: str = "L",
        kernel_size: int = 5,
    ) -> nn.Module:
        """Create MedNeXt-L k5 (ConvNeXt-based challenger)."""
        if not HAS_MEDNEXT:
            raise ImportError("MedNeXt is not installed. Run: pip install git+https://github.com/MIC-DKFZ/MedNeXt.git")
        
        # MedNeXt parameters (from their official repository)
        # We enforce deep supervision and gradient checkpointing for 256^3
        return create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=out_channels,
            model_id=model_id,
            kernel_size=kernel_size,
            deep_supervision=True
        )
    
    @staticmethod
    def create_model(
        model_type: str = "UNet",
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        **kwargs
    ) -> nn.Module:
        """Create model from type name."""
        model_type = model_type.lower()
        
        # Extract img_size if present
        img_size = kwargs.pop("img_size", (256, 256, 256))
        
        if model_type == "unet":
            return ModelFactory.create_unet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        elif model_type == "segresnet":
            return ModelFactory.create_segresnet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        elif model_type == "swinunetr" or model_type == "swin":
            return ModelFactory.create_swinunetr(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        elif model_type == "unetresenc":
            return ModelFactory.create_unet_resenc(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        elif model_type == "mednext":
            return ModelFactory.create_mednext(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    unet = ModelFactory.create_unet()
    print(f"UNet parameters: {ModelFactory.count_parameters(unet):,}")
    
    segresnet = ModelFactory.create_segresnet()
    print(f"SegResNet parameters: {ModelFactory.count_parameters(segresnet):,}")
