"""
Data loading utilities for training.
Handles transforms, caching, and batching.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, Spacingd, 
    Orientationd, RandCropByPosNegLabeld, SpatialPadd,
    CenterSpatialCropd, EnsureChannelFirstd,
    AsDiscreted, CastToTyped, EnsureTyped, ScaleIntensityRanged,
    RandFlipd, RandRotate90d
)
from monai.data import Dataset, DataLoader, CacheDataset, list_data_collate

from project_v2.manifest import CaseMetadata


class ROICurriculumScheduler:
    """Scheduler for ROI size curriculum learning."""
    
    def __init__(
        self,
        start_size: int = 96,
        end_size: int = 256,
        curriculum_epochs: int = 50,
    ):
        """Initialize ROI curriculum scheduler.
        
        Args:
            start_size: Starting patch size (cubic)
            end_size: Final patch size (cubic)
            curriculum_epochs: Number of epochs for curriculum
        """
        self.start_size = start_size
        self.end_size = end_size
        self.curriculum_epochs = curriculum_epochs
    
    def get_roi_size(self, epoch: int) -> Tuple[int, int, int]:
        """Get ROI size for current epoch.
        Snaps to multiples of 32 to ensure clean spatial dim halving at all DynUNet stages.
        """
        if epoch >= self.curriculum_epochs:
            size = self.end_size
        else:
            # Linear interpolation
            progress = epoch / self.curriculum_epochs
            size = int(self.start_size + (self.end_size - self.start_size) * progress)
        
        # Snap to nearest multiple of 32 (required for 5 stride-2 stages: 2^5=32)
        size = max(32, (size // 32) * 32)
        
        return (size, size, size)


class SegmentationDataset(Dataset):
    """Custom dataset for segmentation."""
    
    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        roi_size: Tuple[int, int, int],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        transforms=None,
        cache: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize dataset."""
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.roi_size = roi_size
        self.spacing = spacing
        self.transforms = transforms
        self.cache = cache
        self.cache_dir = cache_dir
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single case."""
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load data with nibabel
        img_nib = nib.load(image_path)
        label_nib = nib.load(label_path)
        
        image = np.asarray(img_nib.dataobj).astype(np.float32)
        label = np.asarray(label_nib.dataobj).astype(np.float32)
        
        # Add channel dimension if needed
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        if label.ndim == 3:
            label = label[np.newaxis, ...]
        
        # Normalize intensity
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        
        # Pad to ROI size - consistent padding for all samples
        # Build padding tuple for all spatial dims
        pad_list = [(0, 0)]  # No padding for channel dim
        for dim in range(len(self.roi_size)):
            current_size = image.shape[dim + 1]
            target_size = self.roi_size[dim]
            if current_size < target_size:
                pad_size = target_size - current_size
                pad_before = pad_size // 2
                pad_after = pad_size - pad_before
                pad_list.append((pad_before, pad_after))
            else:
                # Crop if larger than target
                pad_list.append((0, 0))
        
        image = np.pad(image, pad_list, mode='constant', constant_values=0)
        label = np.pad(label, pad_list, mode='constant', constant_values=0)
        
        # Crop if any dimension is larger than target
        for dim in range(len(self.roi_size)):
            if image.shape[dim + 1] > self.roi_size[dim]:
                start = (image.shape[dim + 1] - self.roi_size[dim]) // 2
                end = start + self.roi_size[dim]
                if dim == 0:
                    image = image[:, start:end, :, :]
                    label = label[:, start:end, :, :]
                elif dim == 1:
                    image = image[:, :, start:end, :]
                    label = label[:, :, start:end, :]
                else:
                    image = image[:, :, :, start:end]
                    label = label[:, :, :, start:end]
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        return image, label


class DataLoaderFactory:
    """Create data loaders with proper transforms."""
    
    @staticmethod
    def create_train_transforms(
        roi_size: Tuple[int, int, int],
        pos_ratio: float = 0.5,
        use_augmentation: bool = True,
    ):
        """Create training transforms with foreground sampling."""
        transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
            # SOTA Resurrection: Resample to 1.0mm isotropic spacing
            Spacingd(
                keys=["image", "label"], 
                pixdim=(1.0, 1.0, 1.0), 
                mode=("bilinear", "nearest")
            ),
            # MICCAI 2024 Elite Standard: Lung Windowing [-1000, 1000]
            ScaleIntensityRanged(
                keys=["image"], 
                a_min=-1000, a_max=1000, 
                b_min=0.0, b_max=1.0, 
                clip=True
            ),
            # Ensure labels are binary [0, 1]
            AsDiscreted(keys=["label"], threshold=0.5),
            # Pad if image is smaller than ROI size
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            
            # Foreground/Background sampling
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=pos_ratio,
                neg=1 - pos_ratio,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
        ]

        if use_augmentation:
            transforms.extend([
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
            ])

        transforms.extend([
            CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.long]),
            EnsureTyped(keys=["image", "label"]),
        ])

        return Compose(transforms)
    
    @staticmethod
    def create_val_transforms(
        roi_size: Tuple[int, int, int],
    ):
        """Create validation transforms."""
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
            # SOTA Resurrection: Resample to 1.0mm isotropic spacing
            Spacingd(
                keys=["image", "label"], 
                pixdim=(1.0, 1.0, 1.0), 
                mode=("bilinear", "nearest")
            ),
            # MICCAI 2024 Elite Standard: Lung Windowing [-1000, 1000]
            ScaleIntensityRanged(
                keys=["image"], 
                a_min=-1000, a_max=1000, 
                b_min=0.0, b_max=1.0, 
                clip=True
            ),
            AsDiscreted(keys=["label"], threshold=0.5),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
            CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.long]),
            EnsureTyped(keys=["image", "label"]),
        ])
    
    @staticmethod
    def create_train_loader(
        image_paths: List[str],
        label_paths: List[str],
        batch_size: int,
        roi_size: Tuple[int, int, int],
        num_workers: int = 4,
        cache_rate: float = 0.0,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        pos_ratio: float = 0.5,
        use_augmentation: bool = True,
    ) -> DataLoader:
        """Create training data loader."""
        data = [{"image": i, "label": l} for i, l in zip(image_paths, label_paths)]
        transforms = DataLoaderFactory.create_train_transforms(roi_size, pos_ratio, use_augmentation)
        
        if cache_rate > 0:
            dataset = CacheDataset(data=data, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
        else:
            dataset = Dataset(data=data, transform=transforms)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=list_data_collate,
            drop_last=True,
        )
    
    @staticmethod
    def create_val_loader(
        image_paths: List[str],
        label_paths: List[str],
        batch_size: int,
        roi_size: Tuple[int, int, int],
        num_workers: int = 4,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> DataLoader:
        """Create validation data loader."""
        data = [{"image": i, "label": l} for i, l in zip(image_paths, label_paths)]
        transforms = DataLoaderFactory.create_val_transforms(roi_size)
        
        dataset = Dataset(data=data, transform=transforms)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=list_data_collate,
            drop_last=True,
        )


if __name__ == "__main__":
    print("Data loading module loaded successfully")
