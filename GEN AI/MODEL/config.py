"""
Experiment configuration management.
Supports baseline and enterprise profiles.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json
import yaml
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model selection
    model_type: str = "UNet"  # "UNet" or "SegResNet"
    
    # Data settings
    spacing: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    roi_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    
    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    
    # Scheduler
    scheduler: str = "CosineAnnealingWarmRestarts"
    scheduler_t_0: int = 10
    warmup_epochs: int = 5
    max_epochs: int = 80
    
    # Batch and accumulation
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Workers and caching
    num_workers: int = 4
    cache_rate: float = 0.1
    
    # Loss function
    loss_type: str = "DiceCE"  # "Dice", "CrossEntropy", "DiceCE", "FocalLoss"
    
    # Post-processing
    postprocess_largest_component: bool = True
    postprocess_remove_small_objects: bool = True
    postprocess_remove_small_objects_threshold: int = 50
    postprocess_fill_holes: bool = False
    
    # Validation
    early_stopping_patience: int = 20
    early_stopping_metric: str = "dice"  # "dice" or "loss"
    validation_frequency: int = 5
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Sampling
    pos_ratio: float = 0.6  # Ratio of samples that must contain foreground

    # Speed/experiment controls
    train_subset_size: Optional[int] = None
    val_subset_size: Optional[int] = None
    use_roi_curriculum: bool = True
    use_augmentation: bool = True
    
    # Device
    device: str = "cpu"  # Use "cpu" or "cuda" if available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: Path) -> None:
        """Save configuration to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: Path) -> "TrainingConfig":
        """Load configuration from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, filepath: Path) -> None:
        """Save configuration to YAML."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, filepath: Path) -> "TrainingConfig":
        """Load configuration from YAML."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ConfigFactory:
    """Create predefined experiment profiles."""
    
    @staticmethod
    def baseline() -> TrainingConfig:
        """Baseline configuration: UNet, optimized for demo."""
        return TrainingConfig(
            model_type="UNet",
            spacing=[1.0, 1.0, 1.0],
            roi_size=[192, 192, 192],
            optimizer="AdamW",
            learning_rate=2e-4,
            weight_decay=1e-5,
            scheduler="CosineAnnealingWarmRestarts",
            warmup_epochs=5,
            max_epochs=120,
            batch_size=8,
            gradient_accumulation_steps=1,
            num_workers=4,
            cache_rate=1.0,
            loss_type="DiceCE",
            early_stopping_patience=20,
            pos_ratio=0.6,
            seed=42
        )
    
    @staticmethod
    def monday_demo() -> TrainingConfig:
        """BE AIML Final Year Project: Optimized for Monday demo.
        Target: fast convergence for Monday demo on RTX 5090.
        """
        return TrainingConfig(
            model_type="UNet",
            spacing=[1.0, 1.0, 1.0],
            roi_size=[128, 128, 128],
            optimizer="AdamW",
            learning_rate=4e-4,
            weight_decay=1e-5,
            scheduler="CosineAnnealingWarmRestarts",
            warmup_epochs=2,
            max_epochs=60,
            batch_size=8,
            gradient_accumulation_steps=1,
            num_workers=8,
            cache_rate=0.1,
            loss_type="DiceCE",
            early_stopping_patience=12,
            validation_frequency=3,
            pos_ratio=0.8,
            postprocess_largest_component=True,
            postprocess_remove_small_objects=True,
            postprocess_remove_small_objects_threshold=50,
            seed=42,
            device="cuda"
        )

    @staticmethod
    def smoke_demo() -> TrainingConfig:
        """Ultra-fast proof run for tonight.
        Prioritizes quick signal over final-quality convergence.
        """
        return TrainingConfig(
            model_type="UNet",
            spacing=[1.0, 1.0, 1.0],
            roi_size=[96, 96, 96],
            optimizer="AdamW",
            learning_rate=6e-4,
            weight_decay=1e-5,
            scheduler="CosineAnnealingWarmRestarts",
            warmup_epochs=1,
            max_epochs=15,
            batch_size=12,
            gradient_accumulation_steps=1,
            num_workers=0,
            cache_rate=0.0,
            loss_type="DiceCE",
            early_stopping_patience=5,
            validation_frequency=5,
            pos_ratio=0.9,
            train_subset_size=96,
            val_subset_size=16,
            use_roi_curriculum=False,
            use_augmentation=False,
            postprocess_largest_component=True,
            postprocess_remove_small_objects=True,
            postprocess_remove_small_objects_threshold=50,
            seed=42,
            device="cuda"
        )
    
    @staticmethod
    def enterprise() -> TrainingConfig:
        """Enterprise configuration: SegResNet, aggressive optimization."""
        return TrainingConfig(
            model_type="SegResNet",
            spacing=[1.0, 1.0, 1.0],
            roi_size=[160, 160, 160],
            optimizer="AdamW",
            learning_rate=5e-4,
            weight_decay=1e-4,
            scheduler="CosineAnnealingWarmRestarts",
            warmup_epochs=10,
            max_epochs=200,
            batch_size=2,
            gradient_accumulation_steps=2,
            num_workers=8,
            cache_rate=1.0,
            loss_type="DiceCE",
            early_stopping_patience=20,
            postprocess_remove_small_objects_threshold=100,
            seed=42
        )

    @staticmethod
    def high_vram_segresnet() -> TrainingConfig:
        """High-VRAM SegResNet configuration."""
        return TrainingConfig(
            model_type="SegResNet",
            spacing=[1.0, 1.0, 1.0],
            roi_size=[256, 256, 256],
            optimizer="AdamW",
            learning_rate=5e-4,
            weight_decay=1e-4,
            scheduler="CosineAnnealingWarmRestarts",
            warmup_epochs=5,
            max_epochs=200,
            batch_size=4,
            gradient_accumulation_steps=2,
            num_workers=4,
            cache_rate=0.0,
            loss_type="DiceFocal",
            early_stopping_patience=50,
            pos_ratio=0.5,
            seed=42,
            device="cuda"
        )

    @staticmethod
    def elite() -> TrainingConfig:
        """Elite MICCAI 2024 configuration: SegResNet with curriculum learning and composite loss."""
        return TrainingConfig(
            model_type="SegResNet",  # nnU-Net ResEnc equivalent
            spacing=[1.0, 1.0, 1.0],
            roi_size=[256, 256, 256],  # Final size, curriculum starts at 96^3
            optimizer="AdamW",
            learning_rate=1e-3,
            weight_decay=1e-5,
            scheduler="CosineAnnealingWarmRestarts",
            warmup_epochs=5,
            max_epochs=200,  # Extended for elite run
            batch_size=4,
            gradient_accumulation_steps=1,
            num_workers=4,
            cache_rate=0.5,
            loss_type="elite_composite",  # 0.7×DiceFocal + 0.3×ClassAdaptiveDice
            early_stopping_patience=25,
            pos_ratio=0.5,
            seed=42,
            device="cuda"
        )
    
    @staticmethod
    def high_vram_resenc() -> TrainingConfig:
        """High-VRAM nnU-Net ResEnc configuration for RTX 5090 (32GB).
        Effective batch size = batch_size * grad_accum = 4 * 4 = 16.
        """
        return TrainingConfig(
            model_type="UNetResEnc",
            spacing=[1.0, 1.0, 1.0],
            roi_size=[256, 256, 256],
            optimizer="AdamW",
            learning_rate=1e-4,
            weight_decay=1e-5,
            scheduler="CosineAnnealingWarmRestarts",
            scheduler_t_0=25,
            warmup_epochs=10,
            max_epochs=120,
            batch_size=4,                   # Was 2 → 4 (fill that 32GB)
            gradient_accumulation_steps=4,   # Was 2 → 4 (effective BS=16)
            num_workers=2,
            cache_rate=0.0, # Must be 0 for dynamic epoch curriculum
            loss_type="DiceFocal",
            early_stopping_patience=50,
            pos_ratio=0.5,
            seed=42,
            device="cuda"
        )
    
    @staticmethod
    def high_vram_mednext() -> TrainingConfig:
        """High-VRAM MedNeXt-L configuration."""
        return TrainingConfig(
            model_type="MedNeXt",
            spacing=[1.0, 1.0, 1.0],
            roi_size=[128, 128, 128], # MedNeXt is heavier, final size smaller
            optimizer="AdamW",
            learning_rate=1e-3,
            weight_decay=1e-5,
            scheduler="CosineAnnealingWarmRestarts",
            warmup_epochs=5,
            max_epochs=200,
            batch_size=1,
            gradient_accumulation_steps=4,
            num_workers=4,
            cache_rate=0.0,
            loss_type="DiceFocal",
            early_stopping_patience=50,
            pos_ratio=0.5,
            seed=42,
            device="cuda"
        )
    
    @staticmethod
    def from_name(name: str) -> TrainingConfig:
        """Create config from profile name."""
        profiles = {
            "baseline": ConfigFactory.baseline,
            "monday_demo": ConfigFactory.monday_demo,
            "smoke_demo": ConfigFactory.smoke_demo,
            "enterprise": ConfigFactory.enterprise,
            "high_vram_segresnet": ConfigFactory.high_vram_segresnet,
            "high_vram_resenc": ConfigFactory.high_vram_resenc,
            "high_vram_mednext": ConfigFactory.high_vram_mednext,
            "elite": ConfigFactory.elite,
        }
        if name not in profiles:
            raise ValueError(f"Unknown profile: {name}. Available: {list(profiles.keys())}")
        return profiles[name]()


if __name__ == "__main__":
    # Test config creation
    baseline = ConfigFactory.baseline()
    print(f"Baseline config: {baseline}")
    
    enterprise = ConfigFactory.enterprise()
    print(f"Enterprise config: {enterprise}")
