"""
Modular training pipeline.
Includes warmup, cosine schedule, early stopping, checkpointing, and resume logic.
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.utils import MetricReduction
from tqdm import tqdm

from project_v2.config import TrainingConfig
from project_v2.runtime import RuntimeMetadata, SeedManager
from project_v2.device_manager import DeviceManager, MixedDeviceTrainer
from project_v2.datasets import DataLoaderFactory, ROICurriculumScheduler


class WarmupScheduler:
    """Warmup learning rate scheduler."""
    
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, base_lr: float):
        """Initialize warmup scheduler."""
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate for warmup."""
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1


class CheckpointManager:
    """Manage checkpoint saving and loading."""
    
    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "checkpoint_last.pth"
        torch.save(checkpoint, last_path)
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            periodic_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, periodic_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: Optimizer, 
                       scheduler: Optional[LRScheduler] = None,
                       checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_last.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint.get("metrics", {}),
        }


class Trainer:
    """Main training loop."""
    
    def __init__(
        self,
        model: nn.Module,
        train_image_paths: List[str],
        train_label_paths: List[str],
        val_image_paths: List[str],
        val_label_paths: List[str],
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        loss_fn: nn.Module,
        config: TrainingConfig,
        device: str = "cpu",
        run_dir: Optional[Path] = None,
        roi_scheduler: Optional[ROICurriculumScheduler] = None,
    ):
        """Initialize trainer."""
        self.model = model
        self.train_image_paths = train_image_paths
        self.train_label_paths = train_label_paths
        self.val_image_paths = val_image_paths
        self.val_label_paths = val_label_paths
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.roi_scheduler = roi_scheduler
        self.run_dir = Path(run_dir) if run_dir else Path(".")
        
        # Initialize device manager (Defaults to 100% GPU)
        self.device_manager = DeviceManager()
        self.mixed_trainer = MixedDeviceTrainer(self.device_manager)
        self.device = self.device_manager.primary_device
        
        # Elite Resilience: AMP GradScaler
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Move model to primary device
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.run_dir / "checkpoints")
        
        # Training state
        self.best_val_metric = -float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = []
        
        # Elite Oversight: Persistent Progress Log
        self.progress_file = self.run_dir / "progress_status.json"
        
        # Elite Metrics: Medical-Grade Dice Calculation
        # Set include_background=True because we only have 1 foreground channel in our binary masks
        self.dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
        
        # Create initial data loaders
        self._last_roi_size = None
        self._update_data_loaders(epoch=0)
    
    def _update_data_loaders(self, epoch: int):
        """Update data loaders with current ROI size."""
        if self.roi_scheduler is not None:
            roi_size = self.roi_scheduler.get_roi_size(epoch)
        else:
            roi_size = (256, 256, 256)

        # Rebuild only when ROI changes; recaching every epoch is very expensive.
        if self._last_roi_size == roi_size and hasattr(self, "train_loader") and hasattr(self, "val_loader"):
            return
        
        self.train_loader = DataLoaderFactory.create_train_loader(
            image_paths=self.train_image_paths,
            label_paths=self.train_label_paths,
            batch_size=self.config.batch_size,
            roi_size=roi_size,
            num_workers=self.config.num_workers,
            cache_rate=self.config.cache_rate,
            pos_ratio=self.config.pos_ratio,
            use_augmentation=self.config.use_augmentation,
        )
        
        self.val_loader = DataLoaderFactory.create_val_loader(
            image_paths=self.val_image_paths,
            label_paths=self.val_label_paths,
            batch_size=self.config.batch_size,
            roi_size=roi_size,
            num_workers=self.config.num_workers,
        )
        self._last_roi_size = roi_size
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with Elite Resilience (AMP)."""
        self.model.train()
        losses = []
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                images, labels = batch_data["image"], batch_data["label"]
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if batch_idx % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                losses.append(loss.item() * self.config.gradient_accumulation_steps)
                pbar.set_postfix({"loss": np.mean(losses)})
        
        return {"train_loss": np.mean(losses)}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        losses = []
        dice_scores = []
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation") as pbar:
                for batch_data in pbar:
                    images, labels = batch_data["image"], batch_data["label"]
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels)
                    
                    losses.append(loss.item())
                    
                    if isinstance(outputs, torch.Tensor) and outputs.dim() == 6:
                        main_output = outputs[:, 0]
                    elif isinstance(outputs, (list, tuple)):
                        main_output = outputs[0]
                    else:
                        main_output = outputs
                    
                    if main_output.shape[2:] != labels.shape[2:]:
                        main_output = torch.nn.functional.interpolate(
                            main_output.float(), size=labels.shape[2:], mode="trilinear", align_corners=False
                        )
                    
                    # Triple-Check: Clean Metric Logic
                    # DiceMetric expects probabilities or one-hot. 
                    # We pass the discretized foreground prediction.
                    pred = (main_output.argmax(dim=1, keepdim=True) > 0).float()
                    target_bin = (labels > 0).float()
                    self.dice_metric(y_pred=pred, y=target_bin)
                    
                    pbar.set_postfix({"val_loss": np.mean(losses)})
        
        # Aggregate the metric across the entire validation set
        # MONAI aggregate() returns a tuple (mean, count) when get_not_nans=True
        val_dice_tensor, _ = self.dice_metric.aggregate()
        val_dice = val_dice_tensor.item()
        self.dice_metric.reset()
        
        return {
            "val_loss": np.mean(losses),
            "val_dice": val_dice,
        }
    
    def _save_progress_status(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Save high-level progress for external oversight."""
        # Safety: Ensure directory still exists (prevents crash if archived while running)
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch + 1,
            "total_epochs": self.config.max_epochs,
            "progress_percent": round(((epoch + 1) / self.config.max_epochs) * 100, 2),
            "current_dice": round(val_metrics.get("val_dice", 0.0), 4),
            "best_dice": round(self.best_val_metric, 4),
            "train_loss": round(train_metrics.get("train_loss", 0.0), 4),
            "val_loss": round(val_metrics.get("val_loss", 0.0), 4),
            "roi_size": self.roi_scheduler.get_roi_size(epoch) if self.roi_scheduler else [256, 256, 256]
        }
        with open(self.progress_file, 'w') as f:
            json.dump(status, f, indent=2)

    def train(self) -> Dict[str, Any]:
        """Main training loop with Elite Persistence."""
        start_epoch = 0
        checkpoint_path = self.run_dir / "checkpoints" / "checkpoint_last.pth"
        
        if checkpoint_path.exists():
            print(f"\n[PERSISTENCE] Found checkpoint at {checkpoint_path}. Resuming...")
            try:
                checkpoint_data = self.checkpoint_manager.load_checkpoint(
                    self.model, self.optimizer, self.scheduler, checkpoint_path
                )
                start_epoch = checkpoint_data.get("epoch", 0) + 1
                self.best_val_metric = checkpoint_data.get("metrics", {}).get("val_dice", -float('inf'))
                print(f"[PERSISTENCE] Successfully resumed from Epoch {start_epoch}")
            except Exception as e:
                print(f"[WARNING] Resume failed: {e}")

        # Elite Oversight: Master Progress Bar
        master_pbar = tqdm(
            range(start_epoch, self.config.max_epochs), 
            desc="🏆 TOTAL PROGRESS", 
            initial=start_epoch, 
            total=self.config.max_epochs,
            unit="epoch"
        )
        
        for epoch in master_pbar:
            self._update_data_loaders(epoch)
            gc.collect()
            torch.cuda.empty_cache()
            
            train_metrics = self.train_epoch()
            should_validate = ((epoch + 1) % self.config.validation_frequency == 0) or (epoch == start_epoch)
            if should_validate:
                val_metrics = self.validate_epoch()
            else:
                # Keep loop fast between validation checkpoints.
                val_metrics = {
                    "val_loss": float("nan"),
                    "val_dice": self.best_val_metric if self.best_val_metric > -float("inf") else 0.0,
                }
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)
            
            is_best = False
            if should_validate:
                is_best = val_metrics["val_dice"] > self.best_val_metric
                if is_best:
                    self.best_val_metric = val_metrics["val_dice"]
                    self.best_epoch = epoch
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            master_pbar.set_postfix({"Dice": f"{val_metrics['val_dice']:.4f}", "Best": f"{self.best_val_metric:.4f}"})
            self._save_progress_status(epoch, train_metrics, val_metrics)
            
            self.checkpoint_manager.save_checkpoint(
                epoch, self.model, self.optimizer, self.scheduler, 
                epoch_metrics, is_best
            )
            
            if should_validate and self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print(f"\nTraining completed. Best Dice: {self.best_val_metric:.4f} at Epoch {self.best_epoch + 1}")
        return {
            "best_dice": self.best_val_metric, 
            "best_epoch": self.best_epoch,
            "history": self.training_history
        }

    def save_training_history(self, filepath: Path) -> None:
        """Save training history."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
