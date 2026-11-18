# Gen-AI-For-Medical-Diagnosis-and-Drug-Discovery-Using-UNETR

A production-quality medical image segmentation pipeline built on MONAI.

## Overview

a complete redesign of the original segmentation pipeline with focus on:
- **Reproducibility**: Deterministic splits, seeded random operations, complete metadata tracking
- **Quality**: Rich evaluation metrics, comprehensive QA checks, detailed reporting
- **Modularity**: Separated concerns, reusable components, configurable experiments
- **Portability**: Self-contained artifacts, no notebook dependencies, Windows-compatible

## Pipeline Phases

### Phase 1: Data Preparation
1. **Manifest Building**: Scans image/label directories, extracts metadata (shape, spacing, orientation, voxel counts)
2. **Quality Assurance**: Checks for missing pairs, empty masks, metadata mismatches, suspicious ratios
3. **Split Generation**: Creates deterministic train/val/test splits (70/15/15) based on case ID hashing

**Outputs:**
- `artifacts/manifests/manifest_main.json` - Complete dataset metadata
- `artifacts/splits/splits_main.json` - Train/val/test case assignments

### Phase 2: Training
1. **Model Creation**: Instantiates model (UNet or SegResNet) with configurable architecture
2. **Data Loading**: Creates batched data loaders with on-the-fly transforms and augmentation
3. **Training Loop**: Trains with warmup, cosine annealing, configurable loss, early stopping
4. **Checkpointing**: Saves best/last/periodic checkpoints with full state for resume

**Outputs:**
- `artifacts/runs/<run_name>/config.json` - Resolved configuration
- `artifacts/runs/<run_name>/metadata.json` - Runtime environment info
- `artifacts/runs/<run_name>/checkpoints/` - Model checkpoints
- `artifacts/runs/<run_name>/training_history.json` - Per-epoch metrics

### Phase 3: Evaluation
1. **Inference**: Runs trained model on test set with sliding-window for large volumes
2. **Post-Processing**: Applies filters (largest component, small object removal, hole filling)
3. **Metrics Computation**: Calculates Dice, IoU, precision, recall, Hausdorff distance, etc.
4. **Reporting**: Generates aggregate and per-case evaluation reports with visualizations

**Outputs:**
- `artifacts/runs/<run_name>/predictions/` - Segmentation masks
- `artifacts/runs/<run_name>/case_metrics/` - Per-case metrics
- `artifacts/runs/<run_name>/evaluation_report.json` - Aggregate statistics

## Configuration Profiles

### Baseline (Conservative)
- **Model**: UNet with 5 feature levels
- **ROI Size**: 128×128×128 voxels
- **Batch Size**: 4
- **Epochs**: 100
- **Loss**: DiceCE (combined Dice and Cross-Entropy)
- **Learning Rate**: 1e-3

### Enterprise (Aggressive)
- **Model**: SegResNet (higher capacity)
- **ROI Size**: 160×160×160 voxels
- **Batch Size**: 2 (with 2× gradient accumulation)
- **Epochs**: 200
- **Loss**: DiceCE
- **Learning Rate**: 5e-4
- **Cache Rate**: 1.0 (cache all data in memory)

Both profiles use:
- AdamW optimizer with weight decay
- Cosine annealing scheduler with warmup
- Instance normalization
- Data augmentation (flip, rotation, affine)
- Early stopping on validation Dice

## Key Features

### Reproducibility
- ✅ Deterministic splits based on case ID hashing
- ✅ Seed control for all random operations
- ✅ Full environment metadata recorded (Python, CUDA, PyTorch, MONAI versions)
- ✅ Config fingerprint for every run

### Data Quality
- ✅ Explicit exclusion list (e.g., problematic case LUNG1-035)
- ✅ QA checks for missing pairs, empty masks, shape mismatches
- ✅ Spatial metadata tracking (affine, spacing, orientation)
- ✅ Per-case foreground ratio analysis

### Training Quality
- ✅ Warmup + cosine annealing schedule
- ✅ Multiple loss options (Dice, CrossEntropy, DiceCE, Focal)
- ✅ Gradient accumulation for large effective batch sizes
- ✅ Early stopping on validation Dice
- ✅ Best/last/periodic checkpoint management

### Evaluation Quality
- ✅ Comprehensive metrics: Dice, IoU, precision, recall, specificity, Hausdorff distance
- ✅ Per-case and aggregate reporting
- ✅ Best/worst case identification
- ✅ Overlay visualization generation
- ✅ Empty prediction detection

## Data Format

### Input
- **Image files**: `Data/images/*.nii.gz`
  - Single-channel grayscale CT/MRI scans
  - NIfTI format with affine and spacing metadata
  
- **Label files**: `Data/labels/*.nii.gz`
  - Binary segmentation masks (0 = background, 1+ = foreground)
  - Same spatial dimensions and spacing as corresponding image

### Output
- **Predictions**: NIfTI format with original spatial metadata preserved
- **Metrics**: JSON format with per-case and aggregate statistics
- **Checkpoints**: PyTorch format with model weights, optimizer state, scheduler state

## Important Notes

- The original `E:\aryan` project remains untouched as a backup
- This pipeline is deterministic and portable - runs produce identical results
- Data is loaded from the original E:\aryan directory; no copies are made
- All artifact outputs go to `artifacts/` directory under the project root
- The notebook's drug/treatment logic is explicitly marked as demo-only and not used here
- Windows compatibility is maintained - all paths use backslashes, temp files use %temp%

## References

- MONAI Documentation: https://docs.monai.io
- PyTorch Documentation: https://pytorch.org
- NIfTI Format: https://nifti.nimh.nih.gov

## Version

- **Aryan Project**: 2.0
- **Release Date**: 2026-04-02
- **Baseline Config**: UNet 128×128×128
- **Enterprise Config**: SegResNet 160×160×160
