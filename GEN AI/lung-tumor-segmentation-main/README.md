# 🫁 Lung Tumor Segmentation

3D lung tumor segmentation pipeline using **MONAI** and **PyTorch** on the [NSCLC-Radiomics (LUNG1)](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics) dataset.

## Overview

This project implements a deep learning pipeline for automated 3D segmentation of lung tumors from CT scans. It was developed as a final-year major project.

### Models
- **UNet** (MONAI) — primary segmentation model
- **SegResNet** (MONAI) — alternative architecture

### Key Features
- DICOM → NIfTI conversion pipeline
- Weighted loss functions for handling class imbalance (small tumor volumes)
- Deep supervision support
- Optimized DataLoader for high-performance GPU training (RTX 5090)
- Inference notebook with 3D visualization

## Project Structure

```
├── major.ipynb                    # Main training notebook
├── major_production_high_perf.py  # Production training script
├── major_runner.py                # Training runner
├── inference.ipynb                # Inference & visualization
├── segresnet.ipynb                # SegResNet experiments
├── pipeline_config.py             # Pipeline configuration
├── convert_dicom_to_nifti.py      # DICOM → NIfTI conversion
├── requirements.txt               # Dependencies
└── dataset.json                   # Dataset split configuration
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `monai`, `nibabel`, `matplotlib`, `numpy`, `SimpleITK`

## Hardware

Trained on **NVIDIA RTX 5090** with 68GB RAM.

## Results

Sample segmentation outputs are included as `output*.png` and `training.png`.
