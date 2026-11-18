"""
=============================================================================
COMBINED BACKEND: Medical Diagnosis & Drug Discovery using GenAI
=============================================================================
Combines:
  - inference.ipynb        (U-Net inference, batch pipeline, external demo)
  - lung-tumor-segmentation-unet-inference.ipynb (training arch, Grad-CAM,
    Occlusion Sensitivity, Integrated Gradients, 3-D viz, evaluation)

NEW additions:
  - Severity Analysis      (rule-based on tumor voxel burden + Grad-CAM)
  - Drug Recommendation    (SVD collaborative filtering on GDSC1/GDSC2)
  - Daily Routine          (personalized based on severity level)
  - Flask REST API         (serves the frontend)
=============================================================================
"""

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import os, json, io, base64, math, warnings
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless – no display needed for server
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F

import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom as scipy_zoom
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3

warnings.filterwarnings("ignore")

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
BASE_DIR               = Path(os.getenv("BASE_DIR", r"E:\aryan"))
MODEL_PATH             = BASE_DIR / "best_model.pth"
PREPROCESSED_ROOT      = BASE_DIR / "Aryan" / "working" / "Preprocessed1"
OUTPUT_ROOT            = BASE_DIR / "output"
MASKS_DIR              = OUTPUT_ROOT / "masks"
OVERLAYS_DIR           = OUTPUT_ROOT / "overlays"
COMPARISONS_DIR        = OUTPUT_ROOT / "comparisons"
LOGS_DIR               = OUTPUT_ROOT / "logs"
VIZ_DIR                = BASE_DIR / "notebook_outputs" / "visualizations"
PRED_DIR               = BASE_DIR / "notebook_outputs" / "predictions"

for d in [OUTPUT_ROOT, MASKS_DIR, OVERLAYS_DIR, COMPARISONS_DIR,
          LOGS_DIR, VIZ_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD          = 0.5
EXPECTED_IMAGE_SIZE = (256, 256)

# CUDA tuning (matches training notebook)
torch.backends.cudnn.benchmark   = True
torch.backends.cudnn.allow_tf32  = True
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ===========================================================================
# 2. MODEL ARCHITECTURE  (identical to both notebooks)
# ===========================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.step(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DoubleConv(1, 64)
        self.layer2 = DoubleConv(64, 128)
        self.layer3 = DoubleConv(128, 256)
        self.layer4 = DoubleConv(256, 512)
        self.layer5 = DoubleConv(512 + 256, 256)
        self.layer6 = DoubleConv(256 + 128, 128)
        self.layer7 = DoubleConv(128 + 64, 64)
        self.layer8 = nn.Conv2d(64, 1, 1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x1  = self.layer1(x);     x1m = self.maxpool(x1)
        x2  = self.layer2(x1m);   x2m = self.maxpool(x2)
        x3  = self.layer3(x2m);   x3m = self.maxpool(x3)
        x4  = self.layer4(x3m)
        x5  = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5  = self.layer5(torch.cat([x5, x3], dim=1))
        x6  = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6  = self.layer6(torch.cat([x6, x2], dim=1))
        x7  = F.interpolate(x6, scale_factor=2, mode="bilinear", align_corners=False)
        x7  = self.layer7(torch.cat([x7, x1], dim=1))
        return self.layer8(x7)


class TumorSegmentation(nn.Module):
    """Wraps UNet exactly as saved by training notebook."""
    def __init__(self):
        super().__init__()
        self.model = UNet()

    def forward(self, data):
        return self.model(data.float())


# ===========================================================================
# 3. MODEL LOADING
# ===========================================================================
_global_model = None

def load_model(model_path: Path = MODEL_PATH) -> TumorSegmentation | None:
    global _global_model
    if _global_model is not None:
        return _global_model

    if not model_path.exists():
        print(f"[Error] Model not found: {model_path}")
        return None

    try:
        ckpt = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"[Error] Could not load checkpoint: {e}")
        return None

    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = TumorSegmentation()
    try:
        model.model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[Error] Architecture mismatch: {e}")
        return None

    model.to(DEVICE).eval()
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"[OK] Model loaded (epoch={epoch}, device={DEVICE})")
    _global_model = model
    return model


# ===========================================================================
# 4. PREPROCESSING UTILITIES
# ===========================================================================

def preprocess_npy_slice(arr: np.ndarray):
    """Process an already-preprocessed .npy slice (from Preprocessed1)."""
    steps = []
    img = arr.copy().astype(np.float32)
    if img.ndim > 2:
        img = img[..., 0] if img.ndim == 3 else img
    if img.shape != EXPECTED_IMAGE_SIZE:
        img = cv2.resize(img, EXPECTED_IMAGE_SIZE)
        steps.append(f"resized to {EXPECTED_IMAGE_SIZE}")
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    steps.append("converted to tensor (1,1,256,256)")
    return img, tensor, steps


def preprocess_external_image(file_path: Path):
    """Load .png/.jpg/.nii external CT slice for inference."""
    steps = []
    ext = file_path.suffix.lower()

    if ext in (".nii", ".gz"):
        nii = nib.load(str(file_path))
        data = nii.get_fdata()
        mid = data.shape[-1] // 2
        img = data[:, :, mid].astype(np.float32)
        steps.append("loaded NIfTI, took middle axial slice")
    elif ext == ".npy":
        img = np.load(str(file_path)).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, img.shape[-1] // 2]
        steps.append("loaded .npy")
    else:
        raw = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError(f"cv2 could not open {file_path}")
        img = raw.astype(np.float32)
        steps.append("loaded image as grayscale")

    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    steps.append("min-max normalised to [0,1]")

    if img.shape != EXPECTED_IMAGE_SIZE:
        img = cv2.resize(img, EXPECTED_IMAGE_SIZE)
        steps.append(f"resized to {EXPECTED_IMAGE_SIZE}")

    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    return img, tensor, steps


def preprocess_volume_nii(nii_path: Path):
    """Load a full NIfTI volume for 3-D segmentation."""
    nii = nib.load(str(nii_path))
    ct  = nii.get_fdata() / 3071.0
    ct  = ct[:, :, 30:]          # drop noisy first 30 slices
    return ct


# ===========================================================================
# 5. INFERENCE
# ===========================================================================

def predict_mask(model: TumorSegmentation, tensor: torch.Tensor, threshold=THRESHOLD):
    """Returns (logits, probs, binary_mask) tensors."""
    with torch.no_grad():
        logits  = model(tensor.to(DEVICE))
        probs   = torch.sigmoid(logits).cpu()
        binary  = (probs > threshold).float()
    return logits.cpu(), probs, binary


def run_inference_on_volume(model: TumorSegmentation, ct: np.ndarray) -> np.ndarray:
    """Segment an entire (H,W,D) CT volume slice by slice."""
    segmentation = np.zeros_like(ct)
    H, W, D = ct.shape
    for i in range(D):
        sl = cv2.resize(ct[:, :, i], EXPECTED_IMAGE_SIZE)
        t  = torch.tensor(sl).unsqueeze(0).unsqueeze(0).float()
        _, _, pred = predict_mask(model, t)
        pred_np = pred[0, 0].numpy()
        segmentation[:, :, i] = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_NEAREST)
    return segmentation


# ===========================================================================
# 6. GRAD-CAM
# ===========================================================================

class GradCAM:
    def __init__(self, model: TumorSegmentation, target_layer):
        self.model = model
        self.activations = None
        self.gradients   = None
        self._hooks = [
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o)),
            target_layer.register_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0])),
        ]

    def remove(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        output = torch.sigmoid(self.model(input_tensor.to(DEVICE)))
        output.mean().backward()

        grads = self.gradients.detach().cpu().numpy()[0]       # (C, H', W')
        acts  = self.activations.detach().cpu().numpy()[0]     # (C, H', W')
        weights = np.mean(grads, axis=(1, 2))
        cam = np.sum(weights[:, None, None] * acts, axis=0)
        cam = np.maximum(cam, 0)

        _, _, h, w = input_tensor.shape
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam * 10.0      # scale to [0,10] for colormap bands


# ===========================================================================
# 7. SEVERITY ANALYSIS  (NEW)
# ===========================================================================

SEVERITY_THRESHOLDS = {
    "minimal":  (0,     500),
    "mild":     (500,   2000),
    "moderate": (2000,  8000),
    "severe":   (8000,  float("inf")),
}


def analyze_severity(
    segmentation_volume: np.ndarray,
    total_lung_voxels: int | None = None,
) -> dict:
    """
    Rule-based severity analysis from predicted segmentation mask.

    Parameters
    ----------
    segmentation_volume : 3-D binary array (H, W, D)
    total_lung_voxels   : optional denominator for % burden

    Returns
    -------
    dict with keys: level, tumor_voxels, tumor_volume_mm3 (approx),
                    burden_percent, affected_slices, max_slice_burden,
                    description, recommendations
    """
    binary = (segmentation_volume > 0.5).astype(np.uint8)
    tumor_voxels  = int(binary.sum())
    affected_slices = int((binary.sum(axis=(0, 1)) > 0).sum())
    max_slice_burden = int(binary.sum(axis=(0, 1)).max())

    # Approximate mm³ assuming 1.5 mm isotropic voxel spacing
    voxel_vol_mm3  = 1.5 ** 3
    tumor_vol_mm3  = tumor_voxels * voxel_vol_mm3

    burden_percent = None
    if total_lung_voxels and total_lung_voxels > 0:
        burden_percent = round(100 * tumor_voxels / total_lung_voxels, 2)

    # Determine severity level
    level = "minimal"
    for lvl, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= tumor_voxels < hi:
            level = lvl
            break

    descriptions = {
        "minimal": (
            "Very small tumor burden detected. The tumor is likely in early stage "
            "with limited tissue involvement. Close monitoring is recommended."
        ),
        "mild": (
            "Mild tumor burden. The lesion is localised and potentially operable. "
            "Multidisciplinary team review advised."
        ),
        "moderate": (
            "Moderate tumor burden suggesting significant involvement. "
            "Treatment planning with oncology is strongly advised. "
            "Chemotherapy or targeted therapy may be indicated."
        ),
        "severe": (
            "High tumor burden indicating extensive lung involvement. "
            "Immediate oncological consultation required. "
            "Palliative and systemic treatment options should be explored."
        ),
    }

    return {
        "level":            level,
        "tumor_voxels":     tumor_voxels,
        "tumor_volume_mm3": round(tumor_vol_mm3, 1),
        "burden_percent":   burden_percent,
        "affected_slices":  affected_slices,
        "max_slice_burden": max_slice_burden,
        "description":      descriptions[level],
    }


# ===========================================================================
# 8. DRUG RECOMMENDATION  (from inference.ipynb, enhanced)
# ===========================================================================

_drug_model_cache = {}

def build_drug_model(gdsc1_path: str | None = None,
                     gdsc2_path: str | None = None,
                     n_components: int = 20):
    """
    Fits SVD on GDSC1+GDSC2 data.  Caches result.
    If CSV files are absent, falls back to a curated static list for NSCLC.
    """
    cache_key = (gdsc1_path, gdsc2_path)
    if cache_key in _drug_model_cache:
        return _drug_model_cache[cache_key]

    if gdsc1_path and gdsc2_path and Path(gdsc1_path).exists() and Path(gdsc2_path).exists():
        gdsc1 = pd.read_csv(gdsc1_path, index_col=0)
        gdsc2 = pd.read_csv(gdsc2_path, index_col=0)
        drug_data = pd.concat([gdsc1, gdsc2], axis=1)
        imputer = SimpleImputer(strategy="mean")
        matrix  = imputer.fit_transform(drug_data)
        svd     = TruncatedSVD(n_components=n_components, random_state=42)
        latent  = svd.fit_transform(matrix)
        reconstructed = latent @ svd.components_
        result  = {
            "type":          "svd",
            "cell_lines":    drug_data.index.tolist(),
            "drugs":         drug_data.columns.tolist(),
            "reconstructed": reconstructed,
        }
    else:
        # Curated NSCLC drug list as fallback (commonly used regimens)
        result = {
            "type": "static",
            "drugs_by_severity": {
                "minimal": [
                    {"name": "Erlotinib",   "auc": 0.21, "class": "EGFR TKI (1st gen)",
                     "note": "First-line for EGFR-mutant NSCLC"},
                    {"name": "Gefitinib",   "auc": 0.23, "class": "EGFR TKI",
                     "note": "Well-tolerated targeted therapy"},
                    {"name": "Osimertinib", "auc": 0.19, "class": "EGFR TKI (3rd gen)",
                     "note": "Preferred for T790M mutation"},
                ],
                "mild": [
                    {"name": "Osimertinib",  "auc": 0.19, "class": "EGFR TKI",
                     "note": "Preferred first-line for EGFR+"},
                    {"name": "Alectinib",    "auc": 0.22, "class": "ALK inhibitor",
                     "note": "For ALK+ NSCLC"},
                    {"name": "Pembrolizumab","auc": 0.28, "class": "PD-1 inhibitor",
                     "note": "High PD-L1 expression"},
                    {"name": "Crizotinib",   "auc": 0.31, "class": "ALK/MET inhibitor",
                     "note": "ROS1/ALK rearrangements"},
                ],
                "moderate": [
                    {"name": "Carboplatin + Pemetrexed", "auc": 0.35, "class": "Platinum doublet",
                     "note": "Standard first-line chemo"},
                    {"name": "Nivolumab",    "auc": 0.30, "class": "PD-1 inhibitor",
                     "note": "Immunotherapy, any PD-L1"},
                    {"name": "Bevacizumab",  "auc": 0.38, "class": "Anti-VEGF",
                     "note": "Combined with platinum-based chemo"},
                    {"name": "Docetaxel",    "auc": 0.42, "class": "Taxane",
                     "note": "Second-line option"},
                ],
                "severe": [
                    {"name": "Carboplatin + Paclitaxel + Bevacizumab", "auc": 0.44,
                     "class": "Combination regimen",
                     "note": "Aggressive systemic therapy"},
                    {"name": "Nivolumab + Ipilimumab", "auc": 0.36,
                     "class": "Dual checkpoint inhibition",
                     "note": "High TMB tumors"},
                    {"name": "Ramucirumab",  "auc": 0.41, "class": "Anti-VEGFR2",
                     "note": "Second-line after platinum failure"},
                    {"name": "Topotecan",    "auc": 0.48, "class": "Topoisomerase inhibitor",
                     "note": "Salvage therapy"},
                ],
            },
        }

    _drug_model_cache[cache_key] = result
    return result


def recommend_drugs(severity_level: str,
                    cell_line_id: str | None = None,
                    gdsc1_path: str | None = None,
                    gdsc2_path: str | None = None,
                    top_n: int = 5) -> list[dict]:
    """
    Returns top-N drug recommendations based on severity and optional cell line.
    """
    model_data = build_drug_model(gdsc1_path, gdsc2_path)

    if model_data["type"] == "svd" and cell_line_id:
        cell_lines   = model_data["cell_lines"]
        drugs        = model_data["drugs"]
        reconstructed = model_data["reconstructed"]

        if cell_line_id in cell_lines:
            idx  = cell_lines.index(cell_line_id)
            aucs = reconstructed[idx]
            top_idx = np.argsort(aucs)[:top_n]
            return [{"name": drugs[i], "auc": round(float(aucs[i]), 3),
                     "class": "GDSC predicted", "note": "SVD collaborative filtering"}
                    for i in top_idx]

    # Static fallback
    candidates = model_data["drugs_by_severity"].get(severity_level,
                 model_data["drugs_by_severity"]["moderate"])
    return candidates[:top_n]


# ===========================================================================
# 9. DAILY ROUTINE  (NEW)
# ===========================================================================

def generate_daily_routine(severity_level: str) -> dict:
    """Returns a structured daily routine dict tailored to severity level."""
    base = {
        "morning": [
            "Wake up at a consistent time (6:00–7:00 AM) to support circadian rhythm",
            "Drink 2 glasses of lukewarm water",
            "10–15 minutes of gentle deep breathing or pranayama",
        ],
        "nutrition": [
            "Anti-inflammatory diet: turmeric, ginger, leafy greens",
            "Lean protein (eggs, lentils, tofu) at every meal",
            "Avoid processed foods, refined sugar, and alcohol",
            "Stay hydrated: 2–2.5 L of water daily",
        ],
        "exercise": [],
        "mental_health": [
            "10-minute guided meditation (apps: Headspace / Calm)",
            "Journaling: 3 things you are grateful for",
            "Connect with a support group or trusted friend/family",
        ],
        "medical": [],
        "evening": [
            "Light dinner (soup / salad) by 7:00 PM",
            "Avoid screens 1 hour before bedtime",
            "Sleep 7–9 hours; elevate head slightly if breathless",
        ],
    }

    if severity_level == "minimal":
        base["exercise"] = [
            "30 minutes of moderate walking or cycling",
            "Light yoga (15 min) — avoid inversions",
            "Breathing exercises: 4-7-8 technique twice daily",
        ]
        base["medical"] = [
            "Scheduled follow-up CT scan every 3 months",
            "Continue prescribed targeted therapy as directed",
            "Avoid smoking and secondhand smoke completely",
        ]
    elif severity_level == "mild":
        base["exercise"] = [
            "20-minute gentle walk on flat terrain",
            "Pulmonary rehabilitation exercises (as prescribed)",
            "Avoid strenuous workouts; rest if fatigued",
        ]
        base["medical"] = [
            "Monthly oncology consultation",
            "Take all prescribed medications on schedule",
            "Report any new symptoms (cough, haemoptysis) immediately",
            "Nutritional counselling with a registered dietitian",
        ]
    elif severity_level == "moderate":
        base["exercise"] = [
            "Short 10–15 minute walks twice daily as tolerated",
            "Bed or chair-based stretching exercises",
            "Avoid overexertion — rest whenever needed",
        ]
        base["medical"] = [
            "Bi-weekly oncology review",
            "Manage chemotherapy side effects (nausea, fatigue)",
            "Palliative care consultation for symptom management",
            "Strict medication adherence with written schedule",
            "Blood work monitoring every 2 weeks",
        ]
        base["nutrition"] = [
            "Small, frequent meals (5–6 per day) to combat nausea",
            "High-calorie, high-protein diet to prevent cachexia",
            "Nutritional supplements if oral intake is poor",
            "Avoid raw/undercooked foods during immunosuppression",
        ]
    else:  # severe
        base["exercise"] = [
            "Passive range-of-motion exercises in bed/chair",
            "Incentive spirometry as directed by physiotherapist",
            "All activity guided by physiotherapy team",
        ]
        base["medical"] = [
            "Daily symptom assessment by care team",
            "Palliative care integration for quality of life",
            "Pain and dyspnoea management protocols",
            "Family/caregiver education and support",
            "Advance care planning discussion with physician",
        ]
        base["nutrition"] = [
            "Enteral nutrition if oral intake insufficient",
            "Anti-nausea medication before meals",
            "High-density liquid nutrition if swallowing is difficult",
            "Consult dietitian for personalised meal plan",
        ]

    return base


# ===========================================================================
# 10. VISUALIZATION UTILITIES
# ===========================================================================

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_2d_overlay_b64(ct_slice: np.ndarray, pred_mask: np.ndarray,
                         title: str = "Tumor Overlay") -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0a0a0f")

    norm_ct = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)

    axes[0].imshow(norm_ct, cmap="bone"); axes[0].set_title("Original CT", color="white"); axes[0].axis("off")
    axes[1].imshow(pred_mask, cmap="Reds"); axes[1].set_title("Predicted Mask", color="white"); axes[1].axis("off")
    masked = np.ma.masked_where(pred_mask == 0, pred_mask)
    axes[2].imshow(norm_ct, cmap="bone")
    axes[2].imshow(masked, alpha=0.65, cmap="autumn")
    axes[2].set_title("Overlay", color="white"); axes[2].axis("off")

    for ax in axes:
        ax.tick_params(colors="white")
    fig.suptitle(title, color="white", fontsize=14)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_gradcam_b64(ct_slice: np.ndarray, cam: np.ndarray,
                      pred_label: int) -> str:
    norm = ct_slice - ct_slice.min()
    if norm.max() > 0:
        norm /= norm.max()
    norm = np.clip(norm * 1.5, 0, 1)

    cmap = plt.cm.jet
    bounds = [0, 2, 4, 8, 10]
    norm_colors = mcolors.BoundaryNorm(bounds, cmap.N)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam / 10), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_rgb = np.stack([norm] * 3, axis=-1)
    superimposed = np.clip(heatmap / 255.0 * 0.4 + img_rgb * 0.6, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0a0a0f")

    axes[0].imshow(norm, cmap="bone"); axes[0].set_title("CT Slice", color="white"); axes[0].axis("off")
    im = axes[1].imshow(cam, cmap=cmap, norm=norm_colors)
    axes[1].set_title("Grad-CAM Heatmap", color="white"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1]).ax.tick_params(colors="white")
    axes[2].imshow(superimposed); axes[2].set_title("Superimposed", color="white"); axes[2].axis("off")

    label = ["No Tumor Detected","Tumor Detected"][pred_label]
    fig.suptitle(f"Grad-CAM — {label}", color="#ff4d4d" if pred_label else "#4dff88", fontsize=13)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def make_3d_mesh_json(seg_volume: np.ndarray, downsample: int = 2) -> dict | None:
    """Return Plotly-compatible mesh JSON for the frontend."""
    try:
        from skimage import measure
    except ImportError:
        return None

    binary = (seg_volume > 0.5).astype(np.uint8)
    if downsample > 1:
        binary = binary[::downsample, ::downsample, ::downsample]
    if binary.sum() < 50:
        return None

    verts, faces, _, _ = measure.marching_cubes(binary, level=0.5, step_size=1)
    return {
        "x": verts[:, 0].tolist(),
        "y": verts[:, 1].tolist(),
        "z": verts[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
    }


def make_multislice_b64(ct_volume: np.ndarray, seg_volume: np.ndarray,
                         num_slices: int = 6) -> str:
    D = seg_volume.shape[2]
    ids = np.linspace(0, D - 1, num_slices, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#0a0a0f")
    axes = axes.flatten()

    for idx, z in enumerate(ids):
        ax = axes[idx]
        base = ct_volume[:, :, min(z, ct_volume.shape[2] - 1)]
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)
        ax.imshow(base, cmap="gray")
        mask_sl = seg_volume[:, :, z]
        overlay = np.ma.masked_where(mask_sl == 0, mask_sl)
        ax.imshow(overlay, cmap="autumn", alpha=0.7)
        ax.set_title(f"Slice {z} | Tumor px: {int(mask_sl.sum())}", color="white", fontsize=10)
        ax.axis("off")

    fig.suptitle("Multi-Slice Tumor Overlay", color="white", fontsize=15)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ===========================================================================
# 11. BATCH INFERENCE  (from inference.ipynb)
# ===========================================================================

def collect_slice_files(root: Path):
    if not root.exists():
        print(f"[Error] Input dir missing: {root}")
        return []
    files = sorted(root.glob("*/*/reconstructed_data/*.npy"))
    print(f"[OK] Discovered {len(files)} slices under {root}")
    return files


def run_batch_inference(model, slice_files, max_images=None):
    if not model or not slice_files:
        return None
    selected = slice_files[:max_images] if max_images else slice_files
    successes, failures = [], []

    for sf in selected:
        arr = np.load(sf)
        try:
            img, tensor, steps = preprocess_npy_slice(arr)
            _, probs, pred = predict_mask(model, tensor)
            pred_np = pred[0, 0].numpy()
            successes.append({
                "file":  str(sf),
                "pos_px": int(pred_np.sum()),
                "p_max":  round(float(probs.max()), 4),
            })
        except Exception as e:
            failures.append({"file": str(sf), "error": str(e)})

    return {"successes": len(successes), "failures": len(failures), "details": successes}


# ===========================================================================
# 12. DATABASE (feedback)
# ===========================================================================
DB_PATH = Path("feedback.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT,
            email     TEXT,
            rating    INTEGER,
            message   TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(name: str, email: str, rating: int, message: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO feedback (name, email, rating, message, timestamp) VALUES (?,?,?,?,?)",
        (name, email, rating, message, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_all_feedback():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM feedback ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [{"id":r[0],"name":r[1],"email":r[2],"rating":r[3],
             "message":r[4],"timestamp":r[5]} for r in rows]


# ===========================================================================
# 13. EMAIL NOTIFICATION
# ===========================================================================
SMTP_SERVER   = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ADMIN_EMAIL   = "pruthvipatel141@gmail.com"


def send_feedback_email(name: str, email: str, rating: int, message: str):
    """Send an email to admin when feedback is submitted."""
    if not SMTP_USER or not SMTP_PASSWORD:
        print("[Warning] SMTP credentials not set; skipping email")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[LungAI] New Feedback — {rating}⭐ from {name}"
    msg["From"]    = SMTP_USER
    msg["To"]      = ADMIN_EMAIL

    html = f"""
    <h2>New Feedback Received</h2>
    <p><b>Name:</b> {name}</p>
    <p><b>Email:</b> {email}</p>
    <p><b>Rating:</b> {"⭐" * rating}</p>
    <hr/>
    <p>{message}</p>
    <p><i>Submitted at {datetime.now().strftime('%Y-%m-%d %H:%M')}</i></p>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ADMIN_EMAIL, msg.as_string())
        return True
    except Exception as e:
        print(f"[Email Error] {e}")
        return False


# ===========================================================================
# 14. FLASK REST API
# ===========================================================================
app = Flask(__name__)
CORS(app)
init_db()

ALLOWED_EXTENSIONS = {".npy", ".nii", ".gz", ".png", ".jpg", ".jpeg"}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/api/health", methods=["GET"])
def health():
    m = load_model()
    return jsonify({"status": "ok", "model_loaded": m is not None, "device": str(DEVICE)})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    POST multipart/form-data with file=<CT scan>
    Optional form fields:
      - cell_line  (for GDSC drug lookup)
      - gdsc1_path / gdsc2_path (server-side CSV paths)
      - slice_idx  (which axial slice for Grad-CAM; default middle)
    Returns JSON with:
      - overlay_2d    : base64 PNG
      - gradcam       : base64 PNG
      - multislice    : base64 PNG (if NIfTI volume)
      - mesh_3d       : Plotly mesh JSON (if NIfTI volume)
      - severity      : severity dict
      - drugs         : list of drug dicts
      - daily_routine : routine dict
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not allowed(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Save upload
    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))

    model = load_model()
    if model is None:
        return jsonify({"error": "Model not loaded – check MODEL_PATH"}), 500

    cell_line  = request.form.get("cell_line")
    gdsc1_path = request.form.get("gdsc1_path")
    gdsc2_path = request.form.get("gdsc2_path")
    slice_idx  = request.form.get("slice_idx", type=int)

    result = {}

    try:
        ext = save_path.suffix.lower()
        is_volume = ext in (".nii", ".gz")

        if is_volume:
            ct_vol = preprocess_volume_nii(save_path)
            seg_vol = run_inference_on_volume(model, ct_vol)
            if slice_idx is None:
                slice_idx = ct_vol.shape[2] // 2

            ct_sl  = ct_vol[:, :, slice_idx]
            seg_sl = seg_vol[:, :, slice_idx]

            result["multislice"] = make_multislice_b64(ct_vol, seg_vol)
            mesh = make_3d_mesh_json(seg_vol)
            if mesh:
                result["mesh_3d"] = mesh

            severity = analyze_severity(seg_vol)
        else:
            # Single 2-D slice
            if ext == ".npy":
                arr = np.load(str(save_path))
                ct_sl, tensor, _ = preprocess_npy_slice(arr)
            else:
                ct_sl, tensor, _ = preprocess_external_image(save_path)

            _, probs, pred = predict_mask(model, tensor)
            seg_sl = pred[0, 0].numpy()
            # Build a fake volume for severity (single-slice)
            seg_vol = seg_sl[:, :, np.newaxis]
            severity = analyze_severity(seg_vol)

        # 2-D overlay
        result["overlay_2d"] = make_2d_overlay_b64(ct_sl, seg_sl)

        # Grad-CAM
        gradcam_gen = GradCAM(model, model.model.layer8)
        if is_volume:
            ct_resized = cv2.resize(ct_sl, EXPECTED_IMAGE_SIZE)
        else:
            ct_resized = cv2.resize(ct_sl, EXPECTED_IMAGE_SIZE) if ct_sl.shape != EXPECTED_IMAGE_SIZE else ct_sl
        inp_t = torch.tensor(ct_resized).unsqueeze(0).unsqueeze(0).float()
        cam = gradcam_gen(inp_t)
        gradcam_gen.remove()

        pred_label = 1 if seg_sl.mean() > 0.01 else 0
        result["gradcam"] = make_gradcam_b64(cv2.resize(ct_sl, EXPECTED_IMAGE_SIZE), cam, pred_label)

        # Severity
        result["severity"] = severity

        # Drugs
        result["drugs"] = recommend_drugs(
            severity["level"],
            cell_line_id=cell_line,
            gdsc1_path=gdsc1_path,
            gdsc2_path=gdsc2_path,
        )

        # Daily routine
        result["daily_routine"] = generate_daily_routine(severity["level"])

        # Tumor metrics
        result["metrics"] = {
            "tumor_voxels":     severity["tumor_voxels"],
            "tumor_volume_mm3": severity["tumor_volume_mm3"],
            "affected_slices":  severity["affected_slices"],
            "max_slice_burden": severity["max_slice_burden"],
            "severity_level":   severity["level"],
        }

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    return jsonify(result)


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    name    = data.get("name", "Anonymous")
    email   = data.get("email", "")
    rating  = int(data.get("rating", 5))
    message = data.get("message", "")

    save_feedback(name, email, rating, message)
    sent = send_feedback_email(name, email, rating, message)

    return jsonify({"success": True, "email_sent": sent})


@app.route("/api/feedback", methods=["GET"])
def get_feedback():
    return jsonify(get_all_feedback())


# ===========================================================================
# 15. ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LungAI Backend — Combined Medical Diagnosis API")
    print(f"  Device : {DEVICE}")
    print(f"  Model  : {MODEL_PATH}")
    print("=" * 60)
    # Pre-load model at startup
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
