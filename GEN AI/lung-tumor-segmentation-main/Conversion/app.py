import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import time

from inference_utils import (
    load_trained_model,
    preprocess_image,
    predict_mask,
    load_external_manifest,
    TumorSeverityAnalyzer,
    DrugRecommendationEngine,
    create_treatment_protocol
)

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Lung Tumor Segmentation & Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    .report-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 10px;
    }
    .dark-mode .report-card {
        background-color: #1e1e1e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Global Settings
# ==============================================================================
BASE_DIR = Path(r"E:\aryan")
MODEL_PATH = BASE_DIR / "best_model.pth"
DEMO_IMAGES_DIR = BASE_DIR / "external_ct_demo"
MANIFEST_PATH = DEMO_IMAGES_DIR / "manifest.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# Core Functions (Cached)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return load_trained_model(MODEL_PATH, DEVICE)

@st.cache_data(show_spinner=False)
def get_manifest():
    return load_external_manifest(MANIFEST_PATH)

def create_overlay(image_2d, mask_2d):
    """Creates a red overlay of the mask on the original grayscale image."""
    img_uint8 = (image_2d * 255).astype(np.uint8)
    rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    
    red_mask = np.zeros_like(rgb)
    red_mask[:, :, 0] = (mask_2d * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(rgb, 1.0, red_mask, 0.4, 0)
    return overlay

def generate_report_ui(mask_2d, original_2d):
    """Runs analyzers and renders the report UI in Streamlit."""
    analyzer = TumorSeverityAnalyzer(voxel_spacing=(1.0, 1.0))
    severity_report = analyzer.analyze(mask_2d, original_2d)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Severity Score", f"{severity_report['severity_percentage']}%")
    col2.metric("Clinical Stage", severity_report['clinical_stage'])
    col3.metric("Risk Level", severity_report['risk_level'])
    tumor_cm3 = severity_report['tumor_volume_cm3']
    col4.metric("Est. Volume", f"{tumor_cm3:.2f} cm³" if tumor_cm3 > 0 else "0 cm³")
    
    if severity_report['severity_percentage'] > 0:
        st.markdown("### Clinical Protocol & Drug Recommendations")
        engine = DrugRecommendationEngine()
        recommendations = engine.recommend_drugs(severity_report)
        protocol = create_treatment_protocol(recommendations, severity_report)
        
        with st.expander("View Treatment Plan", expanded=True):
            if protocol['treatment_plan']:
                plan = protocol['treatment_plan'][0]
                st.write(f"**Line:** {plan['line']} | **Approach:** {plan['approach']}")
                if recommendations:
                    st.write(f"**Primary Top Drug:** {recommendations[0]['drug_name']} ({recommendations[0]['category'].title()})")
                st.write(f"**Duration:** {plan['duration']}")
            else:
                st.write("No active pharmacological treatment plan matched.")
            
            st.divider()
            
            if len(recommendations) > 0:
                st.write("**Top Alternative Drugs Examined**")
                for rec in recommendations[1:4]:
                    st.write(f"- {rec['drug_name']} (Efficacy: {rec['efficacy_score']*100:.0f}%) - {rec['rationale']}")
    else:
        st.success("No significant tumor voxels detected.")

def process_and_display(image_path, model, manifest, threshold=0.5):
    """Orchestrates processing for a single image layout."""
    try:
        # Preprocess
        slice_array, tensor = preprocess_image(image_path, manifest_dict=manifest)
        
        # Inference
        probs, pred_mask = predict_mask(model, tensor, DEVICE, threshold=threshold)
        pred_mask_np = pred_mask[0, 0].numpy()
        
        # Overlay
        overlay = create_overlay(slice_array, pred_mask_np)
        
        # Layout View
        c1, c2, c3 = st.columns(3)
        c1.image(slice_array, caption="Original Image", use_container_width=True, clamp=True)
        c2.image(pred_mask_np, caption=f"Predicted Mask (Thresh: {threshold:.2f})", use_container_width=True, clamp=True)
        c3.image(overlay, caption="Overlay Alignment", use_container_width=True)
        
        # Report
        generate_report_ui(pred_mask_np, slice_array)

    except Exception as e:
        st.error(f"Error processing image {image_path.name}: {str(e)}")

# ==============================================================================
# UI Structure
# ==============================================================================

st.title("🫁 Lung Tumor Segmentation System")
st.markdown("Deep learning-based segmentation framework with severity scoring and rule-based protocol prototyping.")

# Initialize Model
with st.spinner("Loading Neural Network Checkpoint..."):
    try:
        model_instance = load_model()
        app_manifest = get_manifest()
    except Exception as exc:
        st.error(f"Failed to load model. Is 'best_model.pth' at root {BASE_DIR}? \\n Error: {exc}")
        st.stop()

# Sidebar
st.sidebar.header("Configuration")
inference_threshold = st.sidebar.slider("Inference Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
st.sidebar.markdown("---")
st.sidebar.info("Upload CT slice images (.png) to see real-time inference and severity reporting.")

tab1, tab2 = st.tabs(["📊 Live Interactive Demo", "📤 Upload Custom Scan"])

# -----------------
# TAB 1: DEMO
# -----------------
with tab1:
    st.markdown("### Pre-Processed Clinical Examples")
    st.write("Browse through representative examples of the model's performance on external validation sets.")
    
    # Try to grab 3 good samples
    sample_names = ["val_subject_61_slice_131.png", "val_subject_58_slice_290.png", "val_subject_60_slice_20.png"]
    valid_samples = [DEMO_IMAGES_DIR / nm for nm in sample_names if (DEMO_IMAGES_DIR / nm).exists()]
    
    if not valid_samples:
        valid_samples = list(DEMO_IMAGES_DIR.glob("*.png"))[:3]
        
    if not valid_samples:
        st.warning("No demo images found. Please ensure `external_ct_demo/` has PNG files.")
    else:
        # Create tabs inside the demo for each sample
        demo_tabs = st.tabs([f"Case {i+1}: {p.name}" for i, p in enumerate(valid_samples)])
        for idx, d_tab in enumerate(demo_tabs):
            with d_tab:
                process_and_display(valid_samples[idx], model_instance, app_manifest, inference_threshold)


# -----------------
# TAB 2: UPLOAD
# -----------------
with tab2:
    st.markdown("### Process Your Own Imagery")
    uploaded_files = st.file_uploader("Upload Medical CT Slice (PNG/JPG/JPEG)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if uploaded_files:
        for idx, up_file in enumerate(uploaded_files):
            st.markdown(f"#### Processing Upload: `{up_file.name}`")
            # Save uploaded file temporarily to process
            temp_path = BASE_DIR / f"temp_upload_{idx}.png"
            with open(temp_path, "wb") as f:
                f.write(up_file.getbuffer())
            
            with st.spinner(f"Analyzing {up_file.name}..."):
                process_and_display(temp_path, model_instance, app_manifest, inference_threshold)
            
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
            st.divider()
