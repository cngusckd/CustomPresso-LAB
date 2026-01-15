import streamlit as st
import torch
import cv2
import numpy as np
import json
import os
from PIL import Image
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_model, draw_boxes

# Standard PASCAL VOC Classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

st.set_page_config(page_title="CustomPresso-LAB YOLOv12n", layout="wide")

st.title("🚀 YOLOv12n CustomPresso-LAB Dashboard")
st.markdown("From-Scratch Implementation to Optimized Deployment (RTX 3070 Target)")

# Sidebar
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox("Select Model Variant", ["Baseline (FP32)", "Quantized (Int8)", "TensorRT (FP16/INT8)"])
conf_thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.text(f"Device: {device}")

# Tabs
tab1, tab2 = st.tabs(["🖼️ Inference Demo", "📊 Benchmarks & Stats"])

# --- TAB 1: INFERENCE ---
with tab1:
    st.header("Real-time Inference")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption="Original Image", width="stretch")
            
        if st.button("Run Detection"):
            with st.spinner("Running Inference..."):
                # Load Model (In real app, cache this)
                model = load_model(model_type, device)
                
                # Preprocess
                # Resize to 640x640 letterbox or simple resize for demo
                img_np = np.array(image)
                h0, w0 = img_np.shape[:2]
                img_resized = cv2.resize(img_np, (640, 640))
                
                # To Tensor
                img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
                
                # Inference
                # Ensure model returns predictions
                # Our YOLOv12n.forward returns self.detect(x) -> which returns self._predict(x) in eval mode!
                with torch.no_grad():
                     preds = model(img_t) 
                
                # Post-process (NMS is inside _predict now? Yes, I added _nms in Phase 2)
                # preds is list of [N, 6] tensors (one per batch item)
                det = preds[0]
                
                # Filter by conf
                det = det[det[:, 4] > conf_thres]
                
                # Scale boxes back to original size
                # x1, y1, x2, y2
                det[:, 0] *= w0 / 640
                det[:, 2] *= w0 / 640
                det[:, 1] *= h0 / 640
                det[:, 3] *= h0 / 640
                
                # Draw
                res_img = draw_boxes(image, det, VOC_CLASSES)
                
                with col2:
                    st.image(res_img, caption=f"Detections ({model_type})", width="stretch")
                    st.success(f"Found {len(det)} objects.")

# --- TAB 2: BENCHMARKS ---
with tab2:
    st.header("Comparative Benchmarking Results")
    
    # Load Results
    results_path = "benchmarks/results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
            
        # Display Metrics
        st.subheader("Performance Metrics")
        cols = st.columns(len(data))
        for i, (name, metrics) in enumerate(data.items()):
            with cols[i]:
                st.metric(label=name, value=f"{metrics['fps']:.2f} FPS", delta=f"{metrics['latency']:.2f} ms")
                st.text(f"Size: {metrics['size']:.2f} MB")
                st.text(f"VRAM: {metrics['vram']:.2f} MB")
                st.text(f"mAP: {metrics['map']:.2f}")
                
        # Display Plots
        st.subheader("Visualization Analysis")
        plot_cols = st.columns(2)
        
        plots_dir = "benchmarks/plots"
        plots = ["latency_comparison.png", "throughput_comparison.png", "vram_comparison.png", "pareto_chart.png"]
        
        for i, p_name in enumerate(plots):
            p_path = os.path.join(plots_dir, p_name)
            if os.path.exists(p_path):
                with plot_cols[i % 2]:
                    st.image(p_path, caption=p_name.replace("_", " ").title().replace(".Png", ""), width="stretch")
    else:
        st.warning("No benchmark results found. Run 'run_benchmarks.py' first.")
