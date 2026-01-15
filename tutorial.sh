#!/bin/bash

# =================================================================================
# CustomPresso-LAB: YOLOv12n From-Scratch to Optimized Deployment
# Tutorial & Setup Script
# =================================================================================

echo "🚀 Starting Setup for Custom YOLOv12n Framework..."

# 1. Environment Setup
echo "----------------------------------------------------------------"
echo "[Step 1] Installing Dependencies..."
echo "----------------------------------------------------------------"
# Check for pip/python3
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install Python."
    exit 1
fi

pip install torch torchvision numpy opencv-python tqdm matplotlib streamlit requests
# Optional for TRT: pip install tensorrt
echo "✅ Dependencies Installed."

# 2. Data Preparation
echo "----------------------------------------------------------------"
echo "[Step 2] Preparing PASCAL VOC Data..."
echo "----------------------------------------------------------------"
echo "Running data/download_voc.py (Manual Download & Cache)..."
python3 data/download_voc.py --path ./data
if [ ! -f ./data/voc_train_list.pt ]; then
    echo "❌ Data preparation failed. voc_train_list.pt not found."
    exit 1
fi
echo "✅ Data Preparation Complete."

# 3. Training
echo "----------------------------------------------------------------"
echo "[Step 3] Training YOLOv12n..."
echo "----------------------------------------------------------------"
echo "Running train.py (Dry-run or Full Training)..."
# To run full training, remove 'dry-run' comments or modify epochs in train.py
# usage: python3 train.py
# python3 train.py
echo "ℹ️  Run 'python3 train.py' to start training."
# Check if train.py runs (dry run)
# python3 train.py --epochs 1 --dry-run # Need to implement dry-run flag or just inform user

# 4. Optimization & Compression
echo "----------------------------------------------------------------"
echo "[Step 4] Optimizing Model (Pruning, Quantization, Export)..."
echo "----------------------------------------------------------------"
echo "ℹ️  Scripts located in optimization/:"
echo "   - python3 optimization/pruning.py"
echo "   - python3 optimization/quantization.py"
echo "   - python3 optimization/export_trt.py"

# 5. Benchmarking
echo "----------------------------------------------------------------"
echo "[Step 5] Benchmarking (Latency, VRAM, Throughput)..."
echo "----------------------------------------------------------------"
echo "Running run_benchmarks.py..."
python3 run_benchmarks.py
echo "ℹ️  Results saved to benchmarks/results.json"

# 6. Web Dashboard
echo "----------------------------------------------------------------"
echo "[Step 6] Launching Web Dashboard..."
echo "----------------------------------------------------------------"
echo "Run the following command to start the demo app:"
echo "streamlit run app/dashboard.py"

echo "================================================================"
echo "🎉 Setup & Guide Complete! Happy Coding!"
echo "================================================================"
