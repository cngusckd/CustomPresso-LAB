# Edge-based Real-time Smart City Object Detection System

This repository contains a comprehensive system for training, optimizing, and deploying YOLOv11 models for Edge Computing environments (Smart Cities).

## Features

- **Model Architecture**: YOLOv11 (via `ultralytics`).
- **Data Management**: Automated dataset fetching (simulated) and train/test splitting.
- **Optimization Suite**:
  - Network Pruning (Structured/Unstructured)
  - Filter Decomposition (Low-rank approximation)
  - Post-Training Quantization (INT8/FP16)
- **Deployment**:
  - TensorRT Export
  - ExecuTorch Support (via TorchScript proxy)
- **Interactive UI**: Streamlit Dashboard for end-to-end management.

## Project Structure

```
├── app.py                  # Main Streamlit Dashboard
├── data/
│   ├── manager.py          # Dataset fetching & splitting
├── models/
│   ├── yolo_wrapper.py     # YOLOv11 Interface
├── optimization/
│   ├── pruning.py          # Pruning logic
│   ├── decomposition.py    # Decomposition logic
│   ├── quantization.py     # PTQ logic
├── deployment/
│   ├── converter.py        # TensorRT/ExecuTorch conversion
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- [Optional] CUDA-enabled GPU for training acceleration and TensorRT.

### Setting up the Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Install Dependencies
Once the virtual environment is active, install the required packages:
```bash
pip install -r requirements.txt
```

> **Note:** TensorRT and ExecuTorch require additional platform-specific installation steps.

### Automated Setup & Verification
**Windows:**
```bash
run_tests.bat
```

**Linux/macOS:**
```bash
bash run_tests.sh
```

## Usage

Run the Streamlit Dashboard:

```bash
streamlit run app.py
```

### Modules

1. **Data Management**: Fetch datasets (e.g., 'demo_city') and split them.
2. **Training**: Train YOLOv11 on the split dataset.
3. **Optimization**: Apply pruning, decomposition, or quantization to the trained model.
4. **Deployment**: Export the optimized model to TensorRT or ExecuTorch.
5. **Inference**: Upload images to test the model in real-time.

## Optimization Details

- **Pruning**: Uses `torch.nn.utils.prune` to remove weights.
- **Decomposition**: Estimates rank using SVD and approximates Conv2d layers.
- **Quantization**: Leverages Ultralytics export or PyTorch quantization for INT8/FP16.
