# Mamba-YOLO Evaluation & Object Tracking Pipeline

This repository contains the source code for training, evaluating, and running inference with **Mamba-YOLO** combined with the **ByteTrack** association engine for multi-object tracking (MOT). The pipeline is specifically tuned for challenging scenarios, such as tracking small objects from aerial/drone imagery (e.g., the AFO dataset).

---

## 🛠️ Installation & Setup

To run the detector and tracking pipeline, you need to create a Python virtual environment. Python 3.12 is recommended, but Python 3.10+ is also supported.

### 1. Environment Activation
Create and activate your `.venv`:
```bash
python -m venv .venv
source .venv/bin/activate
```
### 2. Install PyTorch and Dependencies
Install the appropriate PyTorch version matching your CUDA setup, followed by core computer vision packages:
```bash
pip3 install torch torchvision torchaudio
pip install seaborn thop timm einops loguru tqdm opencv-python
```

### 3. Compile the Core SSM Module
Compile the native selective_scan hardware-forwarding module required for State Space Models (SSM) inside the Mamba architecture:
```bash
cd Mamba-YOLO/selective_scan && pip install . && cd ..
```

### 4. Editable Package Installation
Install the project in editable mode to link all internal dependencies:
```bash
pip install -v -e .
```

## Usage Guide (Two-Stage Tracking Pipeline)
The tracking workflow is split into a two-stage pipeline (Tracking-by-Detection). This decouples heavy deep learning inference from state estimation, optimizing VRAM consumption and allowing independent hyperparameter tuning.

### Stage 1: Generate Object Detections
Run the trained Mamba-YOLO model on your sequence images. This script utilizes the GPU to perform inference and freezes the raw detections into structured .txt files:
```bash
python gen_tracking_afo_safe.py
```


