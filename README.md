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

# Running Individual Modules

## 0. All requirements - summary

Download the AFO dataset using the script `utils/afo_download.py`.
Download the model weights intended for detection: `yolox_x.pth` (or `yolox_m.pth`) (from https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0) or the pre-trained Mamba-YOLO weights (the `best.pt` file) and save them, for example, in the `weights/` folder.
Download the VOT model weights (the `rl_training_best_afo_1.pt` file).

Clone the following repositories into the root directory of the project:
 - https://github.com/JackWoo0831/Mamba_Trackers
 - https://github.com/FoundationVision/ByteTrack
 - https://github.com/HZAI-ZJNU/Mamba-YOLO

And install the tools included in the cloned repositories according to the instructions in their respective README files.

Install the requirements from `requirements.txt`.

## 1. Run Detection

Generate detection results using one of the models.

For yolox_x:
```bash
python tools/gen_det_afo.py \
    --split test \
    --data_root ./data/afo \
    --exp_file exps/yolox_x_afo.py \
    --model_path weights/yolox_x.pth \
    --save_dir out/det_results_yolox_x/afo/{split} \
    --generate_meta_data \
    --vis \
    --native_size  # Keep original image size
```

For yolox_m:
```bash
python tools/gen_det_afo.py \
    --split test \
    --data_root ./data/afo \
    --exp_file exps/yolox_m_afo.py \
    --model_path weights/yolox_m.pth \
    --save_dir out/det_results_yolox_m/afo/{split} \
    --generate_meta_data \
    --vis \
    --native_size  # Keep original image size
```

For mamba-yolo:
```bash
python tools/gen_det_afo_mamba.py \
    --split test \
    --data_root ./data/afo \
    --model_path ./weights/best_two.pt \
    --save_dir ./out/det_results_mamba/afo/{split} \
    --generate_meta_data \
    --vis \
    --imgsz 960 \
    --conf 0.05
```

The output consists of .txt files containing detections for each sequence, located under the path specified by `--save_dir`.

Detection results can be evaluated using the script `models/VOT/tracking/refine_afo_tracking.py`, after appropriately adjusting the paths to the detection output files (hardcoded constants in the script).

## 2. Run Tracking

Next, based on the detection results—regardless of how they were obtained—run the tracking algorithm.

Tracking using the ByteTrack-based algorithm for detection results from yolo_m:
```bash
python -m tools.track_kalman \
    --det_path out/det_results_yolox_m/afo/test \
    --motion byte \
    --data_root data/afo/{split}/img/{seq}_{frame_id:03d}.jpg \
    --save_dir out/track_res/{dataset_name}/{split} \
    --vis
```
Tracking using the ByteTrack-based algorithm for detection results from mamba-yolo:
```bash
python -m tools.track_kalman \
    --det_path out/det_results_mamba/afo/test \
    --motion byte \
    --data_root data/afo/{split}/img/{seq}_{frame_id:03d}.jpg \
    --save_dir out/track_res_mamba/{dataset_name}/{split} \
    --vis
```

The output consists of .txt files with tracking results for each sequence, located under the path specified by `--save_dir`.

### Output Format

The `tools/track_kalman.py` script creates .txt files (one for each video sequence) matching the MOTChallenge format—this format was used by the creators of the Mamba_Trackers project. Each line represents a single object in a single frame in the following layout:

`frame_id, target_id, x, y, width, height, score, -1, -1, -1`

Each line of the .txt file represents a single object detection in a given frame, saved in the MOTChallenge format:
- frame_id: Frame number (1-indexed).
- target_id: A unique identifier assigned to a specific object (remains constant across subsequent frames).
- x_min, y_min: Coordinates of the top-left corner of the bounding box.
- width, height: Width and height of the bounding box in pixels.
- score: The model's confidence score for this object.
- -1, -1, -1: Default values (placeholders) required by the MOT format specification.

Tracking results using Kalman filters are saved in `track_res/afo/test/afo_byte/`, and after refinement by VOT, in `track_res/afo/test/afo_byte_refined/`.

Evaluation results are in `out/eval_results/`—I have placed the detection metrics and the outcome of the VOT tracking refinement there; this is where the IoU values before and after VOT can be found.

## 3. Refine Tracking Results Using VOT

Run the script `models/VOT/tracking/refine_afo_tracking.py`—the paths to the input and output data need to be configured as constants within the script's code.

The output consists of .txt files with tracking results for each sequence, using an identical format to the output files from the previous module (Kalman filter tracking).

## 4. Results - Visualization

Detection Visualization - If the `--vis` option was selected during detection generation, the images with bounding boxes can be found in the results under the `vis_results/` subfolder.

Ground Truth Visualization for Tracking - Generating images with bounding boxes:
```bash
python tools/track_gt_vis_afo.py --data_root data/afo --save_dir out/gt_vis --split test
```

Tracking Results Visualization after VOT refinement - Generating images:
```bash
python tools/track_res_vis_afo.py \
  --track_dir out/track_res/afo/test/afo_byte_refined/ \
  --img_root data/afo/test/ \
  --save_dir out/refined_track_vis/
```

### Displaying Results

If all images from the previous step have been generated, you can display the results for a selected sequence using the following script:

```bash
python3 -m tools.display_all_vis
```

The specific sequence identifier and paths to the saved images with results can be configured inside the code of the `tools/display_all_vis.py` script.
```

