# Object Momentum Classification Using Semantic Segmentation DNNs
- May 2025, ELEG5491 Project
- modernized DeepLabV3 architecture on COCO-Stuff dataset.

## How to Use
### 1. Requirements
```bash
pip install -r requirements.txt
```
### 2. Dataset Preparation
Place your dataset directly into the `./dataset/` directory. 
```text
dataset/
  ├── images/        # Put all raw training and validation images here
  ├── annotations/   # Put all pixel-level semantic mask maps here
```

### 3. Training
```bash
python src/train.py --data-root dataset --epochs 40 --batch-size 16 --lr 1e-4
```
**Arguments:**
- `--data-root`: Path to the root of your dataset folder.
- `--epochs`: Total number of training passes (default 40).
- `--batch-size`: Batch size per step (default 16).
- `--log-dir`: Directory for saving TensorBoard metrics (default `logs`).
- `--ckpt-dir`: Directory for saving model weights (default `checkpoints`).

### 4. Inference / Prediction
```bash
python src/infer.py --image test_image.jpg --weights checkpoints/best.pth --output result.png
```
### 5.Plotting
```bash
python plot/LossCurv.py
```
