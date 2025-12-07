# How to Run All Files - Responsible AI Segmentation Project

This guide provides comprehensive instructions for running all scripts in this responsible AI segmentation project, including training, evaluation, and various explainability visualizations.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Setup](#dataset-setup)
3. [Installation](#installation)
4. [Training the Model](#training-the-model)
5. [Visualization & Explainability](#visualization--explainability)
   - [Basic Grad-CAM Visualization](#1-basic-grad-cam-visualization)
   - [Uncertainty Quantification](#2-uncertainty-quantification)
   - [Score-CAM Explainability](#3-score-cam-explainability)
   - [Occlusion Sensitivity](#4-occlusion-sensitivity)
6. [Understanding Outputs](#understanding-outputs)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.7 or higher
- **GPU**: CUDA-capable GPU recommended (CPU will work but much slower)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Disk Space**: ~5GB for dataset and outputs

### Required Python Packages

```bash
pip install torch torchvision
pip install segmentation-models-pytorch>=0.3.4
pip install torchmetrics==1.4.0
pip install albumentations
pip install opencv-python
pip install pandas
pip install matplotlib
pip install Pillow
pip install tqdm
```

Or install all at once:

```bash
pip install torch torchvision segmentation-models-pytorch>=0.3.4 torchmetrics==1.4.0 \
    albumentations opencv-python pandas matplotlib Pillow tqdm
```

---

## Dataset Setup

### Expected Directory Structure

```
responsible_ai/
├── phenocyte_seg/
│   ├── images/              # Input images
│   ├── masks/               # Ground truth masks
│   └── dataset_split.csv    # Train/val/test split
├── visualize_pretrained_model.py
├── visualize_uncertainty.py
├── visualize_scorecam.py
├── visualize_occlusion.py
├── no_bg_inter_all_classes_4_images.py
└── HOW_TO_RUN.md (this file)
```

### Dataset CSV Format

The `dataset_split.csv` should have columns:
- `img_name`: Image filename
- `set`: Split name (`train`, `val`, or `test`)

Example:
```csv
img_name,set
image_001.png,train
image_002.png,val
image_003.png,test
```

### Class Labels

The project segments 5 classes:
1. **background**: Background pixels
2. **root**: Plant root
3. **hypocotyl**: Hypocotyl region
4. **cotyledon**: Cotyledon (seed leaves)
5. **seed**: Seed region

---

## Installation

1. **Clone or download the repository:**
```bash
cd /path/to/responsible_ai
```

2. **Install dependencies:**
```bash
pip install torch torchvision segmentation-models-pytorch>=0.3.4 torchmetrics==1.4.0 \
    albumentations opencv-python pandas matplotlib Pillow tqdm
```

3. **Verify dataset setup:**
```bash
# Check if dataset exists
ls phenocyte_seg/

# Should show: images/ masks/ dataset_split.csv
```

---

## Training the Model

### Script: `no_bg_inter_all_classes_4_images.py`

This script trains a semantic segmentation model with:
- **Architecture**: FPN (Feature Pyramid Network)
- **Encoder**: SE-ResNeXt50
- **Loss**: Combined Dice + Focal Loss with class weighting
- **Features**: Per-class metrics, TP/FP/FN visualization, Grad-CAM

### Basic Training

```bash
python no_bg_inter_all_classes_4_images.py
```

### What Happens During Training

1. **Computes class weights** from training data (handles class imbalance)
2. **Trains for 50 epochs** with learning rate decay at epoch 25
3. **Validates after each epoch** with per-class IoU metrics
4. **Saves best model** as `best_model_FPN.pth`
5. **Generates visualizations**:
   - TP/FP/FN error maps for 5 test samples
   - Grad-CAM explainability for 3 test samples
6. **Saves training curves** as `training_curves.png`
7. **Saves metrics report** as `fpn-se_resnext50_training_result_improved.txt`

### Training Outputs

```
responsible_ai/
├── best_model_FPN.pth                        # Best trained model
├── training_curves.png                       # Training/validation curves
├── fpn-se_resnext50_training_result_improved.txt  # Metrics report
├── tpfpfn_maps_all_class_4_img/              # TP/FP/FN visualizations
│   ├── sample0_cls_background.png
│   ├── sample0_cls_root.png
│   ├── sample0_cls_hypocotyl.png
│   ├── sample0_cls_cotyledon.png
│   └── sample0_cls_seed.png
└── gradcam_visualizations/                   # Grad-CAM heatmaps
    └── sample_0/
        ├── gradcam_background.png
        ├── gradcam_root.png
        └── ...
```

### Customizing Training

Edit the script to modify:

```python
# Line 746-750: Modify hyperparameters
BATCH_SIZE = 4          # Increase for faster training (if GPU memory allows)
NUM_EPOCHS = 50         # Number of training epochs
LEARNING_RATE = 0.0001  # Learning rate
IMG_SIZE = (512, 256)   # Input image size

# Line 743: Select classes to train on
SELECTED_CLASSES = ["background", "root", "hypocotyl", "cotyledon", "seed"]
# Or exclude background:
# SELECTED_CLASSES = ["root", "hypocotyl", "cotyledon", "seed"]
```

### Expected Training Time

- **GPU (NVIDIA RTX 3090)**: ~2-3 hours for 50 epochs
- **GPU (NVIDIA GTX 1080)**: ~4-5 hours for 50 epochs
- **CPU**: Not recommended (20+ hours)

---

## Visualization & Explainability

After training, you can generate various explainability visualizations using your trained model.

### 1. Basic Grad-CAM Visualization

**Script**: `visualize_pretrained_model.py`

Generates TP/FP/FN error maps and Grad-CAM explainability heatmaps.

#### Quick Start

```bash
python visualize_pretrained_model.py --model_path best_model_FPN.pth
```

#### Full Options

```bash
python visualize_pretrained_model.py \
    --model_path best_model_FPN.pth \
    --num_samples 10 \
    --split test \
    --output_dir pretrained_visualizations \
    --encoder se_resnext50_32x4d \
    --img_size 512 256
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | **Required** | Path to trained model (.pth) |
| `--num_samples` | `5` | Number of samples to visualize |
| `--split` | `test` | Dataset split (`train`, `val`, `test`) |
| `--output_dir` | `pretrained_visualizations` | Output directory |
| `--encoder` | `se_resnext50_32x4d` | Encoder architecture |
| `--img_size` | `512 256` | Image size (width height) |

#### Output

```
pretrained_visualizations/
├── tpfpfn_maps/       # TP/FP/FN error visualizations
└── gradcam_maps/      # Grad-CAM explainability heatmaps
```

#### When to Use

- **Quick model diagnosis**: See where model makes errors
- **Fast explainability**: Understand what model focuses on
- **General exploration**: Initial model analysis

---

### 2. Uncertainty Quantification

**Script**: `visualize_uncertainty.py`

Uses Monte Carlo Dropout to estimate prediction uncertainty.

#### Quick Start

```bash
python visualize_uncertainty.py --model_path best_model_FPN.pth
```

#### Full Options

```bash
python visualize_uncertainty.py \
    --model_path best_model_FPN.pth \
    --num_samples 5 \
    --num_mc_samples 20 \
    --split test \
    --output_dir uncertainty_visualizations
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | **Required** | Path to trained model (.pth) |
| `--num_samples` | `5` | Number of images to process |
| `--num_mc_samples` | `20` | Number of MC Dropout passes |
| `--split` | `test` | Dataset split |
| `--output_dir` | `uncertainty_visualizations` | Output directory |

#### Output

```
uncertainty_visualizations/
├── tpfpfn_maps/       # Error analysis
└── uncertainty_maps/  # Uncertainty visualizations
    └── sample0_uncertainty.png  # 6-panel visualization:
        # 1. Original image
        # 2. Predicted classes
        # 3. Confidence map
        # 4. Variance uncertainty
        # 5. Entropy uncertainty
        # 6. Variance overlay
```

#### Interpreting Uncertainty

- **Blue regions**: Low uncertainty (model is confident)
- **Red regions**: High uncertainty (model is guessing)
- **Use case**: Identify ambiguous regions where predictions are unreliable

#### When to Use

- **Medical/safety-critical applications**: Know when model is uncertain
- **Active learning**: Find samples to label next
- **Quality control**: Flag uncertain predictions for human review
- **Model debugging**: Find regions where model struggles

#### Expected Runtime

- **20 MC samples**: ~30-60 seconds per image
- **Increase `--num_mc_samples`** for better uncertainty estimates (slower)
- **Decrease for faster processing** (less accurate)

---

### 3. Score-CAM Explainability

**Script**: `visualize_scorecam.py`

Gradient-free explainability method with cleaner, more stable visualizations than Grad-CAM.

#### Quick Start

```bash
python visualize_scorecam.py --model_path best_model_FPN.pth
```

#### Full Options

```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --num_samples 5 \
    --split test \
    --output_dir scorecam_visualizations \
    --batch_size 32 \
    --max_activation_maps 100
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | **Required** | Path to trained model (.pth) |
| `--num_samples` | `5` | Number of samples to visualize |
| `--batch_size` | `32` | Processing batch size |
| `--max_activation_maps` | `None` | Limit activation maps (for speed) |
| `--split` | `test` | Dataset split |
| `--output_dir` | `scorecam_visualizations` | Output directory |

#### Output

```
scorecam_visualizations/
├── tpfpfn_maps/    # Error analysis
└── scorecam_maps/  # Score-CAM heatmaps (3-panel)
    └── sample0_scorecam_root.png
        # 1. Original image
        # 2. Score-CAM heatmap
        # 3. Overlay
```

#### Score-CAM vs Grad-CAM

| Feature | Grad-CAM | Score-CAM |
|---------|----------|-----------|
| **Speed** | Fast (~3s/sample) | Slower (~10-15s/sample) |
| **Uses Gradients** | Yes | No |
| **Stability** | Can be noisy | More stable |
| **Quality** | Good | Better |
| **Small Objects** | May miss | Better localization |

#### When to Use

- **Publication-quality figures**: Cleaner visualizations
- **Small object detection**: Better than Grad-CAM
- **Clinical applications**: More reliable explanations
- **Final analysis**: After exploring with Grad-CAM

#### Speed Optimization

For faster processing (with minimal quality loss):

```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --max_activation_maps 100  # ~5x faster
```

---

### 4. Occlusion Sensitivity

**Script**: `visualize_occlusion.py`

Most interpretable explainability method - directly measures which regions matter by masking them.

#### Quick Start

```bash
python visualize_occlusion.py --model_path best_model_FPN.pth
```

#### Full Options

```bash
python visualize_occlusion.py \
    --model_path best_model_FPN.pth \
    --num_samples 5 \
    --patch_size 32 \
    --stride 16 \
    --occlusion_value mean \
    --split test
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | **Required** | Path to trained model (.pth) |
| `--num_samples` | `5` | Number of samples to process |
| `--patch_size` | `32` | Size of occlusion patch (pixels) |
| `--stride` | `patch_size // 2` | Sliding window stride |
| `--occlusion_value` | `mean` | Fill value (`mean` or `zero`) |
| `--split` | `test` | Dataset split |
| `--output_dir` | `occlusion_visualizations` | Output directory |

#### Output

```
occlusion_visualizations/
├── tpfpfn_maps/      # Error analysis
└── occlusion_maps/   # Occlusion sensitivity heatmaps
    └── sample0_occlusion_root.png
        # 1. Original image
        # 2. Sensitivity map
        # 3. Overlay
```

#### How It Works

1. Divide image into patches (e.g., 32×32)
2. For each patch:
   - Mask it (replace with mean or black)
   - Run forward pass
   - Measure prediction drop
3. High drop = important region

#### Interpreting Results

- **Red/yellow regions**: Important for prediction (removing them hurts)
- **Blue regions**: Not important (can be occluded without effect)
- **Direct causation**: Unlike Grad-CAM, this measures actual impact

#### When to Use

- **Most interpretable**: Direct cause-and-effect
- **No gradient issues**: Works when Grad-CAM fails
- **Clinical/legal applications**: Explainability you can trust
- **Model debugging**: See exactly what model uses

#### Performance Tuning

**Faster (lower quality):**
```bash
python visualize_occlusion.py \
    --model_path best_model_FPN.pth \
    --patch_size 64 \      # Larger patches = faster
    --stride 32            # Larger stride = faster
```

**Better quality (slower):**
```bash
python visualize_occlusion.py \
    --model_path best_model_FPN.pth \
    --patch_size 16 \      # Smaller patches = slower but finer detail
    --stride 8             # Smaller stride = slower but better coverage
```

#### Expected Runtime

- **Patch 32, stride 16**: ~2-5 minutes per image
- **Patch 16, stride 8**: ~8-15 minutes per image
- **Patch 64, stride 32**: ~30-60 seconds per image

---

## Understanding Outputs

### TP/FP/FN Maps (Error Analysis)

4-panel visualization showing:

1. **Original Image**: Input image
2. **Ground Truth**: Correct segmentation mask
3. **Prediction**: Model's prediction
4. **TP/FP/FN Overlay**:
   - **Green**: True Positives (correct)
   - **Red**: False Positives (over-prediction)
   - **Blue**: False Negatives (missed regions)

**Good model**: Mostly green, minimal red/blue
**Bad model**: Lots of red (hallucination) or blue (missing detections)

### Grad-CAM / Score-CAM Heatmaps

3-panel visualization:

1. **Original Image**
2. **Heatmap**: Warm colors (red/yellow) = important regions
3. **Overlay**: Heatmap on image

**Good attention**: Heatmap aligns with actual object locations
**Bad attention**: Scattered or wrong regions highlighted

### Uncertainty Maps

6-panel visualization:

1. **Original Image**
2. **Predicted Classes**: Color-coded predictions
3. **Confidence Map**: Green = high confidence, red = low
4. **Variance Uncertainty**: Variance across MC samples
5. **Entropy Uncertainty**: Prediction entropy
6. **Variance Overlay**: Uncertainty on image

**Interpretation**:
- Use predictions with **low uncertainty** (blue regions)
- Be cautious of **high uncertainty** (red regions)
- High uncertainty often at class boundaries

### Occlusion Sensitivity Maps

3-panel visualization:

1. **Original Image**
2. **Sensitivity Map**: Red = important, blue = not important
3. **Overlay**: Sensitivity on image

**Interpretation**:
- **Red/yellow**: Removing these regions hurts prediction
- **Blue**: Can be occluded without affecting prediction
- Shows **exactly** what model uses for decisions

---

## Comparison Table: Which Method to Use?

| Use Case | Recommended Method |
|----------|-------------------|
| **Quick exploration** | Grad-CAM (`visualize_pretrained_model.py`) |
| **Publication figures** | Score-CAM (`visualize_scorecam.py`) |
| **Uncertainty estimation** | MC Dropout (`visualize_uncertainty.py`) |
| **Most interpretable** | Occlusion (`visualize_occlusion.py`) |
| **Small objects** | Score-CAM or Occlusion |
| **Fast iteration** | Grad-CAM |
| **Clinical/safety-critical** | All methods for comprehensive analysis |
| **Model debugging** | TP/FP/FN maps + Occlusion |

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

For **training** (`no_bg_inter_all_classes_4_images.py`):
```python
# Edit line 746
BATCH_SIZE = 2  # Reduce from 4 to 2
```

For **visualizations**:
```bash
# Process fewer samples
python visualize_scorecam.py --model_path best_model_FPN.pth --num_samples 1

# For Score-CAM specifically:
python visualize_scorecam.py --model_path best_model_FPN.pth --batch_size 8

# For MC Dropout:
python visualize_uncertainty.py --model_path best_model_FPN.pth --num_mc_samples 10
```

#### 2. Model File Not Found

**Error:**
```
FileNotFoundError: best_model_FPN.pth
```

**Solution:**
```bash
# Check if model exists
ls -la best_model_FPN.pth

# Or use absolute path
python visualize_pretrained_model.py --model_path /full/path/to/best_model_FPN.pth
```

#### 3. Dataset Not Found

**Error:**
```
FileNotFoundError: phenocyte_seg/dataset_split.csv
```

**Solution:**
```bash
# Check dataset structure
ls phenocyte_seg/

# Should have: images/ masks/ dataset_split.csv
```

#### 4. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'segmentation_models_pytorch'
```

**Solution:**
```bash
pip install segmentation-models-pytorch>=0.3.4
```

#### 5. Slow Processing

**For Occlusion Sensitivity:**
```bash
# Use larger patches and stride
python visualize_occlusion.py \
    --model_path best_model_FPN.pth \
    --patch_size 64 \
    --stride 32
```

**For MC Dropout:**
```bash
# Reduce MC samples
python visualize_uncertainty.py \
    --model_path best_model_FPN.pth \
    --num_mc_samples 10  # Instead of 20
```

**For Score-CAM:**
```bash
# Limit activation maps
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --max_activation_maps 50
```

#### 6. No Dropout Warning (MC Dropout)

**Warning:**
```
⚠️ WARNING: Model has no dropout layers!
```

**Explanation:**
MC Dropout requires dropout layers in the model. If you see this warning, the uncertainty estimates may not be meaningful. The script will still run but results should be interpreted carefully.

---

## Quick Reference: All Commands

### Training
```bash
python no_bg_inter_all_classes_4_images.py
```

### Visualizations
```bash
# Grad-CAM (Fast exploration)
python visualize_pretrained_model.py --model_path best_model_FPN.pth

# Uncertainty quantification
python visualize_uncertainty.py --model_path best_model_FPN.pth --num_mc_samples 20

# Score-CAM (Publication quality)
python visualize_scorecam.py --model_path best_model_FPN.pth

# Occlusion sensitivity (Most interpretable)
python visualize_occlusion.py --model_path best_model_FPN.pth --patch_size 32
```

---

## Expected Workflow

### 1. Initial Setup
```bash
# Install dependencies
pip install torch torchvision segmentation-models-pytorch>=0.3.4 torchmetrics==1.4.0 \
    albumentations opencv-python pandas matplotlib Pillow tqdm

# Verify dataset
ls phenocyte_seg/
```

### 2. Train Model
```bash
python no_bg_inter_all_classes_4_images.py
# Wait ~2-3 hours on GPU
# Output: best_model_FPN.pth
```

### 3. Quick Exploration
```bash
# Fast Grad-CAM visualization
python visualize_pretrained_model.py --model_path best_model_FPN.pth --num_samples 10
```

### 4. Comprehensive Analysis
```bash
# Uncertainty (where model is uncertain)
python visualize_uncertainty.py --model_path best_model_FPN.pth --num_samples 5

# Score-CAM (publication quality)
python visualize_scorecam.py --model_path best_model_FPN.pth --num_samples 5

# Occlusion (most interpretable)
python visualize_occlusion.py --model_path best_model_FPN.pth --num_samples 3 --patch_size 32
```

---

## Performance Summary

| Script | Time per Sample | Use Case |
|--------|----------------|----------|
| `visualize_pretrained_model.py` | ~3s | Quick exploration |
| `visualize_uncertainty.py` | ~30-60s | Uncertainty estimation |
| `visualize_scorecam.py` | ~10-15s | Publication figures |
| `visualize_occlusion.py` | ~2-5 min | Deep interpretability |

---

## Additional Resources

- **Grad-CAM Paper**: [Selvaraju et al., 2016](https://arxiv.org/abs/1610.02391)
- **Score-CAM Paper**: [Wang et al., 2020](https://arxiv.org/abs/1910.01279)
- **MC Dropout Paper**: [Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)
- **Segmentation Models**: [SMP Documentation](https://segmentation-models-pytorch.readthedocs.io/)

---

## Support

For issues or questions:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Verify dataset structure matches expected format
3. Ensure all dependencies are installed
4. Check GPU memory if using CUDA

---

**Happy Training and Visualizing! 🚀**
