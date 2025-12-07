# Pretrained Model Visualization Guide

This guide explains how to generate **TP/FP/FN maps** and **Grad-CAM visualizations** from your pretrained segmentation model.

---

## 🎯 What You'll Get

### 1. **TP/FP/FN Error Maps**
4-panel visualizations showing:
- Original image
- Ground truth mask
- Predicted mask
- Color-coded overlay:
  - **Green**: True Positives (correct predictions)
  - **Red**: False Positives (model over-predicted)
  - **Blue**: False Negatives (model missed)

### 2. **Grad-CAM Explainability Maps**
3-panel visualizations showing:
- Original image
- Grad-CAM heatmap (warm colors = important regions)
- Heatmap overlay on image

Both visualizations are generated **per class** (background, root, hypocotyl, cotyledon, seed).

---

## 🚀 Quick Start

### **Basic Usage**

```bash
python visualize_pretrained_model.py --model_path best_model_FPN.pth
```

This will:
- Load your pretrained model from `best_model_FPN.pth`
- Process **5 samples** from the **test set**
- Generate visualizations in `pretrained_visualizations/`

---

## 📋 Command Line Options

### **Required Arguments**

| Argument | Description | Example |
|----------|-------------|---------|
| `--model_path` | Path to your pretrained `.pth` file | `best_model_FPN.pth` |

### **Optional Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_samples` | `5` | Number of samples to visualize |
| `--split` | `test` | Dataset split: `train`, `val`, or `test` |
| `--output_dir` | `pretrained_visualizations` | Output directory |
| `--encoder` | `se_resnext50_32x4d` | Encoder architecture |
| `--img_size` | `512 256` | Image size (width height) |

---

## 📝 Usage Examples

### **Example 1: Visualize 10 test samples**

```bash
python visualize_pretrained_model.py \
    --model_path best_model_FPN.pth \
    --num_samples 10
```

### **Example 2: Use validation set**

```bash
python visualize_pretrained_model.py \
    --model_path best_model_FPN.pth \
    --split val \
    --num_samples 3
```

### **Example 3: Custom output directory**

```bash
python visualize_pretrained_model.py \
    --model_path best_model_FPN.pth \
    --num_samples 5 \
    --output_dir my_visualizations
```

### **Example 4: Different image size**

```bash
python visualize_pretrained_model.py \
    --model_path best_model_FPN.pth \
    --img_size 640 320
```

### **Example 5: Different encoder (if you trained with different architecture)**

```bash
python visualize_pretrained_model.py \
    --model_path my_unet_model.pth \
    --encoder resnet50 \
    --num_samples 5
```

---

## 📁 Output Structure

After running the script, you'll find:

```
pretrained_visualizations/
├── tpfpfn_maps/
│   ├── sample0_tpfpfn_background.png
│   ├── sample0_tpfpfn_root.png
│   ├── sample0_tpfpfn_hypocotyl.png
│   ├── sample0_tpfpfn_cotyledon.png
│   ├── sample0_tpfpfn_seed.png
│   ├── sample1_tpfpfn_background.png
│   └── ... (5 images per sample)
│
└── gradcam_maps/
    ├── sample0_gradcam_background.png
    ├── sample0_gradcam_root.png
    ├── sample0_gradcam_hypocotyl.png
    ├── sample0_gradcam_cotyledon.png
    ├── sample0_gradcam_seed.png
    ├── sample1_gradcam_background.png
    └── ... (5 images per sample)
```

**Total files per sample:** 10 images (5 TP/FP/FN + 5 Grad-CAM)

**For 5 samples:** 50 visualization images

---

## 🔍 Interpreting Results

### **TP/FP/FN Maps**

**Good predictions:**
- Mostly **green** (TP) in the class regions
- Very little **red** (FP) or **blue** (FN)

**Bad predictions:**
- **Red scattered everywhere** = Model hallucinates this class
- **Blue in class regions** = Model misses the class
- Mix of red/blue = Poor segmentation boundary

### **Grad-CAM Maps**

**Good attention:**
- **Red/yellow** (hot) regions align with actual class locations
- For "root": heatmap focuses on root regions
- For "cotyledon": heatmap focuses on leaf regions

**Bad attention:**
- **Scattered** heatmap = Model is confused
- **Wrong regions** highlighted = Model uses wrong features
- **Uniform blue** = Model ignores this class

---

## 🛠️ Troubleshooting

### **Error: File not found**

```bash
FileNotFoundError: best_model_FPN.pth
```

**Solution:** Make sure the model path is correct.

```bash
# Check if file exists
ls -la best_model_FPN.pth

# Or use absolute path
python visualize_pretrained_model.py --model_path /full/path/to/best_model_FPN.pth
```

### **Error: CUDA out of memory**

```bash
RuntimeError: CUDA out of memory
```

**Solution:** Process fewer samples at a time:

```bash
python visualize_pretrained_model.py \
    --model_path best_model_FPN.pth \
    --num_samples 1
```

### **Error: Wrong encoder**

```bash
KeyError: 'encoder'
```

**Solution:** Specify the correct encoder you used during training:

```bash
python visualize_pretrained_model.py \
    --model_path best_model_FPN.pth \
    --encoder resnet50  # or whatever you used
```

---

## 🧪 Advanced Usage

### **Programmatic Use (Python Script)**

You can also import and use the function directly:

```python
from visualize_pretrained_model import visualize_pretrained_model

visualize_pretrained_model(
    model_path='best_model_FPN.pth',
    num_samples=10,
    split='test',
    output_dir='my_output',
    selected_classes=['root', 'hypocotyl', 'cotyledon', 'seed']  # Exclude background
)
```

### **Visualize Only Specific Classes**

Edit the script to modify `selected_classes`:

```python
# Around line 385, change:
selected_classes = ['root', 'hypocotyl', 'cotyledon']  # Skip background and seed
```

---

## 📊 Performance

**Processing time per sample (approximate):**
- TP/FP/FN maps: ~0.5 seconds
- Grad-CAM (5 classes): ~2-3 seconds
- **Total: ~3.5 seconds per sample**

**For 5 samples:** ~17 seconds total

**For 10 samples:** ~35 seconds total

---

## ✅ Checklist Before Running

- [ ] Pretrained model file exists (`.pth`)
- [ ] Dataset directory exists (`phenocyte_seg/`)
- [ ] Dataset CSV exists (`phenocyte_seg/dataset_split.csv`)
- [ ] Correct encoder specified (same as training)
- [ ] Enough disk space for output images

---

## 🎓 Citation

If you use this visualization script in your research, please cite:

```bibtex
@article{gradcam2016,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  journal={arXiv preprint arXiv:1610.02391},
  year={2016}
}
```

---

## 📞 Help

For issues or questions, check:
1. Model path is correct
2. Dataset structure matches expected format
3. Dependencies installed: `torch`, `segmentation_models_pytorch`, `opencv-python`, `matplotlib`

---

**Happy Visualizing! 🎨**
