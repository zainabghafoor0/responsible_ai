# Explainability Methods for Semantic Segmentation

This repository now includes three explainability methods to interpret your segmentation model predictions:

1. **GradCAM** - Gradient-weighted Class Activation Mapping (already in main script)
2. **SHAP** - SHapley Additive exPlanations (NEW)
3. **LIME** - Local Interpretable Model-agnostic Explanations (NEW)

---

## 🎯 Quick Comparison

| Method | Speed | Granularity | Model-Agnostic | Best For |
|--------|-------|-------------|----------------|----------|
| **GradCAM** | ⚡ Fast | Region-level | ❌ No | Quick attention maps during training |
| **SHAP** | ⚡⚡ Medium | Pixel-level | ✅ Yes | Understanding pixel importance |
| **LIME** | ⚡⚡⚡ Slow | Superpixel | ✅ Yes | Local explanations of misclassifications |

---

## 📦 Installation

Install required packages:

```bash
# For SHAP
pip install shap

# For LIME
pip install lime scikit-image
```

---

## 🚀 Usage

### 1. SHAP Explanations

SHAP provides **pixel-level importance scores** showing which pixels contribute most to predictions.

#### Basic Usage:

```bash
python visualize_shap.py \
    --model best_model_FPN.pth \
    --image phenocyte_seg/images/sample_001.jpg \
    --class_idx 2 \
    --class_name hypocotyl \
    --method gradient \
    --output_dir shap_explanations
```

#### SHAP Methods:

**GradientSHAP (Recommended - Fast):**
```bash
python visualize_shap.py \
    --model best_model_FPN.pth \
    --image path/to/image.jpg \
    --class_idx 2 \
    --class_name hypocotyl \
    --method gradient \
    --num_samples 50
```

**DeepSHAP (Medium Speed):**
```bash
python visualize_shap.py \
    --model best_model_FPN.pth \
    --image path/to/image.jpg \
    --class_idx 2 \
    --method deep \
    --num_samples 50
```

**KernelSHAP (Slow but Model-Agnostic):**
```bash
python visualize_shap.py \
    --model best_model_FPN.pth \
    --image path/to/image.jpg \
    --class_idx 2 \
    --method kernel \
    --num_samples 100
```

#### SHAP Output:
- **Input Image**: Original image
- **SHAP Importance Map**: Heatmap showing pixel importance (red=high, blue=low)
- **SHAP Overlay**: Importance map blended with original image
- **Ground Truth**: (if available) GT mask
- **Prediction**: Model's prediction mask

---

### 2. LIME Explanations

LIME provides **superpixel-based explanations** by perturbing regions and observing prediction changes.

#### Basic Usage:

```bash
python visualize_lime.py \
    --model best_model_FPN.pth \
    --image phenocyte_seg/images/sample_001.jpg \
    --class_idx 2 \
    --class_name hypocotyl \
    --segmentation slic \
    --num_samples 1000 \
    --output_dir lime_explanations
```

#### Segmentation Algorithms:

**SLIC (Recommended - Balanced):**
```bash
python visualize_lime.py \
    --model best_model_FPN.pth \
    --image path/to/image.jpg \
    --class_idx 2 \
    --segmentation slic \
    --n_segments 50 \
    --num_features 10
```

**Felzenszwalb (Better for Irregular Shapes):**
```bash
python visualize_lime.py \
    --model best_model_FPN.pth \
    --image path/to/image.jpg \
    --class_idx 2 \
    --segmentation felzenszwalb \
    --num_samples 1000
```

**Quickshift (Fast Superpixels):**
```bash
python visualize_lime.py \
    --model best_model_FPN.pth \
    --image path/to/image.jpg \
    --class_idx 2 \
    --segmentation quickshift \
    --num_samples 1000
```

#### LIME Output:
- **Superpixel Segmentation**: Image divided into interpretable regions
- **Positive Features**: Superpixels supporting the prediction (GREEN)
- **Negative Features**: Superpixels against the prediction (RED)
- **Importance Heatmap**: Continuous importance map (red=positive, blue=negative)
- **Ground Truth & Prediction**: (if available)

---

### 3. GradCAM (Already Integrated)

GradCAM is already integrated into the main training script and runs automatically.

Standalone GradCAM visualization is also available via the existing code.

---

## 🔢 Class Indices

Use these class indices for your plant segmentation model:

```python
CLASSES = {
    0: "background",
    1: "root",
    2: "hypocotyl",
    3: "cotyledon",
    4: "seed"
}
```

---

## 📊 Complete Workflow Example

### Analyze a Misclassified Region:

```bash
# 1. Get SHAP explanation (pixel importance)
python visualize_shap.py \
    --model best_model_FPN.pth \
    --image phenocyte_seg/images/sample_042.jpg \
    --class_idx 2 \
    --class_name hypocotyl \
    --method gradient

# 2. Get LIME explanation (superpixel importance)
python visualize_lime.py \
    --model best_model_FPN.pth \
    --image phenocyte_seg/images/sample_042.jpg \
    --class_idx 2 \
    --class_name hypocotyl \
    --num_samples 1000

# 3. Compare with GradCAM (from training script output)
# Check: gradcam_visualizations/sample_X/gradcam_hypocotyl.png
```

---

## 🎨 Visualization Comparison

### When to Use Each Method:

#### **SHAP - Best for:**
- Understanding **why** a specific pixel was classified
- Analyzing **feature importance** with theoretical guarantees
- Comparing **positive vs negative contributions**
- **Publication-quality** explanations with solid theory

#### **LIME - Best for:**
- Explaining **local predictions** in specific regions
- Understanding **misclassifications** in TP/FP/FN analysis
- **Intuitive** superpixel-based explanations
- **Model-agnostic** explanations (works with any model)

#### **GradCAM - Best for:**
- **Quick overview** during training
- Understanding model **attention**
- **Fast** visualization for many samples
- Identifying what regions the model **"looks at"**

---

## 💡 Tips for Better Explanations

### SHAP:
- Use `gradient` method for speed (10-30 seconds per image)
- Increase `num_samples` for more stable explanations (but slower)
- SHAP values show **contribution**, not just attention

### LIME:
- Adjust `n_segments` based on image complexity (20-100)
- More `num_samples` = more accurate but slower (500-2000)
- Try different segmentation algorithms for different image types
- SLIC works well for plant images with regular structures

### GradCAM:
- Use deeper layers for semantic information
- Use shallower layers for fine details
- Already optimized in the training script

---

## 📈 Integration with TP/FP/FN Analysis

Combine explainability with error analysis:

```python
# Pseudo-workflow:
# 1. Run training → generates TP/FP/FN maps
# 2. Identify problematic samples (high FP or FN)
# 3. Run SHAP/LIME on those samples to understand WHY
# 4. Use insights to improve model or data
```

### Example Analysis:
```bash
# Found high FP for "hypocotyl" in sample 042
# → Run SHAP to see which pixels contributed to false positive
python visualize_shap.py --model best_model_FPN.pth \
    --image phenocyte_seg/images/sample_042.jpg \
    --class_idx 2 --class_name hypocotyl

# → Run LIME to see which superpixels are problematic
python visualize_lime.py --model best_model_FPN.pth \
    --image phenocyte_seg/images/sample_042.jpg \
    --class_idx 2 --class_name hypocotyl
```

---

## 📝 Arguments Reference

### SHAP Arguments:
```
--model         : Path to pretrained model (required)
--image         : Path to input image (required)
--class_idx     : Class index to explain (default: 2)
--class_name    : Class name for visualization (default: 'hypocotyl')
--method        : SHAP method (gradient/deep/kernel, default: gradient)
--num_samples   : Background samples (default: 50)
--output_dir    : Output directory (default: 'shap_explanations')
--device        : Device (cuda/cpu, default: cuda)
```

### LIME Arguments:
```
--model         : Path to pretrained model (required)
--image         : Path to input image (required)
--class_idx     : Class index to explain (default: 2)
--class_name    : Class name for visualization (default: 'hypocotyl')
--segmentation  : Algorithm (slic/felzenszwalb/quickshift, default: slic)
--num_samples   : Perturbed samples (default: 1000)
--num_features  : Superpixels to highlight (default: 10)
--n_segments    : Number of superpixels (default: 50, for SLIC)
--output_dir    : Output directory (default: 'lime_explanations')
--device        : Device (cuda/cpu, default: cuda)
```

---

## 🔬 Research & Publications

If using these explainability methods in your research, please cite:

**SHAP:**
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.

**LIME:**
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining predictions of any classifier. KDD.

**GradCAM:**
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks. ICCV.

---

## 🐛 Troubleshooting

**SHAP is too slow:**
- Use `gradient` method instead of `kernel`
- Reduce `num_samples` (try 20-30)
- Use smaller images

**LIME takes forever:**
- Reduce `num_samples` (try 500)
- Reduce `n_segments` (try 30)
- Use `quickshift` segmentation

**Out of memory:**
- Use `--device cpu`
- Process images in smaller batches
- Reduce image size

**SHAP/LIME not installed:**
```bash
pip install shap lime scikit-image
```

---

## 📧 Support

For issues or questions about explainability methods, check:
- SHAP documentation: https://shap.readthedocs.io/
- LIME documentation: https://lime-ml.readthedocs.io/
- Your repository issues page

---

**Happy Explaining! 🎉**
