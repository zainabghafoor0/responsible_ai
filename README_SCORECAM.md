# Score-CAM: Gradient-Free Explainability

**Score-CAM** is an improved explainability method that produces **cleaner, more stable visualizations** than Grad-CAM **without using gradients**.

---

## 🎯 Why Score-CAM is Better Than Grad-CAM

| Feature | Grad-CAM | Score-CAM |
|---------|----------|-----------|
| **Uses Gradients** | ✅ Yes | ❌ No |
| **Stability** | Can be noisy | More stable |
| **Small Objects** | Sometimes misses | Better localization |
| **Multiple Instances** | Can be confused | Handles well |
| **Noise** | More noisy | Cleaner heatmaps |
| **Speed** | Faster | Slower (but more accurate) |

---

## 🧠 How Score-CAM Works

### **Grad-CAM Approach:**
```
1. Forward pass → get activations
2. Backward pass → get gradients
3. Weight activations by gradients
4. Create heatmap
```

### **Score-CAM Approach (No Gradients!):**
```
1. Forward pass → get activation maps
2. For each activation map:
   - Mask input with activation
   - Forward pass masked input
   - Measure increase in target class score
3. Weight activations by scores (not gradients!)
4. Create heatmap
```

**Key insight:** Instead of using gradients to find important features, Score-CAM uses **direct measurement** of how much each activation map increases the target class score.

---

## 🚀 Quick Start

### **Basic Usage**

```bash
python visualize_scorecam.py --model_path best_model_FPN.pth
```

This will:
- Load your pretrained model
- Generate **TP/FP/FN** error maps
- Generate **Score-CAM** explainability heatmaps
- Save to `scorecam_visualizations/`

---

## 📋 Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | **Required** | Path to `.pth` model file |
| `--num_samples` | `5` | Number of samples to visualize |
| `--split` | `test` | Dataset split (`train`, `val`, `test`) |
| `--output_dir` | `scorecam_visualizations` | Output directory |
| `--batch_size` | `32` | Batch size for processing |
| `--max_activation_maps` | `None` | Limit activation maps (for speed) |
| `--encoder` | `se_resnext50_32x4d` | Encoder architecture |
| `--img_size` | `512 256` | Image size |

---

## 📝 Usage Examples

### **Example 1: Basic Score-CAM visualization**

```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --num_samples 5
```

### **Example 2: Faster processing (limit activation maps)**

```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --num_samples 5 \
    --max_activation_maps 100
```

**Note:** Limiting activation maps makes Score-CAM faster but may reduce quality.

### **Example 3: Validation set with custom output**

```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --split val \
    --num_samples 10 \
    --output_dir my_scorecam_results
```

### **Example 4: Adjust batch size for memory**

```bash
# Lower batch size if you get CUDA out of memory
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --batch_size 16
```

---

## 📁 Output Structure

```
scorecam_visualizations/
├── tpfpfn_maps/
│   ├── sample0_tpfpfn_background.png
│   ├── sample0_tpfpfn_root.png
│   ├── sample0_tpfpfn_hypocotyl.png
│   ├── sample0_tpfpfn_cotyledon.png
│   └── sample0_tpfpfn_seed.png
│
└── scorecam_maps/
    ├── sample0_scorecam_background.png
    ├── sample0_scorecam_root.png
    ├── sample0_scorecam_hypocotyl.png
    ├── sample0_scorecam_cotyledon.png
    └── sample0_scorecam_seed.png
```

---

## ⚡ Performance Comparison

**Processing time per sample (approximate):**

| Method | Time per Sample | Quality |
|--------|----------------|---------|
| **Grad-CAM** | ~3 seconds | Good |
| **Score-CAM** | ~10-15 seconds | Better |

**Score-CAM is slower** because it:
- Processes each activation map separately
- Requires multiple forward passes
- No gradients = more computation

**Trade-off:** Slower but **more accurate and stable** visualizations.

---

## 🔍 Score-CAM vs Grad-CAM Examples

### **When Score-CAM is Much Better:**

1. **Multiple instances of same class**
   - Grad-CAM: Highlights all instances equally (blurry)
   - Score-CAM: Distinguishes individual instances

2. **Small objects** (e.g., seeds)
   - Grad-CAM: May miss small objects
   - Score-CAM: Better localization

3. **Boundary precision**
   - Grad-CAM: Fuzzy boundaries
   - Score-CAM: Sharper boundaries

4. **Noisy gradients**
   - Grad-CAM: Noisy heatmaps
   - Score-CAM: Cleaner visualizations

### **When Grad-CAM is Fine:**

1. **Large, well-separated objects**
2. **Quick prototyping** (faster)
3. **General understanding** (not publication-quality)

---

## 🎓 Technical Details

### **How Score-CAM Weights Activation Maps:**

```python
# For each activation map k:
masked_input = input * activation_k
score_k = model(masked_input)[class_idx].sum()

# Normalize scores
weights = normalize(scores)

# Weighted combination
cam = sum(weights[k] * activations[k] for k in range(K))
```

### **Key Parameters:**

**`batch_size`:**
- Higher = faster (more GPU memory)
- Lower = slower (less GPU memory)
- Default: 32 (good balance)

**`max_activation_maps`:**
- Limits number of activation maps used
- Lower = faster but less accurate
- Default: None (use all)
- Recommended: 100-200 for speed

---

## 🛠️ Troubleshooting

### **Error: CUDA out of memory**

**Solution 1:** Lower batch size
```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --batch_size 8
```

**Solution 2:** Limit activation maps
```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --max_activation_maps 50
```

### **Score-CAM is too slow**

**Solution:** Use fewer activation maps
```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --max_activation_maps 100
```

This will be ~5x faster with minimal quality loss.

### **Heatmaps look weird**

**Check:**
1. Model is in eval mode (done automatically)
2. Using correct encoder architecture
3. Input normalization is correct

---

## 📊 When to Use Which Method

| Use Case | Recommended Method |
|----------|-------------------|
| **Quick exploration** | Grad-CAM |
| **Publication figures** | Score-CAM |
| **Small objects** | Score-CAM |
| **Multiple instances** | Score-CAM |
| **Clinical/safety-critical** | Score-CAM |
| **Fast iteration** | Grad-CAM |
| **High accuracy needed** | Score-CAM |

---

## 🎨 Example Command for Best Quality

```bash
python visualize_scorecam.py \
    --model_path best_model_FPN.pth \
    --num_samples 5 \
    --split test \
    --batch_size 32 \
    --output_dir high_quality_scorecam
```

This will generate **publication-quality** visualizations with:
- Cleaner heatmaps than Grad-CAM
- Better localization
- More stable across samples

---

## 🔬 Citation

If you use Score-CAM in your research:

```bibtex
@inproceedings{wang2020scorecam,
  title={Score-CAM: Score-weighted visual explanations for convolutional neural networks},
  author={Wang, Haofan and Wang, Zifan and Du, Mengnan and Yang, Fan and Zhang, Zijian and Ding, Sirui and Mardziel, Piotr and Hu, Xia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={24--25},
  year={2020}
}
```

---

## 💡 Pro Tips

1. **Start with Grad-CAM** to quickly understand your model
2. **Use Score-CAM** for final publication figures
3. **Limit activation maps** if speed matters more than perfect accuracy
4. **Compare both methods** to see differences
5. **Adjust batch_size** based on your GPU memory

---

## 🆚 Comparison with visualize_pretrained_model.py

| Feature | `visualize_pretrained_model.py` | `visualize_scorecam.py` |
|---------|--------------------------------|-------------------------|
| Method | Grad-CAM | Score-CAM |
| Uses gradients | Yes | No |
| Speed | Fast (~3s/sample) | Slower (~10s/sample) |
| Quality | Good | Better |
| Stability | Can be noisy | More stable |
| Memory | Low | Higher |

**Recommendation:** Try both and compare! Score-CAM usually gives cleaner results.

---

**Happy Visualizing! 🎨**

For questions or issues, check the main code or compare outputs with Grad-CAM.
