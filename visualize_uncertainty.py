#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uncertainty Quantification for Pretrained Segmentation Model

Uses Monte Carlo Dropout to estimate prediction uncertainty.
Shows where the model is confident vs uncertain.

Key features:
- Per-pixel uncertainty maps
- Confidence scores
- Identify ambiguous regions
- Works with any dropout-enabled model

How it works:
1. Enable dropout at test time
2. Run model multiple times (e.g., 20 forward passes)
3. Collect predictions
4. Compute variance/entropy = uncertainty

High uncertainty = Model is guessing (don't trust)
Low uncertainty = Model is confident (can trust)

Usage:
    python visualize_uncertainty.py --model_path best_model_FPN.pth \
                                     --num_samples 5 \
                                     --num_mc_samples 20
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_ROOT = Path("phenocyte_seg")
IMG_DIR = DATA_ROOT / "images"
MSK_DIR = DATA_ROOT / "masks"
SPLIT_CSV = DATA_ROOT / "dataset_split.csv"

CLASSES = ["background", "root", "hypocotyl", "cotyledon", "seed"]
NUM_CLASSES = len(CLASSES)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _load_mask_ids(path: Path) -> np.ndarray:
    """Load mask and return HxW integer array with class IDs."""
    arr = np.array(Image.open(path))

    if arr.ndim == 2:
        return arr.astype(np.int64)

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        h, w, _ = arr.shape
        uniq, inv = np.unique(arr.reshape(-1, 3), axis=0, return_inverse=True)
        return inv.reshape(h, w).astype(np.int64)

    raise ValueError(f"Unsupported mask shape: {arr.shape}")


class PhenocyteDataset(Dataset):
    CLASSES = ["background", "root", "hypocotyl", "cotyledon", "seed"]

    def __init__(
        self,
        split: str,
        classes: list = None,
        augmentation=None,
        preprocessing=None,
        mask_suffix: str = "_mask",
    ):
        df = pd.read_csv(SPLIT_CSV)
        self.items = df[df["set"] == split]["img_name"].tolist()
        self.items = [name.replace('%', '_') for name in self.items]
        self.mask_suffix = mask_suffix
        self.split = split

        if classes is None:
            classes = self.CLASSES

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name = self.items[i]
        image_path = IMG_DIR / name
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base = Path(name).stem
        msk_name = f"{base}{self.mask_suffix}.png" if self.mask_suffix else f"{base}.png"
        mask = _load_mask_ids(MSK_DIR / msk_name)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask


def get_validation_augmentation(img_size=(512, 256)):
    """Validation augmentation (no random transforms)."""
    test_transform = [
        albu.LongestMaxSize(max_size=max(img_size), interpolation=1),
        albu.PadIfNeeded(min_height=img_size[1], min_width=img_size[0], border_mode=0),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def tensor_to_uint8(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert normalized tensor to uint8 image."""
    img = img_tensor.clone().detach().cpu().numpy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def overlay_map_on_image(img_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on image."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


# ============================================================================
# MONTE CARLO DROPOUT FOR UNCERTAINTY QUANTIFICATION
# ============================================================================
def enable_dropout(model):
    """Enable dropout layers during test time."""
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for Uncertainty Quantification

    Estimates uncertainty by running the model multiple times with
    dropout enabled at test time.

    How it works:
    1. Enable dropout at test time
    2. Run model N times (e.g., 20)
    3. Collect N predictions
    4. Compute variance/entropy = uncertainty

    High variance = High uncertainty (model guessing)
    Low variance = Low uncertainty (model confident)

    Advantages:
    - Easy to implement (no model changes)
    - Works with any dropout-enabled model
    - Gives calibrated uncertainty estimates
    - Fast (parallel forward passes)
    """

    def __init__(self, model, num_mc_samples=20, dropout_rate=None):
        """
        Args:
            model: Trained segmentation model with dropout
            num_mc_samples: Number of forward passes
            dropout_rate: Optional dropout rate (uses model's default if None)
        """
        self.model = model
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate

    def predict_with_uncertainty(self, input_tensor, verbose=True):
        """
        Run MC Dropout to get predictions with uncertainty.

        Args:
            input_tensor: (1, 3, H, W) input image
            verbose: Show progress bar

        Returns:
            mean_prediction: (C, H, W) mean prediction across MC samples
            uncertainty_variance: (H, W) variance-based uncertainty
            uncertainty_entropy: (H, W) entropy-based uncertainty
            all_predictions: (N, C, H, W) all MC predictions (optional)
        """
        device = input_tensor.device
        _, C_in, H, W = input_tensor.shape

        # Enable dropout
        enable_dropout(self.model)

        # Collect predictions
        predictions = []

        iterator = range(self.num_mc_samples)
        if verbose:
            iterator = tqdm(iterator, desc="  MC Dropout samples")

        for _ in iterator:
            with torch.no_grad():
                logits = self.model(input_tensor)  # (1, C, H, W)
                probs = torch.softmax(logits, dim=1)  # Convert to probabilities
                predictions.append(probs[0].cpu().numpy())  # (C, H, W)

        # Stack predictions: (N, C, H, W)
        predictions = np.stack(predictions, axis=0)

        # Compute mean prediction
        mean_prediction = predictions.mean(axis=0)  # (C, H, W)

        # Compute variance-based uncertainty
        # Variance across MC samples for predicted class
        predicted_class = mean_prediction.argmax(axis=0)  # (H, W)
        variance_map = np.zeros((H, W), dtype=np.float32)

        for h in range(H):
            for w in range(W):
                cls = predicted_class[h, w]
                # Variance of predictions for this class at this pixel
                variance_map[h, w] = predictions[:, cls, h, w].var()

        # Normalize variance to [0, 1]
        if variance_map.max() > 0:
            variance_map = variance_map / variance_map.max()

        # Compute entropy-based uncertainty
        # Entropy of mean prediction
        epsilon = 1e-10
        entropy_map = -(mean_prediction * np.log(mean_prediction + epsilon)).sum(axis=0)

        # Normalize entropy to [0, 1]
        max_entropy = np.log(mean_prediction.shape[0])  # log(num_classes)
        if max_entropy > 0:
            entropy_map = entropy_map / max_entropy

        # Set model back to eval mode
        self.model.eval()

        return mean_prediction, variance_map, entropy_map, predictions


# ============================================================================
# TP/FP/FN VISUALIZATION
# ============================================================================
def show_tp_fp_fn_binary(image, gt_binary, pred_binary,
                         class_name="class", alpha=0.5, save_path=None):
    """Create 4-panel TP/FP/FN visualization."""
    if hasattr(image, "detach"):
        img = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img = image
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    img = np.clip(img, 0, 1)

    gt = gt_binary.astype(bool)
    pr = pred_binary.astype(bool)

    tp = gt & pr
    fp = (~gt) & pr
    fn = gt & (~pr)

    H, W = gt.shape
    overlay = np.zeros((H, W, 3), dtype=float)
    overlay[tp] = [0, 1, 0]
    overlay[fp] = [1, 0, 0]
    overlay[fn] = [0, 0, 1]

    blended = (1 - alpha) * img + alpha * overlay
    blended = np.clip(blended, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title(f"GT: {class_name}")
    axes[1].axis("off")

    axes[2].imshow(pr, cmap="gray")
    axes[2].set_title(f"Pred: {class_name}")
    axes[2].axis("off")

    axes[3].imshow(blended)
    axes[3].set_title(f"TP/FP/FN for {class_name}\n(G=TP, R=FP, B=FN)")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)


def visualize_uncertainty(img_np, mean_pred, uncertainty_variance, uncertainty_entropy,
                         class_names, alpha=0.5):
    """
    Create comprehensive uncertainty visualization.

    Args:
        img_np: (H, W, 3) uint8 image
        mean_pred: (C, H, W) mean prediction
        uncertainty_variance: (H, W) variance-based uncertainty
        uncertainty_entropy: (H, W) entropy-based uncertainty
        class_names: List of class names
        alpha: Overlay transparency
    """
    # Get predicted class
    pred_class = mean_pred.argmax(axis=0)  # (H, W)

    # Create colored prediction mask
    H, W = pred_class.shape
    pred_color = np.zeros((H, W, 3), dtype=np.uint8)

    # Simple color mapping for classes
    colors = [
        [0, 0, 0],       # background - black
        [0, 255, 0],     # root - green
        [255, 255, 0],   # hypocotyl - yellow
        [0, 255, 255],   # cotyledon - cyan
        [255, 0, 255],   # seed - magenta
    ]

    for cls_idx in range(len(class_names)):
        mask = pred_class == cls_idx
        pred_color[mask] = colors[cls_idx][:3] if cls_idx < len(colors) else [128, 128, 128]

    # Overlay uncertainty on image
    variance_overlay = overlay_map_on_image(img_np, uncertainty_variance, alpha=alpha)
    entropy_overlay = overlay_map_on_image(img_np, uncertainty_entropy, alpha=alpha)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original Image", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_color)
    axes[0, 1].set_title("Predicted Classes", fontsize=12)
    axes[0, 1].axis('off')

    # Confidence map (1 - max_uncertainty)
    max_confidence = mean_pred.max(axis=0)
    axes[0, 2].imshow(max_confidence, cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 2].set_title("Confidence Map\n(Green=High, Red=Low)", fontsize=12)
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(uncertainty_variance, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title("Variance Uncertainty\n(Blue=Low, Red=High)", fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(uncertainty_entropy, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title("Entropy Uncertainty\n(Blue=Low, Red=High)", fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(variance_overlay)
    axes[1, 2].set_title("Variance Overlay on Image", fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================
def visualize_pretrained_model(
    model_path: str,
    num_samples: int = 5,
    split: str = 'test',
    encoder: str = 'se_resnext50_32x4d',
    encoder_weights: str = 'imagenet',
    selected_classes: list = None,
    img_size: tuple = (512, 256),
    output_dir: str = 'uncertainty_visualizations',
    num_mc_samples: int = 20
):
    """
    Load pretrained model and generate uncertainty visualizations.

    Args:
        model_path: Path to pretrained model (.pth file)
        num_samples: Number of samples to visualize
        split: Dataset split ('test', 'val', or 'train')
        encoder: Encoder architecture
        encoder_weights: Encoder weights
        selected_classes: List of class names
        img_size: Image size (width, height)
        output_dir: Output directory
        num_mc_samples: Number of MC Dropout samples
    """
    if selected_classes is None:
        selected_classes = CLASSES

    print(f"{'='*80}")
    print(f"Uncertainty Quantification via Monte Carlo Dropout")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Model path: {model_path}")
    print(f"Dataset split: {split}")
    print(f"Number of samples: {num_samples}")
    print(f"MC Dropout samples: {num_mc_samples}")
    print(f"Classes: {selected_classes}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Load model
    print("Loading pretrained model...")
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully\n")

    # Check if model has dropout
    has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in model.modules())
    if not has_dropout:
        print("⚠️  WARNING: Model has no dropout layers!")
        print("   MC Dropout requires dropout layers for uncertainty estimation.")
        print("   Results may not be meaningful.\n")

    # Setup preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    # Create dataset
    print(f"Loading {split} dataset...")
    dataset = PhenocyteDataset(
        split=split,
        classes=selected_classes,
        augmentation=get_validation_augmentation(img_size),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")

    num_samples = min(num_samples, len(dataset))

    # Create output directories
    tpfpfn_dir = os.path.join(output_dir, "tpfpfn_maps")
    uncertainty_dir = os.path.join(output_dir, "uncertainty_maps")
    os.makedirs(tpfpfn_dir, exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)

    # Setup MC Dropout
    print("Setting up MC Dropout...")
    mc_dropout = MCDropoutUncertainty(model, num_mc_samples=num_mc_samples)
    print("✓ MC Dropout ready\n")

    # Unnormalize helper
    def unnormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        return img_tensor * std + mean

    # Process samples
    print(f"{'='*80}")
    print(f"Processing {num_samples} samples...")
    print(f"{'='*80}\n")

    for sample_idx in range(num_samples):
        print(f"[{sample_idx + 1}/{num_samples}] Processing sample {sample_idx}...")

        # Get sample
        image, mask = dataset[sample_idx]
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)

        # Get ground truth
        masks_idx = torch.argmax(mask_tensor, dim=1)[0]

        # Unnormalize image
        img_unnorm = unnormalize(image_tensor[0]).cpu()

        # ================================================================
        # MONTE CARLO DROPOUT PREDICTION WITH UNCERTAINTY
        # ================================================================
        print(f"  Running MC Dropout ({num_mc_samples} samples)...")

        mean_pred, uncertainty_var, uncertainty_ent, all_preds = \
            mc_dropout.predict_with_uncertainty(image_tensor, verbose=True)

        # Get predicted class
        preds_idx = mean_pred.argmax(axis=0)

        # ================================================================
        # GENERATE TP/FP/FN MAPS
        # ================================================================
        print(f"  Generating TP/FP/FN maps...")
        for cls_idx, cls_name in enumerate(selected_classes):
            gt_binary = (masks_idx.cpu().numpy() == cls_idx)
            pred_binary = (preds_idx == cls_idx)

            save_path = os.path.join(
                tpfpfn_dir,
                f"sample{sample_idx}_tpfpfn_{cls_name}.png"
            )

            show_tp_fp_fn_binary(
                image=img_unnorm,
                gt_binary=gt_binary,
                pred_binary=pred_binary,
                class_name=cls_name,
                alpha=0.5,
                save_path=save_path,
            )

        # ================================================================
        # GENERATE UNCERTAINTY VISUALIZATION
        # ================================================================
        print(f"  Generating uncertainty visualization...")
        img_np = tensor_to_uint8(image_tensor[0].cpu())

        fig = visualize_uncertainty(
            img_np,
            mean_pred,
            uncertainty_var,
            uncertainty_ent,
            selected_classes
        )

        # Save
        save_path = os.path.join(
            uncertainty_dir,
            f"sample{sample_idx}_uncertainty.png"
        )
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Compute statistics
        mean_variance = uncertainty_var.mean()
        max_variance = uncertainty_var.max()
        mean_entropy = uncertainty_ent.mean()

        print(f"  Uncertainty Statistics:")
        print(f"    Mean Variance: {mean_variance:.4f}")
        print(f"    Max Variance: {max_variance:.4f}")
        print(f"    Mean Entropy: {mean_entropy:.4f}")
        print(f"  ✓ Sample {sample_idx} complete\n")

    print(f"{'='*80}")
    print(f"Uncertainty Quantification Complete!")
    print(f"{'='*80}")
    print(f"TP/FP/FN maps saved to: {tpfpfn_dir}/")
    print(f"Uncertainty maps saved to: {uncertainty_dir}/")
    print(f"{'='*80}\n")
    print("Interpretation:")
    print("  - Blue regions = Low uncertainty (Model is confident)")
    print("  - Red regions = High uncertainty (Model is guessing)")
    print("  - Use low uncertainty regions for reliable predictions")
    print(f"{'='*80}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate uncertainty maps via Monte Carlo Dropout'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to pretrained model (.pth file)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to visualize (default: 5)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to use (default: test)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='uncertainty_visualizations',
        help='Output directory (default: uncertainty_visualizations)'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='se_resnext50_32x4d',
        help='Encoder architecture (default: se_resnext50_32x4d)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        nargs=2,
        default=[512, 256],
        help='Image size as width height (default: 512 256)'
    )
    parser.add_argument(
        '--num_mc_samples',
        type=int,
        default=20,
        help='Number of MC Dropout samples (default: 20)'
    )

    args = parser.parse_args()

    visualize_pretrained_model(
        model_path=args.model_path,
        num_samples=args.num_samples,
        split=args.split,
        encoder=args.encoder,
        img_size=tuple(args.img_size),
        output_dir=args.output_dir,
        num_mc_samples=args.num_mc_samples
    )


if __name__ == "__main__":
    main()
