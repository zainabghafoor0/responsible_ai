#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-Wise Uncertainty Quantification for Segmentation Models

Generates per-class uncertainty maps showing model confidence for each individual class.

Key Differences from visualize_uncertainty.py:
- visualize_uncertainty.py: Shows OVERALL uncertainty across all classes
- THIS FILE: Shows uncertainty for EACH CLASS SEPARATELY

Use Cases:
- Identify which classes the model is uncertain about
- Find regions where the model confuses specific classes
- Analyze class-specific prediction reliability
- Compare uncertainty across different classes

Usage:
    python visualize_uncertainty_classwise.py \\
        --model_path best_model_FPN.pth \\
        --num_samples 5 \\
        --num_mc_samples 20 \\
        --class_idx 2  # Optional: specific class only
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
from torch.utils.data import Dataset
import albumentations as albu
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
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
# DATASET AND PREPROCESSING
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

    def __init__(self, split: str, classes: list = None,
                 augmentation=None, preprocessing=None, mask_suffix: str = "_mask"):
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


# ============================================================================
# CLASS-WISE MC DROPOUT UNCERTAINTY
# ============================================================================
def enable_dropout(model):
    """Enable dropout layers during test time."""
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


class ClassWiseUncertainty:
    """
    Per-Class Monte Carlo Dropout Uncertainty Quantification

    Computes uncertainty for EACH CLASS INDIVIDUALLY, showing:
    - How uncertain the model is about each specific class
    - Which pixels are ambiguous for each class
    - Class-specific confidence scores
    """

    def __init__(self, model, num_mc_samples=20):
        self.model = model
        self.num_mc_samples = num_mc_samples

    def predict_with_classwise_uncertainty(self, input_tensor, class_idx=None, verbose=True):
        """
        Run MC Dropout to get per-class uncertainty.

        Args:
            input_tensor: (1, 3, H, W) input image
            class_idx: Specific class to compute (None = all classes)
            verbose: Show progress bar

        Returns:
            class_uncertainties: Dict mapping class_idx -> (mean_prob, variance, entropy)
            all_predictions: (N, C, H, W) all MC predictions
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
                probs = torch.sigmoid(logits)  # Per-class probabilities (multilabel)
                predictions.append(probs[0].cpu().numpy())  # (C, H, W)

        # Stack predictions: (N, C, H, W)
        predictions = np.stack(predictions, axis=0)

        # Compute per-class statistics
        num_classes = predictions.shape[1]
        class_uncertainties = {}

        classes_to_compute = [class_idx] if class_idx is not None else range(num_classes)

        for cls in classes_to_compute:
            # Get all predictions for this class: (N, H, W)
            class_preds = predictions[:, cls, :, :]

            # Mean probability for this class
            mean_prob = class_preds.mean(axis=0)  # (H, W)

            # Variance-based uncertainty
            variance_map = class_preds.var(axis=0)  # (H, W)

            # Normalize variance to [0, 1]
            if variance_map.max() > 0:
                variance_map_norm = variance_map / variance_map.max()
            else:
                variance_map_norm = variance_map

            # Entropy-based uncertainty for binary classification per class
            epsilon = 1e-10
            p = mean_prob
            entropy_map = -(p * np.log(p + epsilon) + (1 - p) * np.log(1 - p + epsilon))

            # Normalize entropy to [0, 1]
            max_entropy = np.log(2)  # Binary entropy maximum
            entropy_map_norm = entropy_map / max_entropy

            class_uncertainties[cls] = {
                'mean_prob': mean_prob,
                'variance': variance_map,
                'variance_norm': variance_map_norm,
                'entropy': entropy_map,
                'entropy_norm': entropy_map_norm,
            }

        # Set model back to eval mode
        self.model.eval()

        return class_uncertainties, predictions


# ============================================================================
# CLASS-WISE VISUALIZATION
# ============================================================================
def visualize_classwise_uncertainty(img_np, class_uncertainty, class_name,
                                   gt_mask=None, pred_mask=None):
    """
    Create comprehensive visualization for a single class.

    Args:
        img_np: (H, W, 3) uint8 image
        class_uncertainty: Dict with 'mean_prob', 'variance_norm', 'entropy_norm'
        class_name: Name of the class
        gt_mask: (H, W) boolean ground truth mask for this class
        pred_mask: (H, W) boolean predicted mask for this class

    Returns:
        fig: Matplotlib figure
    """
    mean_prob = class_uncertainty['mean_prob']
    variance = class_uncertainty['variance_norm']
    entropy = class_uncertainty['entropy_norm']

    # Threshold for prediction
    pred_binary = (mean_prob > 0.5)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Class-Wise Uncertainty Analysis: {class_name}',
                 fontsize=18, fontweight='bold', color='#8B0000')

    # Row 1: Basic visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_np)
    ax1.set_title("Input Image", fontsize=12, fontweight='bold')
    ax1.axis('off')

    if gt_mask is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(gt_mask, cmap='gray')
        ax2.set_title(f"Ground Truth: {class_name}", fontsize=12, fontweight='bold')
        ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred_binary, cmap='gray')
    ax3.set_title(f"Prediction: {class_name}", fontsize=12, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(mean_prob, cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_title(f"Mean Probability\n(Green=High, Red=Low)", fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Row 2: Uncertainty maps
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(variance, cmap='jet', vmin=0, vmax=1)
    ax5.set_title(f"Variance Uncertainty\n(Blue=Confident, Red=Uncertain)",
                 fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(entropy, cmap='jet', vmin=0, vmax=1)
    ax6.set_title(f"Entropy Uncertainty\n(Blue=Confident, Red=Uncertain)",
                 fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # Confidence map (1 - entropy)
    confidence = 1 - entropy
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    ax7.set_title(f"Confidence Map\n(Green=High, Red=Low)",
                 fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    # High uncertainty regions (entropy > 0.5)
    high_uncertainty = entropy > 0.5
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(high_uncertainty, cmap='Reds')
    ax8.set_title(f"High Uncertainty Regions\n(Entropy > 0.5)",
                 fontsize=12, fontweight='bold')
    ax8.axis('off')

    # Row 3: Overlays and TP/FP/FN
    # Overlay uncertainty on image
    uncertainty_overlay = np.copy(img_np)
    high_unc_mask = entropy > 0.5
    uncertainty_overlay[high_unc_mask] = [255, 0, 0]  # Red for high uncertainty

    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(uncertainty_overlay)
    ax9.set_title(f"High Uncertainty Overlay\n(Red = Uncertain)",
                 fontsize=12, fontweight='bold')
    ax9.axis('off')

    # TP/FP/FN with uncertainty
    if gt_mask is not None:
        tp = gt_mask & pred_binary
        fp = (~gt_mask) & pred_binary
        fn = gt_mask & (~pred_binary)

        # TP/FP/FN overlay
        tpfpfn_overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        tpfpfn_overlay[tp] = [0, 255, 0]   # Green = TP
        tpfpfn_overlay[fp] = [255, 0, 0]   # Red = FP
        tpfpfn_overlay[fn] = [0, 0, 255]   # Blue = FN

        ax10 = fig.add_subplot(gs[2, 1])
        blended = (0.5 * img_np + 0.5 * tpfpfn_overlay).astype(np.uint8)
        ax10.imshow(blended)
        ax10.set_title(f"TP/FP/FN\n(G=TP, R=FP, B=FN)",
                      fontsize=12, fontweight='bold')
        ax10.axis('off')

        # Uncertainty in FP regions
        fp_uncertainty = np.zeros_like(entropy)
        fp_uncertainty[fp] = entropy[fp]

        ax11 = fig.add_subplot(gs[2, 2])
        im11 = ax11.imshow(fp_uncertainty, cmap='Reds', vmin=0, vmax=1)
        ax11.set_title(f"Uncertainty in False Positives",
                      fontsize=12, fontweight='bold')
        ax11.axis('off')
        plt.colorbar(im11, ax=ax11, fraction=0.046, pad=0.04)

        # Uncertainty in FN regions
        fn_uncertainty = np.zeros_like(entropy)
        fn_uncertainty[fn] = entropy[fn]

        ax12 = fig.add_subplot(gs[2, 3])
        im12 = ax12.imshow(fn_uncertainty, cmap='Blues', vmin=0, vmax=1)
        ax12.set_title(f"Uncertainty in False Negatives",
                      fontsize=12, fontweight='bold')
        ax12.axis('off')
        plt.colorbar(im12, ax=ax12, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate per-class uncertainty maps'
    )
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model (.pth file)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize (default: 5)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use (default: test)')
    parser.add_argument('--output_dir', type=str,
                       default='classwise_uncertainty_visualizations',
                       help='Output directory')
    parser.add_argument('--encoder', type=str, default='se_resnext50_32x4d',
                       help='Encoder architecture')
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 256],
                       help='Image size as width height (default: 512 256)')
    parser.add_argument('--num_mc_samples', type=int, default=20,
                       help='Number of MC Dropout samples (default: 20)')
    parser.add_argument('--class_idx', type=int, default=None,
                       help='Specific class index to visualize (default: all classes)')
    parser.add_argument('--class_name', type=str, default=None,
                       help='Specific class name to visualize')

    args = parser.parse_args()

    # Resolve class index if class name provided
    if args.class_name:
        args.class_idx = CLASSES.index(args.class_name.lower())

    print(f"{'='*80}")
    print(f"Class-Wise Uncertainty Quantification")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model_path}")
    print(f"MC Samples: {args.num_mc_samples}")
    if args.class_idx is not None:
        print(f"Analyzing Class: {CLASSES[args.class_idx]} (index {args.class_idx})")
    else:
        print(f"Analyzing All Classes: {CLASSES}")
    print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()
    print("✓ Model loaded\n")

    # Setup preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, 'imagenet')

    # Create dataset
    dataset = PhenocyteDataset(
        split=args.split,
        classes=CLASSES,
        augmentation=get_validation_augmentation(tuple(args.img_size)),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    num_samples = min(args.num_samples, len(dataset))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup uncertainty estimator
    uncertainty_estimator = ClassWiseUncertainty(model, num_mc_samples=args.num_mc_samples)

    # Unnormalize helper
    def unnormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        return img_tensor * std + mean

    # Process samples
    print(f"Processing {num_samples} samples...\n")

    for sample_idx in range(num_samples):
        print(f"[{sample_idx + 1}/{num_samples}] Sample {sample_idx}...")

        # Get sample
        image, mask = dataset[sample_idx]
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)

        # Get ground truth class indices
        masks_idx = torch.argmax(mask_tensor, dim=1)[0]

        # Get unnormalized image
        img_np = tensor_to_uint8(image_tensor[0].cpu())

        # Compute class-wise uncertainty
        class_uncertainties, all_preds = \
            uncertainty_estimator.predict_with_classwise_uncertainty(
                image_tensor, class_idx=args.class_idx, verbose=True
            )

        # Generate visualizations for each class
        classes_to_viz = [args.class_idx] if args.class_idx is not None else range(NUM_CLASSES)

        for cls_idx in classes_to_viz:
            class_name = CLASSES[cls_idx]
            print(f"  Generating visualization for: {class_name}")

            # Get ground truth and prediction for this class
            gt_mask = (masks_idx.cpu().numpy() == cls_idx)
            pred_mask = (class_uncertainties[cls_idx]['mean_prob'] > 0.5)

            # Create visualization
            fig = visualize_classwise_uncertainty(
                img_np,
                class_uncertainties[cls_idx],
                class_name,
                gt_mask=gt_mask,
                pred_mask=pred_mask
            )

            # Save
            save_path = os.path.join(
                args.output_dir,
                f"sample{sample_idx}_class_{class_name}_uncertainty.png"
            )
            fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            # Print statistics
            unc = class_uncertainties[cls_idx]
            print(f"    Mean Variance: {unc['variance_norm'].mean():.4f}")
            print(f"    Mean Entropy: {unc['entropy_norm'].mean():.4f}")
            print(f"    High Uncertainty Pixels: {(unc['entropy_norm'] > 0.5).sum()}")

        print(f"  ✓ Sample {sample_idx} complete\n")

    print(f"{'='*80}")
    print(f"Class-wise uncertainty visualizations saved to: {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
