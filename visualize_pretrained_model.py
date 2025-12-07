#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference and Visualization Script for Pretrained Segmentation Model

This script loads a pretrained model and generates:
1. TP/FP/FN error visualization maps
2. Grad-CAM explainability heatmaps

Usage:
    python visualize_pretrained_model.py --model_path best_model_FPN.pth \
                                          --num_samples 5 \
                                          --split test
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
# HELPER FUNCTIONS FROM MAIN SCRIPT
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


# ============================================================================
# TP/FP/FN VISUALIZATION
# ============================================================================
def show_tp_fp_fn_binary(image, gt_binary, pred_binary,
                         class_name="class", alpha=0.5, save_path=None):
    """
    Create 4-panel TP/FP/FN visualization.

    Args:
        image: (C, H, W) torch tensor or (H, W, C) numpy
        gt_binary: (H, W) bool/0-1 array (GT for ONE class)
        pred_binary: (H, W) bool/0-1 array (prediction for ONE class)
        class_name: Name of the class
        alpha: Transparency for overlay
        save_path: Path to save figure
    """
    # Convert image to numpy in [0,1] for display
    if hasattr(image, "detach"):
        img = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img = image
    img = img.astype(np.float32)
    if img.max() > 1.5:   # 0–255 -> 0–1
        img = img / 255.0
    img = np.clip(img, 0, 1)

    gt = gt_binary.astype(bool)
    pr = pred_binary.astype(bool)

    tp = gt & pr
    fp = (~gt) & pr
    fn = gt & (~pr)

    H, W = gt.shape
    overlay = np.zeros((H, W, 3), dtype=float)
    overlay[tp] = [0, 1, 0]   # green = TP
    overlay[fp] = [1, 0, 0]   # red   = FP
    overlay[fn] = [0, 0, 1]   # blue  = FN

    blended = (1 - alpha) * img + alpha * overlay
    blended = np.clip(blended, 0, 1)

    # 4-panel figure
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


# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================
class SegmentationGradCAM:
    """Grad-CAM for semantic segmentation models."""

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    def __del__(self):
        if hasattr(self, 'fwd_handle'):
            self.fwd_handle.remove()
        if hasattr(self, 'bwd_handle'):
            self.bwd_handle.remove()

    def generate(self, image_tensor, class_idx):
        """Generate Grad-CAM heatmap for a specific class."""
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Forward pass
        logits = self.model(image_tensor)  # (1, C, H, W)

        # Sum over all spatial positions (gives scalar)
        target = logits[0, class_idx].sum()

        # Backward pass
        target.backward(retain_graph=True)

        # Get activations and gradients
        A = self.activations
        G = self.gradients

        # Global average pool over spatial dims
        weights = G.mean(dim=(2, 3))[0]  # (K,)

        # Weighted sum over channels
        cam = torch.zeros(A.shape[2:], dtype=torch.float32, device=A.device)
        for k, w in enumerate(weights):
            cam += w * A[0, k, :, :]

        # ReLU
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Resize to input size
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False)[0, 0]

        return cam.detach().cpu().numpy()


def tensor_to_uint8(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert normalized tensor to uint8 image."""
    img = img_tensor.clone().detach().cpu().numpy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


def overlay_cam_on_image(img_np, cam, alpha=0.5):
    """Overlay Grad-CAM heatmap on image."""
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * cam_color + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


def visualize_gradcam(img_np, cam, class_name, alpha=0.5):
    """Create 3-panel Grad-CAM visualization."""
    cam_overlay = overlay_cam_on_image(img_np, cam, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f"Grad-CAM: {class_name}")
    axes[1].axis('off')

    axes[2].imshow(cam_overlay)
    axes[2].set_title(f"Overlay: {class_name}")
    axes[2].axis('off')

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
    output_dir: str = 'pretrained_visualizations'
):
    """
    Load pretrained model and generate visualizations.

    Args:
        model_path: Path to pretrained model (.pth file)
        num_samples: Number of samples to visualize
        split: Dataset split ('test', 'val', or 'train')
        encoder: Encoder architecture
        encoder_weights: Encoder weights
        selected_classes: List of class names to visualize
        img_size: Image size (width, height)
        output_dir: Output directory for visualizations
    """
    if selected_classes is None:
        selected_classes = CLASSES

    print(f"{'='*80}")
    print(f"Visualizing Pretrained Model")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Model path: {model_path}")
    print(f"Dataset split: {split}")
    print(f"Number of samples: {num_samples}")
    print(f"Classes: {selected_classes}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Load model
    print("Loading pretrained model...")
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully\n")

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

    # Limit number of samples
    num_samples = min(num_samples, len(dataset))

    # Create output directories
    tpfpfn_dir = os.path.join(output_dir, "tpfpfn_maps")
    gradcam_dir = os.path.join(output_dir, "gradcam_maps")
    os.makedirs(tpfpfn_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)

    # Setup Grad-CAM
    print("Setting up Grad-CAM...")
    target_layer = model.encoder.layer4
    gradcam = SegmentationGradCAM(model, target_layer)
    print("✓ Grad-CAM ready\n")

    # Unnormalize helper
    def unnormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        return img_tensor * std + mean

    # Process samples
    print(f"{'='*80}")
    print(f"Processing {num_samples} samples...")
    print(f"{'='*80}\n")

    with torch.no_grad():
        for sample_idx in range(num_samples):
            print(f"[{sample_idx + 1}/{num_samples}] Processing sample {sample_idx}...")

            # Get sample
            image, mask = dataset[sample_idx]
            image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)

            # Forward pass
            logits = model(image_tensor)
            preds_idx = torch.argmax(logits, dim=1)[0]  # [H, W]
            masks_idx = torch.argmax(mask_tensor, dim=1)[0]  # [H, W]

            # Unnormalize image for visualization
            img_unnorm = unnormalize(image_tensor[0]).cpu()

            # ================================================================
            # GENERATE TP/FP/FN MAPS FOR EACH CLASS
            # ================================================================
            print(f"  Generating TP/FP/FN maps...")
            for cls_idx, cls_name in enumerate(selected_classes):
                gt_binary = (masks_idx.cpu().numpy() == cls_idx)
                pred_binary = (preds_idx.cpu().numpy() == cls_idx)

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
            # GENERATE GRAD-CAM FOR EACH CLASS
            # ================================================================
            print(f"  Generating Grad-CAM maps...")
            img_np = tensor_to_uint8(image_tensor[0].cpu())

            for cls_idx, cls_name in enumerate(selected_classes):
                # Generate CAM (requires gradients, so no torch.no_grad here)
                with torch.enable_grad():
                    cam = gradcam.generate(image_tensor, cls_idx)

                # Visualize
                fig = visualize_gradcam(img_np, cam, cls_name)

                # Save
                save_path = os.path.join(
                    gradcam_dir,
                    f"sample{sample_idx}_gradcam_{cls_name}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            print(f"  ✓ Sample {sample_idx} complete\n")

    # Cleanup
    del gradcam

    print(f"{'='*80}")
    print(f"Visualization Complete!")
    print(f"{'='*80}")
    print(f"TP/FP/FN maps saved to: {tpfpfn_dir}/")
    print(f"Grad-CAM maps saved to: {gradcam_dir}/")
    print(f"{'='*80}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate TP/FP/FN and Grad-CAM visualizations from pretrained model'
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
        default='pretrained_visualizations',
        help='Output directory (default: pretrained_visualizations)'
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

    args = parser.parse_args()

    visualize_pretrained_model(
        model_path=args.model_path,
        num_samples=args.num_samples,
        split=args.split,
        encoder=args.encoder,
        img_size=tuple(args.img_size),
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
