#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score-CAM Visualization Script for Pretrained Segmentation Model

Score-CAM is an improvement over Grad-CAM that doesn't use gradients.
It produces cleaner, more stable visualizations by using activation-based weighting.

Key advantages over Grad-CAM:
- No gradients needed (more stable)
- Less noisy visualizations
- Better localization for multiple instances
- Works better for small objects

Usage:
    python visualize_scorecam.py --model_path best_model_FPN.pth \
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


def overlay_cam_on_image(img_np, cam, alpha=0.5):
    """Overlay CAM heatmap on image."""
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * cam_color + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


# ============================================================================
# SCORE-CAM IMPLEMENTATION
# ============================================================================
class ScoreCAM:
    """
    Score-CAM: Score-Weighted Visual Explanations for CNNs

    Unlike Grad-CAM, Score-CAM doesn't use gradients. Instead, it:
    1. Gets activation maps from target layer
    2. Masks input with each activation map
    3. Measures increase in target class score
    4. Uses scores as weights for activation maps

    Advantages:
    - No gradients = more stable
    - Less noisy than Grad-CAM
    - Better for multiple instances
    - Better for small objects
    """

    def __init__(self, model, target_layer, batch_size=32):
        """
        Args:
            model: Trained segmentation model
            target_layer: Layer to extract activations from
            batch_size: Batch size for processing activation masks
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.batch_size = batch_size

        self.activations = None

        # Register forward hook
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        self.hook = target_layer.register_forward_hook(forward_hook)

    def __del__(self):
        """Clean up hook"""
        if hasattr(self, 'hook'):
            self.hook.remove()

    def generate(self, input_tensor, class_idx, num_samples=None):
        """
        Generate Score-CAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) input image
            class_idx: Target class index
            num_samples: Max number of activation maps to use (None = all)

        Returns:
            cam: (H, W) numpy array in [0, 1]
        """
        batch_size, C, H, W = input_tensor.shape
        device = input_tensor.device

        # Step 1: Get base prediction and activation maps
        with torch.no_grad():
            logits = self.model(input_tensor)  # (1, num_classes, H, W)
            activations = self.activations     # (1, K, H', W')

        # Base score for target class (sum over spatial dims)
        base_score = logits[0, class_idx].sum().item()

        B, K, H_act, W_act = activations.shape

        # Optionally limit number of activation maps for speed
        if num_samples is not None and K > num_samples:
            # Sample random subset
            indices = torch.randperm(K)[:num_samples]
            activations = activations[:, indices, :, :]
            K = num_samples

        # Step 2: Upsample activation maps to input size
        activations_upsampled = F.interpolate(
            activations,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (1, K, H, W)

        # Normalize each activation map to [0, 1]
        activations_norm = activations_upsampled.squeeze(0)  # (K, H, W)
        for k in range(K):
            act_min = activations_norm[k].min()
            act_max = activations_norm[k].max()
            if act_max - act_min > 1e-7:
                activations_norm[k] = (activations_norm[k] - act_min) / (act_max - act_min)
            else:
                activations_norm[k] = 0

        # Step 3: Mask input with each activation and get scores
        scores = []

        # Process in batches for efficiency
        for i in range(0, K, self.batch_size):
            batch_end = min(i + self.batch_size, K)
            batch_acts = activations_norm[i:batch_end]  # (batch, H, W)
            batch_size_actual = batch_end - i

            # Create masked inputs: input * activation_map
            # Expand input: (1, 3, H, W) -> (batch, 3, H, W)
            input_batch = input_tensor.expand(batch_size_actual, -1, -1, -1)

            # Expand activations: (batch, H, W) -> (batch, 1, H, W) -> (batch, 3, H, W)
            masks = batch_acts.unsqueeze(1).expand(-1, 3, -1, -1)

            # Masked input
            masked_input = input_batch * masks

            # Forward pass
            with torch.no_grad():
                masked_logits = self.model(masked_input)  # (batch, num_classes, H, W)

            # Get scores for target class
            batch_scores = masked_logits[:, class_idx].sum(dim=(1, 2))  # (batch,)
            scores.append(batch_scores.cpu())

        # Concatenate all scores
        scores = torch.cat(scores)  # (K,)

        # Step 4: Normalize scores to [0, 1]
        scores = F.relu(scores)  # Only positive contributions
        if scores.max() > 1e-7:
            scores = scores / scores.max()

        # Step 5: Weighted combination of activation maps
        weights = scores.to(device).unsqueeze(1).unsqueeze(2)  # (K, 1, 1)
        activations_norm = activations_norm.to(device)  # (K, H, W)

        cam = (weights * activations_norm).sum(dim=0)  # (H, W)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-7:
            cam /= cam.max()

        return cam


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


def visualize_scorecam(img_np, cam, class_name, alpha=0.5):
    """Create 3-panel Score-CAM visualization."""
    cam_overlay = overlay_cam_on_image(img_np, cam, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f"Score-CAM: {class_name}")
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
    output_dir: str = 'scorecam_visualizations',
    batch_size: int = 32,
    max_activation_maps: int = None
):
    """
    Load pretrained model and generate Score-CAM visualizations.

    Args:
        model_path: Path to pretrained model (.pth file)
        num_samples: Number of samples to visualize
        split: Dataset split ('test', 'val', or 'train')
        encoder: Encoder architecture
        encoder_weights: Encoder weights
        selected_classes: List of class names to visualize
        img_size: Image size (width, height)
        output_dir: Output directory for visualizations
        batch_size: Batch size for Score-CAM processing
        max_activation_maps: Max activation maps to use (None = all)
    """
    if selected_classes is None:
        selected_classes = CLASSES

    print(f"{'='*80}")
    print(f"Score-CAM Visualization (Gradient-Free Explainability)")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Model path: {model_path}")
    print(f"Dataset split: {split}")
    print(f"Number of samples: {num_samples}")
    print(f"Classes: {selected_classes}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
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

    num_samples = min(num_samples, len(dataset))

    # Create output directories
    tpfpfn_dir = os.path.join(output_dir, "tpfpfn_maps")
    scorecam_dir = os.path.join(output_dir, "scorecam_maps")
    os.makedirs(tpfpfn_dir, exist_ok=True)
    os.makedirs(scorecam_dir, exist_ok=True)

    # Setup Score-CAM
    print("Setting up Score-CAM...")
    target_layer = model.encoder.layer4
    scorecam = ScoreCAM(model, target_layer, batch_size=batch_size)
    print("✓ Score-CAM ready\n")

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
            preds_idx = torch.argmax(logits, dim=1)[0]
            masks_idx = torch.argmax(mask_tensor, dim=1)[0]

            # Unnormalize image
            img_unnorm = unnormalize(image_tensor[0]).cpu()

            # ================================================================
            # GENERATE TP/FP/FN MAPS
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
            # GENERATE SCORE-CAM FOR EACH CLASS
            # ================================================================
            print(f"  Generating Score-CAM maps...")
            img_np = tensor_to_uint8(image_tensor[0].cpu())

            for cls_idx, cls_name in enumerate(selected_classes):
                print(f"    - {cls_name}...")

                # Generate Score-CAM (no gradients needed!)
                cam = scorecam.generate(
                    image_tensor,
                    cls_idx,
                    num_samples=max_activation_maps
                )

                # Visualize
                fig = visualize_scorecam(img_np, cam, cls_name)

                # Save
                save_path = os.path.join(
                    scorecam_dir,
                    f"sample{sample_idx}_scorecam_{cls_name}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            print(f"  ✓ Sample {sample_idx} complete\n")

    # Cleanup
    del scorecam

    print(f"{'='*80}")
    print(f"Score-CAM Visualization Complete!")
    print(f"{'='*80}")
    print(f"TP/FP/FN maps saved to: {tpfpfn_dir}/")
    print(f"Score-CAM maps saved to: {scorecam_dir}/")
    print(f"{'='*80}\n")
    print("Note: Score-CAM uses NO GRADIENTS - more stable than Grad-CAM!")
    print(f"{'='*80}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate Score-CAM visualizations (gradient-free explainability)'
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
        default='scorecam_visualizations',
        help='Output directory (default: scorecam_visualizations)'
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
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for Score-CAM processing (default: 32)'
    )
    parser.add_argument(
        '--max_activation_maps',
        type=int,
        default=None,
        help='Max activation maps to use (default: None = all)'
    )

    args = parser.parse_args()

    visualize_pretrained_model(
        model_path=args.model_path,
        num_samples=args.num_samples,
        split=args.split,
        encoder=args.encoder,
        img_size=tuple(args.img_size),
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_activation_maps=args.max_activation_maps
    )


if __name__ == "__main__":
    main()
