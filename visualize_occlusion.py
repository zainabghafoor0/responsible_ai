#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Occlusion Sensitivity Visualization for Pretrained Segmentation Model

Occlusion Sensitivity is the most interpretable explainability method.
It directly measures which regions matter by systematically masking them.

Key advantages:
- Most interpretable (direct causation)
- No gradients needed (always works)
- Model-agnostic (works with any model)
- Works when Grad-CAM/Score-CAM fail

How it works:
1. Divide image into grid of patches
2. Mask each patch (set to mean or zero)
3. Measure prediction drop
4. High drop = important region

Usage:
    python visualize_occlusion.py --model_path best_model_FPN.pth \
                                   --num_samples 5 \
                                   --patch_size 32
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


def overlay_map_on_image(img_np, heatmap, alpha=0.5):
    """Overlay heatmap on image."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


# ============================================================================
# OCCLUSION SENSITIVITY IMPLEMENTATION
# ============================================================================
class OcclusionSensitivity:
    """
    Occlusion Sensitivity for Semantic Segmentation

    Most interpretable explainability method - directly measures
    which regions matter by systematically masking them.

    How it works:
    1. Divide image into grid of patches
    2. For each patch:
       - Mask it (replace with mean or zero)
       - Run forward pass
       - Measure prediction drop for target class
    3. High drop = important region

    Advantages:
    - Direct causation (not correlation)
    - No gradients needed
    - Model-agnostic
    - Always works
    - Most interpretable
    """

    def __init__(self, model, patch_size=32, stride=None, occlusion_value='mean'):
        """
        Args:
            model: Trained segmentation model
            patch_size: Size of occlusion patch
            stride: Stride for sliding window (default: patch_size // 2)
            occlusion_value: 'mean' (replace with mean) or 'zero' (black)
        """
        self.model = model
        self.model.eval()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size // 2
        self.occlusion_value = occlusion_value

    def generate(self, input_tensor, class_idx, verbose=True):
        """
        Generate occlusion sensitivity map.

        Args:
            input_tensor: (1, 3, H, W) input image
            class_idx: Target class index
            verbose: Show progress bar

        Returns:
            sensitivity_map: (H, W) numpy array in [0, 1]
        """
        device = input_tensor.device
        _, C, H, W = input_tensor.shape

        # Get baseline prediction (no occlusion)
        with torch.no_grad():
            baseline_logits = self.model(input_tensor)
            baseline_score = baseline_logits[0, class_idx].sum().item()

        # Prepare occlusion value
        if self.occlusion_value == 'mean':
            # Use mean of input image
            fill_value = input_tensor.mean(dim=(2, 3), keepdim=True)  # (1, 3, 1, 1)
        else:
            # Use zero (black)
            fill_value = torch.zeros((1, C, 1, 1), device=device)

        # Create sensitivity map
        sensitivity_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        # Generate grid of patch positions
        y_positions = list(range(0, H - self.patch_size + 1, self.stride))
        x_positions = list(range(0, W - self.patch_size + 1, self.stride))

        # Add final positions if not covered
        if y_positions[-1] + self.patch_size < H:
            y_positions.append(H - self.patch_size)
        if x_positions[-1] + self.patch_size < W:
            x_positions.append(W - self.patch_size)

        total_patches = len(y_positions) * len(x_positions)

        # Iterate over patches
        iterator = tqdm(total=total_patches, desc=f"  Occluding patches") if verbose else None

        for y in y_positions:
            for x in x_positions:
                # Create occluded image
                occluded = input_tensor.clone()

                # Replace patch with occlusion value
                y_end = min(y + self.patch_size, H)
                x_end = min(x + self.patch_size, W)
                occluded[:, :, y:y_end, x:x_end] = fill_value

                # Forward pass
                with torch.no_grad():
                    occluded_logits = self.model(occluded)
                    occluded_score = occluded_logits[0, class_idx].sum().item()

                # Compute sensitivity (prediction drop)
                sensitivity = baseline_score - occluded_score

                # Accumulate in map
                sensitivity_map[y:y_end, x:x_end] += sensitivity
                count_map[y:y_end, x:x_end] += 1

                if iterator:
                    iterator.update(1)

        if iterator:
            iterator.close()

        # Average overlapping regions
        sensitivity_map = np.divide(
            sensitivity_map,
            count_map,
            out=np.zeros_like(sensitivity_map),
            where=count_map > 0
        )

        # Normalize to [0, 1]
        sensitivity_map = np.maximum(sensitivity_map, 0)  # Only positive sensitivity
        if sensitivity_map.max() > 1e-7:
            sensitivity_map = sensitivity_map / sensitivity_map.max()

        return sensitivity_map


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


def visualize_occlusion(img_np, sensitivity_map, class_name, alpha=0.5):
    """Create 3-panel Occlusion Sensitivity visualization."""
    overlay = overlay_map_on_image(img_np, sensitivity_map, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(sensitivity_map, cmap='jet')
    axes[1].set_title(f"Occlusion Sensitivity: {class_name}")
    axes[1].axis('off')

    axes[2].imshow(overlay)
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
    output_dir: str = 'occlusion_visualizations',
    patch_size: int = 32,
    stride: int = None,
    occlusion_value: str = 'mean'
):
    """
    Load pretrained model and generate Occlusion Sensitivity visualizations.

    Args:
        model_path: Path to pretrained model (.pth file)
        num_samples: Number of samples to visualize
        split: Dataset split ('test', 'val', or 'train')
        encoder: Encoder architecture
        encoder_weights: Encoder weights
        selected_classes: List of class names to visualize
        img_size: Image size (width, height)
        output_dir: Output directory for visualizations
        patch_size: Size of occlusion patch
        stride: Stride for sliding window
        occlusion_value: 'mean' or 'zero'
    """
    if selected_classes is None:
        selected_classes = CLASSES

    print(f"{'='*80}")
    print(f"Occlusion Sensitivity (Most Interpretable Explainability)")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Model path: {model_path}")
    print(f"Dataset split: {split}")
    print(f"Number of samples: {num_samples}")
    print(f"Classes: {selected_classes}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Stride: {stride if stride else patch_size // 2}")
    print(f"Occlusion value: {occlusion_value}")
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

    num_samples = min(num_samples, len(dataset))

    # Create output directories
    tpfpfn_dir = os.path.join(output_dir, "tpfpfn_maps")
    occlusion_dir = os.path.join(output_dir, "occlusion_maps")
    os.makedirs(tpfpfn_dir, exist_ok=True)
    os.makedirs(occlusion_dir, exist_ok=True)

    # Setup Occlusion Sensitivity
    print("Setting up Occlusion Sensitivity...")
    occlusion = OcclusionSensitivity(
        model,
        patch_size=patch_size,
        stride=stride,
        occlusion_value=occlusion_value
    )
    print("✓ Occlusion Sensitivity ready\n")

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
            # GENERATE OCCLUSION SENSITIVITY FOR EACH CLASS
            # ================================================================
            print(f"  Generating Occlusion Sensitivity maps...")
            img_np = tensor_to_uint8(image_tensor[0].cpu())

            for cls_idx, cls_name in enumerate(selected_classes):
                print(f"    - {cls_name}...")

                # Generate Occlusion Sensitivity
                sensitivity_map = occlusion.generate(
                    image_tensor,
                    cls_idx,
                    verbose=True
                )

                # Visualize
                fig = visualize_occlusion(img_np, sensitivity_map, cls_name)

                # Save
                save_path = os.path.join(
                    occlusion_dir,
                    f"sample{sample_idx}_occlusion_{cls_name}.png"
                )
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            print(f"  ✓ Sample {sample_idx} complete\n")

    print(f"{'='*80}")
    print(f"Occlusion Sensitivity Visualization Complete!")
    print(f"{'='*80}")
    print(f"TP/FP/FN maps saved to: {tpfpfn_dir}/")
    print(f"Occlusion maps saved to: {occlusion_dir}/")
    print(f"{'='*80}\n")
    print("Note: Occlusion Sensitivity = Direct Causation (Most Interpretable!)")
    print(f"{'='*80}\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate Occlusion Sensitivity visualizations (most interpretable!)'
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
        default='occlusion_visualizations',
        help='Output directory (default: occlusion_visualizations)'
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
        '--patch_size',
        type=int,
        default=32,
        help='Occlusion patch size (default: 32)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=None,
        help='Stride for sliding window (default: patch_size // 2)'
    )
    parser.add_argument(
        '--occlusion_value',
        type=str,
        default='mean',
        choices=['mean', 'zero'],
        help='Occlusion fill value: mean or zero (default: mean)'
    )

    args = parser.parse_args()

    visualize_pretrained_model(
        model_path=args.model_path,
        num_samples=args.num_samples,
        split=args.split,
        encoder=args.encoder,
        img_size=tuple(args.img_size),
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        occlusion_value=args.occlusion_value
    )


if __name__ == "__main__":
    main()
