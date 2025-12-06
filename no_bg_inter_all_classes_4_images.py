# -*- coding: utf-8 -*-
"""Koushik of running model acc for each class - IMPROVED.ipynb"""

# !pip install -q segmentation-models-pytorch>=0.3.4 torchmetrics==1.4.0

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
# Fix the imports - use the correct path for your version
from segmentation_models_pytorch.utils import losses
from segmentation_models_pytorch.utils import metrics as smp_metrics  # For metric classes
from segmentation_models_pytorch.utils import train

import numpy as np
import matplotlib.pyplot as plt
import os

def show_tp_fp_fn_binary(image, gt_binary, pred_binary,
                         class_name="class", alpha=0.5, save_path=None):
    """
    image      : (C, H, W) torch tensor or (H, W, C) numpy
    gt_binary  : (H, W) bool/0-1 array (GT for ONE class)
    pred_binary: (H, W) bool/0-1 array (prediction for ONE class)

    Creates a 4-panel figure:
      1) Input image
      2) GT mask for this class
      3) Predicted mask for this class
      4) TP/FP/FN overlay (G=TP, R=FP, B=FN)
    """

    # ---- convert image to numpy in [0,1] for display ----
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

    # ---- 4-panel figure ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1) input
    axes[0].imshow(img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # 2) GT mask
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title(f"GT: {class_name}")
    axes[1].axis("off")

    # 3) predicted mask
    axes[2].imshow(pr, cmap="gray")
    axes[2].set_title(f"Pred: {class_name}")
    axes[2].axis("off")

    # 4) TP/FP/FN overlay
    axes[3].imshow(blended)
    axes[3].set_title(f"TP/FP/FN for {class_name}\n(G=TP, R=FP, B=FN)")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)  # important on cluster to free memory



DATA_ROOT = Path("phenocyte_seg")
IMG_DIR = DATA_ROOT / "images"
MSK_DIR = DATA_ROOT / "masks"
SPLIT_CSV = DATA_ROOT / "dataset_split.csv"

CLASSES = ["background", "root", "hypocotyl", "cotyledon", "seed"]
NUM_CLASSES = len(CLASSES)

def _load_mask_ids(path: Path) -> np.ndarray:
    """
    Load mask and return HxW integer array with class IDs.
    Works with single-channel ID masks or RGB palette masks.
    """
    arr = np.array(Image.open(path))

    if arr.ndim == 2:  # Already class IDs
        return arr.astype(np.int64)

    if arr.ndim == 3 and arr.shape[2] in (3, 4):  # RGB/RGBA -> map to IDs
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        h, w, _ = arr.shape
        # Map unique RGB colors to sequential class IDs
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
        # Load split from CSV
        df = pd.read_csv(SPLIT_CSV)
        self.items = df[df["set"] == split]["img_name"].tolist()
        self.items = [name.replace('%', '_') for name in self.items]
        self.mask_suffix = mask_suffix
        self.split = split  # Store split name

        # Setup class filtering (like CamVid)
        if classes is None:
            classes = self.CLASSES

        # Convert class names to indices
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        # Load image
        name = self.items[i]
        image_path = IMG_DIR / name
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        base = Path(name).stem
        msk_name = f"{base}{self.mask_suffix}.png" if self.mask_suffix else f"{base}.png"
        mask = _load_mask_ids(MSK_DIR / msk_name)

        # Extract specific classes (like CamVid does for 'car')
        # Create binary masks for each selected class
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __repr__(self):
        return f"PhenocyteDataset(split={self.split}, classes={self.class_values}, n_samples={len(self)})"


def get_training_augmentation(img_size=(512, 256)):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(
            scale_limit=0.3,
            rotate_limit=30,
            shift_limit=0.15,
            p=0.7,
            border_mode=0
        ),

        # Add elastic transform for organic shapes
        albu.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),

        # Pad and crop to target size
        albu.PadIfNeeded(min_height=img_size[1], min_width=img_size[0], border_mode=0),
        albu.RandomCrop(height=img_size[1], width=img_size[0]),

        # Color augmentations
        albu.OneOf([
            albu.CLAHE(p=1),
            albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            albu.RandomGamma(p=1),
        ], p=0.7),

        albu.OneOf([
            albu.Sharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
            albu.GaussNoise(p=1),
        ], p=0.5),

        albu.OneOf([
            albu.HueSaturationValue(p=1),
        ], p=0.5),
    ]
    return albu.Compose(train_transform)


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


# ============================================================================
# NEW: COMPUTE CLASS WEIGHTS FOR HANDLING IMBALANCE
# ============================================================================
def compute_class_weights(dataset, num_classes, device='cpu'):
    """
    Calculate inverse frequency weights for imbalanced classes.
    
    Args:
        dataset: PyTorch dataset that returns (image, mask) tuples
        num_classes: Number of classes
        device: Device to place weights on
        
    Returns:
        torch.Tensor: Normalized class weights
    """
    print("\nComputing class weights from training data...")
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    
    # Count pixels for each class
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        
        # mask is [C, H, W] after preprocessing
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        for c in range(num_classes):
            class_pixel_counts[c] += mask[c].sum()
    
    print(f"Class pixel counts: {class_pixel_counts.numpy()}")
    
    # Compute inverse frequency weights
    # Add small epsilon to avoid division by zero
    total_pixels = class_pixel_counts.sum()
    class_weights = total_pixels / (num_classes * (class_pixel_counts + 1e-6))
    
    # Normalize weights so they sum to num_classes
    class_weights = class_weights / class_weights.sum() * num_classes
    
    # Optional: Cap maximum weight to prevent extreme values
    max_weight = 10.0
    class_weights = torch.clamp(class_weights, max=max_weight)
    
    print(f"Computed class weights: {class_weights.numpy()}")
    
    return class_weights.to(device)


# ============================================================================
# NEW: COMBINED LOSS FUNCTION
# ============================================================================
class CombinedLoss(nn.Module):
    """
    Combines Dice Loss and Focal Loss for better optimization.
    
    Dice Loss: Good for segmentation, handles class imbalance
    Focal Loss: Focuses on hard examples, reduces weight of easy examples
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, class_weights=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Initialize losses with class weights if provided
        if class_weights is not None:
            # For multilabel, we need to pass weights differently
            self.dice = DiceLoss(mode='multilabel', from_logits=False)
            self.focal = FocalLoss(mode='multilabel')
            self.class_weights = class_weights
            self.use_weights = True
        else:
            self.dice = DiceLoss(mode='multilabel', from_logits=False)
            self.focal = FocalLoss(mode='multilabel')
            self.use_weights = False
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
        """
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        
        # Weighted combination
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        
        # Apply class weights by computing per-class loss
        if self.use_weights:
            # Compute per-class losses and apply weights
            per_class_dice = self._per_class_dice_loss(pred, target)
            per_class_focal = self._per_class_focal_loss(pred, target)
            
            weighted_dice = (per_class_dice * self.class_weights).mean()
            weighted_focal = (per_class_focal * self.class_weights).mean()
            
            total_loss = self.dice_weight * weighted_dice + self.focal_weight * weighted_focal
        
        return total_loss
    
    def _per_class_dice_loss(self, pred, target):
        """Compute Dice loss per class"""
        smooth = 1e-5
        num_classes = pred.shape[1]
        losses = []
        
        for c in range(num_classes):
            pred_c = pred[:, c, :, :]
            target_c = target[:, c, :, :]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2. * intersection + smooth) / (union + smooth)
            losses.append(1 - dice)
        
        return torch.stack(losses)
    
    def _per_class_focal_loss(self, pred, target):
        """Compute Focal loss per class"""
        num_classes = pred.shape[1]
        losses = []
        
        for c in range(num_classes):
            pred_c = pred[:, c, :, :].flatten()
            target_c = target[:, c, :, :].flatten()
            
            # Focal loss components
            bce = -(target_c * torch.log(pred_c + 1e-7) + (1 - target_c) * torch.log(1 - pred_c + 1e-7))
            pt = torch.where(target_c == 1, pred_c, 1 - pred_c)
            focal = ((1 - pt) ** 2) * bce
            
            losses.append(focal.mean())
        
        return torch.stack(losses)


# ============================================================================
# NEW: FUNCTION TO COMPUTE PER-CLASS IOU
# ============================================================================
def compute_per_class_iou(model, dataloader, device, num_classes, class_names):
    """
    Compute per-class IoU on a given dataloader and optionally save TP/FP/FN maps.

    Args:
        model: Trained model
        dataloader: DataLoader to evaluate on
        device: Device to run on
        num_classes: Number of classes
        class_names: List of class names (length = num_classes)

    Returns:
        dict: Dictionary with per-class and mean IoU scores
    """
    model.eval()

    # Accumulators for TP, FP, FN, TN per class
    tp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=device)
    tn = torch.zeros(num_classes, dtype=torch.float64, device=device)

    # ---- visualization settings ----
    viz_count = 0
    max_viz_samples = 5   # how many images to visualize
    tpfpfn_dir = "tpfpfn_maps_all_class_4_img"
    os.makedirs(tpfpfn_dir, exist_ok=True)

    # unnormalize helper (ImageNet stats)
    def unnormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        return img_tensor * std + mean

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)   # [B, C, H, W]
            masks = masks.to(device)     # [B, C, H, W] one-hot

            # Forward pass
            logits = model(images)       # [B, C, H, W]

            # Apply threshold for multilabel
            # preds = (logits > 0.5).float()

            # Convert to class indices for metrics
            # preds_idx = torch.argmax(preds, dim=1)  # [B, H, W]
            preds_idx = torch.argmax(logits, dim=1)   # [B, H, W]
            masks_idx = torch.argmax(masks, dim=1)  # [B, H, W]

            # -------- TP/FP/FN VISUALIZATION BLOCK --------
            if viz_count < max_viz_samples:
                remaining = max_viz_samples - viz_count
                batch_size = images.size(0)
                num_to_take = min(batch_size, remaining)

                for b in range(num_to_take):
                    # unnormalize and move to CPU for plotting
                    img_b = unnormalize(images[b]).cpu()
                    gt_idx = masks_idx[b].cpu().numpy()
                    pr_idx = preds_idx[b].cpu().numpy()

                    sample_id = viz_count

                    # generate a map for EACH class
                    for cls_idx, cls_name in enumerate(class_names):
                        gt_binary = (gt_idx == cls_idx)
                        pred_binary = (pr_idx == cls_idx)

                        save_path = os.path.join(
                            tpfpfn_dir,
                            f"sample{sample_id}_cls_{cls_name}.png"
                        )

                        show_tp_fp_fn_binary(
                            image=img_b,
                            gt_binary=gt_binary,
                            pred_binary=pred_binary,
                            class_name=cls_name,
                            alpha=0.5,
                            save_path=save_path,
                        )

                    viz_count += 1
                    if viz_count >= max_viz_samples:
                        break
            # -------- END VISUALIZATION BLOCK --------

            # Batch-wise stats
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                preds_idx,
                masks_idx,
                mode="multiclass",
                num_classes=num_classes,
            )
            tp += batch_tp.sum(dim=0).to(device)
            fp += batch_fp.sum(dim=0).to(device)
            fn += batch_fn.sum(dim=0).to(device)
            tn += batch_tn.sum(dim=0).to(device)

    # Compute per-class IoU
    per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")

    results = {
        "per_class_iou": per_class_iou.cpu().numpy(),
        "mean_iou": per_class_iou.mean().item(),
    }

    # Add individual class scores
    for idx, cls_name in enumerate(class_names):
        results[f"iou_{cls_name}"] = per_class_iou[idx].item()

    return results



# ============================================================================
# MAIN TRAINING CODE
# ============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    DATA_ROOT = Path("phenocyte_seg")
    SPLIT_CSV = 'phenocyte_seg/dataset_split.csv'
    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # EXCLUDE BACKGROUND - only segment the plant parts we care about
    SELECTED_CLASSES = CLASSES
    ACTIVATION = 'sigmoid'
    IMG_SIZE = (512, 256)
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50
    LR_DECAY_EPOCH = 25
    
    print(f"Device: {DEVICE}")
    print(f"Creating model for classes: {SELECTED_CLASSES}")
    print(f"Number of output channels: {len(SELECTED_CLASSES)}")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    num_classes = len(SELECTED_CLASSES)

    # Create datasets
    train_dataset = PhenocyteDataset(
        split='train',
        classes=SELECTED_CLASSES,
        augmentation=get_training_augmentation(IMG_SIZE),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = PhenocyteDataset(
        split='val',
        classes=SELECTED_CLASSES,
        augmentation=get_validation_augmentation(IMG_SIZE),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_dataset = PhenocyteDataset(
        split='test',
        classes=SELECTED_CLASSES,
        augmentation=get_validation_augmentation(IMG_SIZE),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # ========================================================================
    # COMPUTE CLASS WEIGHTS
    # ========================================================================
    class_weights = compute_class_weights(train_dataset, num_classes, device=DEVICE)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ========================================================================
    # CREATE COMBINED LOSS WITH CLASS WEIGHTS
    # ========================================================================
    loss = CombinedLoss(
        dice_weight=0.5,
        focal_weight=0.5,
        class_weights=class_weights
    )
    loss.__name__ = "CombinedLoss_Weighted"
    
    print(f"\nUsing loss: {loss.__name__}")
    print(f"  - Dice weight: 0.5")
    print(f"  - Focal weight: 0.5")
    print(f"  - Class weights applied: Yes")

    # Fix metrics - use renamed module to avoid conflict
    metric_list = [
        smp_metrics.IoU(threshold=0.5),
        smp_metrics.Fscore(threshold=0.5),
    ]

    # ========================================================================
    # TRAIN MODEL
    # ========================================================================
    models = [
        smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(SELECTED_CLASSES),
            activation=ACTIVATION
        )
    ]
    
    report = ''
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"Training {model.name}")
        print(f"{'='*80}")
        
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=LEARNING_RATE),
        ])

        # Use the train module imported from segmentation_models_pytorch.utils
        train_epoch = train.TrainEpoch(
            model,
            loss=loss,
            metrics=metric_list,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = train.ValidEpoch(
            model,
            loss=loss,
            metrics=metric_list,
            device=DEVICE,
            verbose=True,
        )

        max_score = 0
        best_epoch = 0
        
        # Track training history
        train_iou_history = []
        valid_iou_history = []
        
        # Track per-class IoU history
        train_per_class_history = {cls: [] for cls in SELECTED_CLASSES}
        valid_per_class_history = {cls: [] for cls in SELECTED_CLASSES}

        for i in range(NUM_EPOCHS):
            print(f'\n{"="*80}')
            print(f'Epoch: {i}/{NUM_EPOCHS-1}')
            print(f'{"="*80}')

            # Run standard training/validation
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            
            # Compute per-class IoU for this epoch
            print(f"\n{'-'*80}")
            print("Computing per-class IoU on validation set...")
            print(f"{'-'*80}")
            
            valid_per_class = compute_per_class_iou(
                model, valid_loader, DEVICE, num_classes, SELECTED_CLASSES
            )
            
            # Print per-class results
            print(f"\n{'Class':<15} {'IoU':>8}")
            print(f"{'-'*25}")
            for cls_name in SELECTED_CLASSES:
                iou_val = valid_per_class[f'iou_{cls_name}']
                print(f"{cls_name:<15} {iou_val:>8.4f}")
                valid_per_class_history[cls_name].append(iou_val)
            
            print(f"{'-'*25}")
            print(f"{'Mean IoU':<15} {valid_per_class['mean_iou']:>8.4f}")
            print(f"{'Overall IoU':<15} {valid_logs['iou_score']:>8.4f}")
            print(f"{'-'*25}\n")
            
            train_iou_history.append(train_logs['iou_score'])
            valid_iou_history.append(valid_logs['iou_score'])

            # Save best model based on overall IoU
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                best_epoch = i
                model_saved_at = f'best_model_{model.name}.pth'
                torch.save(model, model_saved_at)
                print(f'✓ Model saved! (Best IoU: {max_score:.4f})')

            # Learning rate decay
            if i == LR_DECAY_EPOCH:
                optimizer.param_groups[0]['lr'] = LEARNING_RATE / 10
                print(f'\n→ Decreased learning rate to {optimizer.param_groups[0]["lr"]}!')

        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"{'='*80}")
        best_result = f"Best IoU score: {max_score:.4f} at epoch {best_epoch}"
        print(best_result)

        # ====================================================================
        # EVALUATE ON TEST SET WITH PER-CLASS METRICS
        # ====================================================================
        print(f"\n{'='*80}")
        print("Evaluating on test set...")
        print(f"{'='*80}")
        
        best_model = torch.load(model_saved_at, weights_only=False, map_location=DEVICE)
        best_model.eval()
        
        # Accumulators for TP, FP, FN, TN per class
        tp = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
        fp = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
        fn = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
        tn = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                # Forward pass
                logits = best_model(images)  # [B, C, H, W]

                # Apply threshold for multilabel
                preds = (logits > 0.5).float()

                # Convert to class indices for metrics
                preds_idx = torch.argmax(preds, dim=1)  # [B, H, W]
                masks_idx = torch.argmax(masks, dim=1)  # [B, H, W]

                # Get batch-wise stats - use smp.metrics
                batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                    preds_idx,
                    masks_idx,
                    mode="multiclass",
                    num_classes=num_classes,
                )
                tp += batch_tp.sum(dim=0).to(DEVICE)
                fp += batch_fp.sum(dim=0).to(DEVICE)
                fn += batch_fn.sum(dim=0).to(DEVICE)
                tn += batch_tn.sum(dim=0).to(DEVICE)

        # Compute per-class metrics - use smp.metrics
        per_class_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        per_class_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        per_class_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="none")
        per_class_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="none")

        # Generate report
        model_name = f"\n{'='*80}\n"
        model_name += f"Per-class Metrics: {model.name} model\n"
        model_name += f"{best_result}\n"
        model_name += f"Loss: {loss.__name__}\n"
        model_name += f"{'='*80}\n\n"
        report += model_name
        print(model_name)

        # Print per-class results
        print(f"{'Class':<15} {'IoU':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        print(f"{'-'*60}")
        
        for idx, cls_name in enumerate(SELECTED_CLASSES):
            result = (f"{cls_name:<15} "
                     f"{per_class_iou[idx].item():>8.4f} "
                     f"{per_class_f1[idx].item():>8.4f} "
                     f"{per_class_precision[idx].item():>10.4f} "
                     f"{per_class_recall[idx].item():>8.4f}\n")
            print(result, end='')
            report += result

        # FIXED: Compute mean over all classes (background already excluded)
        mean_iou = per_class_iou.mean().item()  # All classes
        mean_f1 = per_class_f1.mean().item()
        
        summary = f"\n{'-'*60}\n"
        summary += f"Mean IoU (all plant parts): {mean_iou:.4f}\n"
        summary += f"Mean F1 (all plant parts): {mean_f1:.4f}\n"
        summary += f"{'-'*60}\n"
        print(summary)
        report += summary

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    with open('fpn-se_resnext50_training_result_improved.txt', 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to: fpn-se_resnext50_training_result_improved.txt")
    
    # ========================================================================
    # PLOT TRAINING CURVES - Overall and Per-Class
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: Overall IoU
    axes[0].plot(train_iou_history, label='Train IoU', linewidth=2)
    axes[0].plot(valid_iou_history, label='Validation IoU', linewidth=2)
    axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('IoU Score', fontsize=12)
    axes[0].set_title('Overall IoU Training Progress', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Per-Class Validation IoU
    for cls_name in SELECTED_CLASSES:
        axes[1].plot(valid_per_class_history[cls_name], label=cls_name, linewidth=2, marker='o', markersize=3)
    axes[1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('IoU Score', fontsize=12)
    axes[1].set_title('Per-Class Validation IoU', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to: training_curves.png")
    
    print(f"\n{'='*80}")
    print("All training and evaluation complete!")
    print(f"{'='*80}")