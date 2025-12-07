#!/usr/bin/env python3
"""
Create an improved TP/FP/FN Maps slide with confusion matrix diagram
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os


def create_confusion_matrix_diagram(ax):
    """
    Create a visual confusion matrix diagram showing TP/FP/FN/TN regions

    Args:
        ax: Matplotlib axis to draw on
    """
    # Define colors
    colors = {
        'TP': '#90EE90',  # Light green
        'FP': '#FFB6C1',  # Light red/pink
        'FN': '#ADD8E6',  # Light blue
        'TN': '#D3D3D3',  # Light gray
    }

    # Create the 2x2 confusion matrix grid
    cell_size = 1

    # Draw cells
    # Top-left: TP (Predicted Positive, True Class Positive)
    tp_rect = mpatches.Rectangle((0, cell_size), cell_size, cell_size,
                                  facecolor=colors['TP'], edgecolor='black', linewidth=2)
    ax.add_patch(tp_rect)
    ax.text(0.5, 1.5, 'TP', ha='center', va='center', fontsize=24, fontweight='bold')

    # Top-right: FP (Predicted Positive, True Class Negative)
    fp_rect = mpatches.Rectangle((cell_size, cell_size), cell_size, cell_size,
                                  facecolor=colors['FP'], edgecolor='black', linewidth=2)
    ax.add_patch(fp_rect)
    ax.text(1.5, 1.5, 'FP', ha='center', va='center', fontsize=24, fontweight='bold')

    # Bottom-left: FN (Predicted Negative, True Class Positive)
    fn_rect = mpatches.Rectangle((0, 0), cell_size, cell_size,
                                  facecolor=colors['FN'], edgecolor='black', linewidth=2)
    ax.add_patch(fn_rect)
    ax.text(0.5, 0.5, 'FN', ha='center', va='center', fontsize=24, fontweight='bold')

    # Bottom-right: TN (Predicted Negative, True Class Negative)
    tn_rect = mpatches.Rectangle((cell_size, 0), cell_size, cell_size,
                                  facecolor=colors['TN'], edgecolor='black', linewidth=2)
    ax.add_patch(tn_rect)
    ax.text(1.5, 0.5, 'TN', ha='center', va='center', fontsize=24, fontweight='bold')

    # Add labels
    # True Class (vertical)
    ax.text(-0.3, 1.5, 'Positive', rotation=90, ha='center', va='center',
            fontsize=12, color='#FF6B6B', fontweight='bold')
    ax.text(-0.3, 0.5, 'Negative', rotation=90, ha='center', va='center',
            fontsize=12, color='#4ECDC4', fontweight='bold')
    ax.text(-0.5, 1.0, 'True Class', rotation=90, ha='center', va='center',
            fontsize=14, fontweight='bold')

    # Predicted Class (horizontal)
    ax.text(0.5, 2.3, 'Positive', ha='center', va='center',
            fontsize=12, color='#FF6B6B', fontweight='bold')
    ax.text(1.5, 2.3, 'Negative', ha='center', va='center',
            fontsize=12, color='#4ECDC4', fontweight='bold')
    ax.text(1.0, 2.55, 'Predicted Class', ha='center', va='center',
            fontsize=14, fontweight='bold')

    # Set axis properties
    ax.set_xlim(-0.7, 2.2)
    ax.set_ylim(-0.2, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)


def create_tpfpfn_slide(image, gt_binary, pred_binary, class_name="class",
                        alpha=0.5, save_path=None, metrics=None):
    """
    Create a comprehensive TP/FP/FN slide with confusion matrix diagram

    Args:
        image: (H, W, 3) numpy array or (3, H, W) torch tensor
        gt_binary: (H, W) bool/0-1 array (GT for ONE class)
        pred_binary: (H, W) bool/0-1 array (prediction for ONE class)
        class_name: Name of the class being visualized
        alpha: Transparency for overlay blending
        save_path: Path to save the figure
        metrics: Optional dict with IoU, F1, Precision, Recall

    Returns:
        fig: Matplotlib figure object
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

    # Convert to boolean
    gt = gt_binary.astype(bool)
    pr = pred_binary.astype(bool)

    # Compute TP, FP, FN
    tp = gt & pr
    fp = (~gt) & pr
    fn = gt & (~pr)

    # Create overlay
    H, W = gt.shape
    overlay = np.zeros((H, W, 3), dtype=float)
    overlay[tp] = [0, 1, 0]   # green = TP
    overlay[fp] = [1, 0, 0]   # red   = FP
    overlay[fn] = [0, 0, 1]   # blue  = FN

    # Blend with original image
    blended = (1 - alpha) * img + alpha * overlay
    blended = np.clip(blended, 0, 1)

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 2],
                  hspace=0.3, wspace=0.3, left=0.05, right=0.95, top=0.92, bottom=0.05)

    # Title
    title_text = f'TP/FP/FN Maps'
    if class_name:
        title_text += f' - {class_name}'
    fig.suptitle(title_text, fontsize=28, fontweight='bold', color='#8B0000')

    # Top left: Confusion Matrix Diagram
    ax_cm = fig.add_subplot(gs[0, 0:2])
    create_confusion_matrix_diagram(ax_cm)

    # Top right: Metrics (if provided)
    if metrics:
        ax_metrics = fig.add_subplot(gs[0, 2:])
        ax_metrics.axis('off')

        metrics_text = "Performance Metrics\n" + "="*30 + "\n\n"
        if 'iou' in metrics:
            metrics_text += f"IoU:        {metrics['iou']:.4f}\n"
        if 'f1' in metrics:
            metrics_text += f"F1 Score:   {metrics['f1']:.4f}\n"
        if 'precision' in metrics:
            metrics_text += f"Precision:  {metrics['precision']:.4f}\n"
        if 'recall' in metrics:
            metrics_text += f"Recall:     {metrics['recall']:.4f}\n"

        # Add counts
        tp_count = tp.sum()
        fp_count = fp.sum()
        fn_count = fn.sum()
        tn_count = ((~gt) & (~pr)).sum()

        metrics_text += "\n" + "Pixel Counts\n" + "="*30 + "\n\n"
        metrics_text += f"TP: {tp_count:,}\n"
        metrics_text += f"FP: {fp_count:,}\n"
        metrics_text += f"FN: {fn_count:,}\n"
        metrics_text += f"TN: {tn_count:,}\n"

        ax_metrics.text(0.1, 0.5, metrics_text, ha='left', va='center',
                       fontsize=14, fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Bottom row: Image panels
    # 1) Input image
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(img)
    ax1.set_title("Input Image", fontsize=16, fontweight='bold')
    ax1.axis("off")

    # 2) Ground Truth
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(gt, cmap="gray")
    ax2.set_title(f"GT: {class_name}", fontsize=16, fontweight='bold')
    ax2.axis("off")

    # 3) Prediction
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(pr, cmap="gray")
    ax3.set_title(f"Pred: {class_name}", fontsize=16, fontweight='bold')
    ax3.axis("off")

    # 4) TP/FP/FN overlay
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.imshow(blended)
    ax4.set_title(f"TP/FP/FN for {class_name}\n(G=TP, R=FP, B=FN)",
                 fontsize=16, fontweight='bold')
    ax4.axis("off")

    # 5) Color legend
    ax5 = fig.add_subplot(gs[1, 4])
    ax5.axis('off')

    # Create color legend
    legend_elements = [
        mpatches.Patch(color='green', label=f'TP (True Positive)\n{tp.sum():,} pixels'),
        mpatches.Patch(color='red', label=f'FP (False Positive)\n{fp.sum():,} pixels'),
        mpatches.Patch(color='blue', label=f'FN (False Negative)\n{fn.sum():,} pixels'),
    ]

    ax5.legend(handles=legend_elements, loc='center', fontsize=14,
              frameon=True, fancybox=True, shadow=True,
              title='Color Legend', title_fontsize=16)

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Slide saved to: {save_path}")

    return fig


def demo_slide():
    """
    Create a demo slide with synthetic data
    """
    # Create synthetic data
    H, W = 256, 512

    # Synthetic input image (gradient)
    img = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            img[i, j] = [j/W, i/H, 0.5]

    # Create synthetic ground truth (circle)
    center_y, center_x = H//2, W//2
    radius = H//3
    y, x = np.ogrid[:H, :W]
    gt_binary = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)

    # Create synthetic prediction (shifted circle with some error)
    center_x_pred = center_x + 20
    pred_binary = ((x - center_x_pred)**2 + (y - center_y)**2 <= radius**2)

    # Compute metrics
    tp = (gt_binary & pred_binary).sum()
    fp = ((~gt_binary) & pred_binary).sum()
    fn = (gt_binary & (~pred_binary)).sum()
    tn = ((~gt_binary) & (~pred_binary)).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    # Create slide
    fig = create_tpfpfn_slide(
        image=img,
        gt_binary=gt_binary,
        pred_binary=pred_binary,
        class_name="hypocotyl",
        alpha=0.6,
        save_path="demo_tpfpfn_slide.png",
        metrics=metrics
    )

    plt.show()
    print("\nDemo slide created successfully!")
    print(f"IoU: {iou:.4f}")
    print(f"F1: {f1:.4f}")


if __name__ == "__main__":
    print("Creating demo TP/FP/FN slide...")
    demo_slide()
