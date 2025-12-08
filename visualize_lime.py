#!/usr/bin/env python3
"""
LIME (Local Interpretable Model-agnostic Explanations) for Semantic Segmentation

This script uses LIME to explain predictions from a pretrained segmentation model.
LIME creates interpretable explanations by perturbing superpixels and observing
changes in model predictions.

Usage:
    python visualize_lime.py --model best_model_FPN.pth --image path/to/image.jpg --class hypocotyl
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# LIME imports
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    print("LIME not installed. Installing...")
    os.system("pip install lime scikit-image")
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

from skimage.segmentation import slic, felzenszwalb, quickshift


class SegmentationLIME:
    """
    LIME explainer for semantic segmentation models
    """

    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Pretrained segmentation model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.model.eval()
        self.device = device

    def preprocess_image(self, image_path, img_size=(512, 256)):
        """
        Load and preprocess image for model input

        Args:
            image_path: Path to input image
            img_size: Target size (width, height)

        Returns:
            image_np: Preprocessed numpy array for LIME (H, W, 3)
            original_img: Original image for visualization
        """
        # Read image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img.copy()

        # Resize
        img = cv2.resize(img, img_size)

        # Normalize to [0, 1] for LIME
        img = img.astype(np.float32) / 255.0

        return img, original_img

    def predict_fn(self, images, class_idx):
        """
        Prediction function for LIME (batch processing)

        Args:
            images: Batch of images (B, H, W, 3) in [0, 1] range
            class_idx: Index of class to predict

        Returns:
            predictions: Batch of prediction scores (B, 2) for binary classification
        """
        batch_size = images.shape[0]
        predictions = np.zeros((batch_size, 2))

        for i, img in enumerate(images):
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_normalized = (img - mean) / std

            # To tensor
            img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                logits = self.model(img_tensor)
                # Get mean prediction for the class across all pixels
                class_prob = torch.sigmoid(logits[0, class_idx]).mean().item()

            # Binary classification: [negative_class, positive_class]
            predictions[i, 0] = 1 - class_prob  # Not class
            predictions[i, 1] = class_prob      # Is class

        return predictions

    def explain_with_lime(self, image_np, class_idx,
                         num_samples=1000,
                         num_features=10,
                         segmentation_fn='slic',
                         **seg_kwargs):
        """
        Generate LIME explanation

        Args:
            image_np: Input image (H, W, 3) in [0, 1] range
            class_idx: Index of class to explain
            num_samples: Number of perturbed samples
            num_features: Number of superpixels to highlight
            segmentation_fn: Segmentation algorithm ('slic', 'felzenszwalb', 'quickshift')
            **seg_kwargs: Additional arguments for segmentation

        Returns:
            explanation: LIME explanation object
            segments: Superpixel segmentation
        """
        print(f"Generating LIME explanation for class {class_idx}...")
        print(f"Using {segmentation_fn} segmentation with {num_samples} samples...")

        # Create segmentation function
        if segmentation_fn == 'slic':
            seg_fn = lambda img: slic(img,
                                     n_segments=seg_kwargs.get('n_segments', 50),
                                     compactness=seg_kwargs.get('compactness', 10),
                                     sigma=seg_kwargs.get('sigma', 1))
        elif segmentation_fn == 'felzenszwalb':
            seg_fn = lambda img: felzenszwalb(img,
                                             scale=seg_kwargs.get('scale', 100),
                                             sigma=seg_kwargs.get('sigma', 0.5),
                                             min_size=seg_kwargs.get('min_size', 50))
        else:  # quickshift
            seg_fn = lambda img: quickshift(img,
                                           kernel_size=seg_kwargs.get('kernel_size', 4),
                                           max_dist=seg_kwargs.get('max_dist', 200),
                                           ratio=seg_kwargs.get('ratio', 0.5))

        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()

        # Generate explanation
        explanation = explainer.explain_instance(
            image_np,
            classifier_fn=lambda imgs: self.predict_fn(imgs, class_idx),
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=seg_fn
        )

        # Get segments
        segments = seg_fn(image_np)

        return explanation, segments

    def get_importance_map(self, explanation, segments, label=None):
        """
        Convert LIME explanation to pixel-level importance map

        Args:
            explanation: LIME explanation object
            segments: Superpixel segmentation
            label: Label to explain (None = use top label)

        Returns:
            importance_map: Pixel-level importance (H, W)
        """
        # Get the label to explain (use top label if not specified)
        if label is None:
            label = explanation.top_labels[0]

        # Get the local explanation
        local_exp = explanation.local_exp[label]

        # Create importance map
        importance_map = np.zeros(segments.shape, dtype=np.float32)

        for segment_id, weight in local_exp:
            importance_map[segments == segment_id] = weight

        return importance_map


def visualize_lime_explanation(image, explanation, segments, class_name,
                               importance_map=None,
                               prediction=None, ground_truth=None,
                               num_features=10,
                               save_path=None):
    """
    Create comprehensive LIME visualization

    Args:
        image: Original image (H, W, 3)
        explanation: LIME explanation object
        segments: Superpixel segmentation
        class_name: Name of the class being explained
        importance_map: Pixel-level importance map (H, W)
        prediction: Optional prediction mask (H, W)
        ground_truth: Optional ground truth mask (H, W)
        num_features: Number of top features to show
        save_path: Path to save visualization
    """
    # Determine number of panels
    num_panels = 4
    if prediction is not None:
        num_panels += 1
    if ground_truth is not None:
        num_panels += 1

    # Create figure
    fig = plt.figure(figsize=(5 * min(num_panels, 3), 10))

    if num_panels <= 3:
        gs = GridSpec(1, num_panels, figure=fig, wspace=0.3)
    else:
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(f'LIME Explanation - {class_name}',
                 fontsize=18, fontweight='bold', color='#8B0000')

    # Get masked images from LIME
    temp, mask_positive = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )

    temp, mask_negative = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=num_features,
        hide_rest=False,
        negative_only=True
    )

    # Panel positions
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    panel = 0

    # Panel 1: Original image with superpixel boundaries
    ax1 = fig.add_subplot(gs[positions[panel][0], positions[panel][1]])
    ax1.imshow(mark_boundaries(image, segments))
    ax1.set_title("Superpixel Segmentation", fontsize=12, fontweight='bold')
    ax1.axis('off')
    panel += 1

    # Panel 2: Positive features (supporting prediction)
    ax2 = fig.add_subplot(gs[positions[panel][0], positions[panel][1]])
    ax2.imshow(mark_boundaries(image, mask_positive))
    ax2.set_title(f"Positive Features\n(Support {class_name})",
                 fontsize=12, fontweight='bold', color='green')
    ax2.axis('off')
    panel += 1

    # Panel 3: Negative features (against prediction)
    if mask_negative.any():
        ax3 = fig.add_subplot(gs[positions[panel][0], positions[panel][1]])
        ax3.imshow(mark_boundaries(image, mask_negative))
        ax3.set_title(f"Negative Features\n(Against {class_name})",
                     fontsize=12, fontweight='bold', color='red')
        ax3.axis('off')
        panel += 1

    # Panel 4: Importance heatmap
    if importance_map is not None:
        ax4 = fig.add_subplot(gs[positions[panel][0], positions[panel][1]])
        im = ax4.imshow(importance_map, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title("Importance Heatmap", fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        panel += 1

    # Optional: Ground truth
    if ground_truth is not None and panel < len(positions):
        ax_gt = fig.add_subplot(gs[positions[panel][0], positions[panel][1]])
        ax_gt.imshow(ground_truth, cmap='gray')
        ax_gt.set_title(f"GT: {class_name}", fontsize=12, fontweight='bold')
        ax_gt.axis('off')
        panel += 1

    # Optional: Prediction
    if prediction is not None and panel < len(positions):
        ax_pred = fig.add_subplot(gs[positions[panel][0], positions[panel][1]])
        ax_pred.imshow(prediction, cmap='gray')
        ax_pred.set_title(f"Prediction: {class_name}", fontsize=12, fontweight='bold')
        ax_pred.axis('off')
        panel += 1

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                   exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"LIME visualization saved to: {save_path}")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='LIME Explainability for Segmentation')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--class_idx', type=int, default=2,
                       help='Class index to explain (default: 2 for hypocotyl)')
    parser.add_argument('--class_name', type=str, default='hypocotyl',
                       help='Class name for visualization')
    parser.add_argument('--output_dir', type=str, default='lime_explanations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of perturbed samples for LIME')
    parser.add_argument('--num_features', type=int, default=10,
                       help='Number of superpixels to highlight')
    parser.add_argument('--segmentation', type=str, default='slic',
                       choices=['slic', 'felzenszwalb', 'quickshift'],
                       help='Superpixel segmentation algorithm')
    parser.add_argument('--n_segments', type=int, default=50,
                       help='Number of superpixels (for SLIC)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.model}")
    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()

    # Create LIME explainer
    lime_explainer = SegmentationLIME(model, device=device)

    # Preprocess image
    print(f"Loading image: {args.image}")
    image_np, original_img = lime_explainer.preprocess_image(args.image)

    # Get model prediction
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (image_np - mean) / std
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = (logits[0, args.class_idx] > 0.5).cpu().numpy()

    # Generate LIME explanation
    explanation, segments = lime_explainer.explain_with_lime(
        image_np,
        args.class_idx,
        num_samples=args.num_samples,
        num_features=args.num_features,
        segmentation_fn=args.segmentation,
        n_segments=args.n_segments
    )

    # Get importance map (uses top label automatically)
    importance_map = lime_explainer.get_importance_map(explanation, segments)

    # Create visualization
    output_path = os.path.join(args.output_dir,
                               f"lime_{args.class_name}_{args.segmentation}.png")

    fig = visualize_lime_explanation(
        image=image_np,
        explanation=explanation,
        segments=segments,
        class_name=args.class_name,
        importance_map=importance_map,
        prediction=pred_mask,
        num_features=args.num_features,
        save_path=output_path
    )

    plt.show()
    print(f"\nLIME explanation complete!")
    print(f"Segmentation: {args.segmentation.upper()}")
    print(f"Number of superpixels: {segments.max() + 1}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
