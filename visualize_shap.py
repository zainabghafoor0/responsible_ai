#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) for Semantic Segmentation

This script uses SHAP to explain predictions from a pretrained segmentation model.
SHAP provides pixel-level importance scores showing which pixels contribute
most to the model's predictions for each class.

Usage:
    python visualize_shap.py --model best_model_FPN.pth --image path/to/image.jpg --class hypocotyl
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

# SHAP imports
try:
    import shap
except ImportError:
    print("SHAP not installed. Installing...")
    os.system("pip install shap")
    import shap


class SegmentationSHAP:
    """
    SHAP explainer for semantic segmentation models
    """

    def __init__(self, model, device='cuda', num_samples=50):
        """
        Args:
            model: Pretrained segmentation model
            device: Device to run on ('cuda' or 'cpu')
            num_samples: Number of samples for SHAP background (lower = faster)
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.num_samples = num_samples

        # Fix in-place operations for SHAP compatibility
        self._fix_inplace_ops()

    def _fix_inplace_ops(self):
        """
        Recursively disable in-place operations in the model.
        This is necessary for DeepSHAP to work properly.
        """
        def disable_inplace(module):
            for child in module.children():
                if isinstance(child, torch.nn.ReLU):
                    child.inplace = False
                disable_inplace(child)

        disable_inplace(self.model)

    def preprocess_image(self, image_path, img_size=(512, 256)):
        """
        Load and preprocess image for model input

        Args:
            image_path: Path to input image
            img_size: Target size (width, height)

        Returns:
            image_tensor: Preprocessed tensor (1, 3, H, W)
            original_img: Original image for visualization
        """
        # Read image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img.copy()

        # Resize
        img = cv2.resize(img, img_size)

        # Normalize (ImageNet stats)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # To tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        return img_tensor, original_img

    def model_predict(self, x):
        """
        Wrapper for model prediction (needed for SHAP)

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            predictions: Model output (B, C, H, W)
        """
        with torch.no_grad():
            output = self.model(x)
        return output

    def explain_with_gradient_shap(self, image_tensor, class_idx, num_background=10):
        """
        Generate SHAP explanations using GradientSHAP

        Note: For segmentation models, DeepSHAP is recommended as it handles
        spatial outputs better. GradientSHAP uses a scalar summary (sum of predictions).

        Args:
            image_tensor: Input image tensor (1, 3, H, W)
            class_idx: Index of class to explain
            num_background: Number of background samples

        Returns:
            shap_values: SHAP values for the image (H, W)
        """
        print(f"Generating GradientSHAP explanation for class {class_idx}...")
        print("⚠️  Note: For segmentation, 'deep' method is recommended (--method deep)")

        # Create background dataset (using random noise around the input)
        background = image_tensor.clone().repeat(num_background, 1, 1, 1)
        noise = torch.randn_like(background) * 0.1
        background = background + noise

        # Create a wrapper model that outputs scalar (sum of target class)
        class ClassOutputModel(torch.nn.Module):
            def __init__(self, model, class_idx):
                super().__init__()
                self.model = model
                self.class_idx = class_idx

            def forward(self, x):
                output = self.model(x)
                # Return sum over spatial dimensions for the target class (scalar output)
                return output[:, self.class_idx, :, :].sum(dim=(1, 2))

        wrapped_model = ClassOutputModel(self.model, class_idx)

        # Create explainer with wrapped model
        explainer = shap.GradientExplainer(
            model=wrapped_model,
            data=background
        )

        # Get SHAP values for input
        shap_values = explainer.shap_values(image_tensor)

        # Process SHAP values - average over RGB channels
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Shape: (1, 3, H, W) -> (H, W)
        shap_numpy = np.array(shap_values)
        if len(shap_numpy.shape) == 4:
            shap_map = np.abs(shap_numpy[0]).mean(axis=0)  # Average over channels
        elif len(shap_numpy.shape) == 3:
            shap_map = np.abs(shap_numpy).mean(axis=0)  # Average over channels
        else:
            shap_map = np.abs(shap_numpy)

        return shap_map

    def explain_with_deep_shap(self, image_tensor, class_idx, num_background=10):
        """
        Generate SHAP explanations using DeepSHAP (DeepLIFT approximation)

        Note: DeepSHAP is recommended over GradientSHAP for segmentation models
        as it doesn't require scalar outputs.

        Args:
            image_tensor: Input image tensor (1, 3, H, W)
            class_idx: Index of class to explain
            num_background: Number of background samples

        Returns:
            shap_values: SHAP values for the image (H, W)
        """
        print(f"Generating DeepSHAP explanation for class {class_idx}...")
        print("Note: DeepSHAP is the recommended method for segmentation models")

        # Create background dataset
        background = image_tensor.clone().repeat(num_background, 1, 1, 1)
        noise = torch.randn_like(background) * 0.1
        background = background + noise

        # Wrapper model for class-specific output
        class ClassSegmentationModel(torch.nn.Module):
            def __init__(self, model, class_idx):
                super().__init__()
                self.model = model
                self.class_idx = class_idx

            def forward(self, x):
                output = self.model(x)
                # Return only target class channel, keeping spatial dimensions
                return output[:, self.class_idx:self.class_idx+1, :, :]

        wrapped_model = ClassSegmentationModel(self.model, class_idx)

        # Create explainer
        explainer = shap.DeepExplainer(wrapped_model, background)

        # Compute SHAP values
        shap_values = explainer.shap_values(image_tensor)

        # Process SHAP values
        # shap_values shape: (1, 3, H, W) - SHAP values for input channels
        if isinstance(shap_values, list):
            # If list, take first element
            shap_numpy = np.array(shap_values[0])
        else:
            shap_numpy = np.array(shap_values)

        # Average absolute SHAP values over input channels
        if len(shap_numpy.shape) == 4:
            # (1, 3, H, W) -> (H, W)
            shap_map = np.abs(shap_numpy[0]).mean(axis=0)
        elif len(shap_numpy.shape) == 3:
            # (3, H, W) -> (H, W)
            shap_map = np.abs(shap_numpy).mean(axis=0)
        else:
            shap_map = np.abs(shap_numpy)

        return shap_map

    def explain_with_kernel_shap(self, image_tensor, class_idx, num_samples=100):
        """
        Generate SHAP explanations using KernelSHAP (slower but model-agnostic)

        Args:
            image_tensor: Input image tensor (1, 3, H, W)
            class_idx: Index of class to explain
            num_samples: Number of samples for approximation

        Returns:
            shap_values: SHAP values for the image (H, W)
        """
        print(f"Generating KernelSHAP explanation for class {class_idx}...")
        print("Note: KernelSHAP is slow. Consider using GradientSHAP for faster results.")

        # For segmentation, we use superpixel-based explanation
        from skimage.segmentation import slic

        # Convert to numpy for superpixel segmentation
        img_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Generate superpixels
        segments = slic(img_np, n_segments=50, compactness=10)

        # Masking function
        def mask_image(mask, img_tensor):
            """Mask image based on superpixel segments"""
            masked = img_tensor.clone()
            for i in range(len(mask)):
                if mask[i] == 0:
                    masked[0, :, segments == i] = 0
            return masked

        # Prediction function
        def predict_fn(masks):
            """Predict for masked images"""
            outputs = []
            for mask in masks:
                masked_img = mask_image(mask, image_tensor)
                with torch.no_grad():
                    output = self.model(masked_img)
                # Average prediction for the class
                score = output[0, class_idx].mean().item()
                outputs.append(score)
            return np.array(outputs)

        # Create explainer
        explainer = shap.KernelExplainer(
            predict_fn,
            np.ones((1, segments.max() + 1))
        )

        # Compute SHAP values
        shap_values = explainer.shap_values(
            np.ones((1, segments.max() + 1)),
            nsamples=num_samples
        )

        # Map superpixel importance back to pixel space
        shap_map = np.zeros(segments.shape)
        for i in range(segments.max() + 1):
            shap_map[segments == i] = np.abs(shap_values[0, i])

        return shap_map


def visualize_shap_explanation(image, shap_map, class_name,
                               prediction=None, ground_truth=None,
                               save_path=None):
    """
    Create comprehensive SHAP visualization

    Args:
        image: Original image (H, W, 3)
        shap_map: SHAP importance map (H, W)
        class_name: Name of the class being explained
        prediction: Optional prediction mask (H, W)
        ground_truth: Optional ground truth mask (H, W)
        save_path: Path to save visualization
    """
    # Normalize image
    img = image.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    img = np.clip(img, 0, 1)

    # Resize image to match SHAP map if needed
    if img.shape[:2] != shap_map.shape:
        img = cv2.resize(img, (shap_map.shape[1], shap_map.shape[0]))

    # Normalize SHAP values
    shap_norm = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)

    # Create overlay
    shap_colored = plt.cm.jet(shap_norm)[:, :, :3]
    overlay = 0.5 * img + 0.5 * shap_colored

    # Determine number of panels
    num_panels = 3
    if prediction is not None:
        num_panels += 1
    if ground_truth is not None:
        num_panels += 1

    # Create figure
    fig = plt.figure(figsize=(5 * num_panels, 6))
    gs = GridSpec(1, num_panels, figure=fig, wspace=0.3)

    fig.suptitle(f'SHAP Explanation - {class_name}',
                 fontsize=18, fontweight='bold', color='#8B0000')

    panel = 0

    # Panel 1: Original image
    ax1 = fig.add_subplot(gs[0, panel])
    ax1.imshow(img)
    ax1.set_title("Input Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    panel += 1

    # Panel 2: SHAP heatmap
    ax2 = fig.add_subplot(gs[0, panel])
    im = ax2.imshow(shap_map, cmap='jet')
    ax2.set_title("SHAP Importance Map", fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    panel += 1

    # Panel 3: Overlay
    ax3 = fig.add_subplot(gs[0, panel])
    ax3.imshow(overlay)
    ax3.set_title("SHAP Overlay", fontsize=14, fontweight='bold')
    ax3.axis('off')
    panel += 1

    # Optional: Ground truth
    if ground_truth is not None:
        ax_gt = fig.add_subplot(gs[0, panel])
        ax_gt.imshow(ground_truth, cmap='gray')
        ax_gt.set_title(f"GT: {class_name}", fontsize=14, fontweight='bold')
        ax_gt.axis('off')
        panel += 1

    # Optional: Prediction
    if prediction is not None:
        ax_pred = fig.add_subplot(gs[0, panel])
        ax_pred.imshow(prediction, cmap='gray')
        ax_pred.set_title(f"Prediction: {class_name}", fontsize=14, fontweight='bold')
        ax_pred.axis('off')
        panel += 1

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                   exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"SHAP visualization saved to: {save_path}")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='SHAP Explainability for Segmentation')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--class_idx', type=int, default=2,
                       help='Class index to explain (default: 2 for hypocotyl)')
    parser.add_argument('--class_name', type=str, default='hypocotyl',
                       help='Class name for visualization')
    parser.add_argument('--method', type=str, default='deep',
                       choices=['gradient', 'deep', 'kernel'],
                       help='SHAP method: deep (recommended for segmentation), gradient (may fail), kernel (slow)')
    parser.add_argument('--output_dir', type=str, default='shap_explanations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of background samples')
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

    # Create SHAP explainer
    shap_explainer = SegmentationSHAP(model, device=device, num_samples=args.num_samples)

    # Preprocess image
    print(f"Loading image: {args.image}")
    image_tensor, original_img = shap_explainer.preprocess_image(args.image)

    # Get model prediction
    with torch.no_grad():
        logits = model(image_tensor)
        pred_mask = (logits[0, args.class_idx] > 0.5).cpu().numpy()

    # Generate SHAP explanation
    if args.method == 'gradient':
        try:
            shap_map = shap_explainer.explain_with_gradient_shap(
                image_tensor, args.class_idx, num_background=args.num_samples
            )
        except (IndexError, RuntimeError) as e:
            print(f"\n❌ GradientSHAP failed: {str(e)}")
            print("🔄 Automatically switching to DeepSHAP (recommended for segmentation)...\n")
            shap_map = shap_explainer.explain_with_deep_shap(
                image_tensor, args.class_idx, num_background=args.num_samples
            )
            args.method = 'deep'  # Update method name for output file
    elif args.method == 'deep':
        shap_map = shap_explainer.explain_with_deep_shap(
            image_tensor, args.class_idx, num_background=args.num_samples
        )
    else:  # kernel
        shap_map = shap_explainer.explain_with_kernel_shap(
            image_tensor, args.class_idx, num_samples=args.num_samples
        )

    # Create visualization
    output_path = os.path.join(args.output_dir,
                               f"shap_{args.class_name}_{args.method}.png")

    fig = visualize_shap_explanation(
        image=original_img,
        shap_map=shap_map,
        class_name=args.class_name,
        prediction=pred_mask,
        save_path=output_path
    )

    plt.show()
    print(f"\nSHAP explanation complete!")
    print(f"Method: {args.method.upper()}SHAP")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
