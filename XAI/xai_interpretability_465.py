"""
Deep Learning Model Interpretability: Grad-CAM++ and LIME
Visualization pipeline for DenseNet201 Multi-Task CNN with image unification

Pipeline: Original Image → Unification → Grad-CAM++ & LIME

Author: AlImran2122071
Research Implementation - DenseNet201
Date: 2025-12-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Lime for model-agnostic interpretation
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Import the unification module
import sys
from pathlib import Path

# Add Model_inference directory to path
model_inference_path = Path(__file__).parent.parent / 'Model_inference'
sys.path.insert(0, str(model_inference_path))

from image_unifier_v2 import unify_image_for_visualization, process_bg_removed_image
from bg_remove_465 import remove_background

# =========================================================================
# CONFIGURATION - EASY TO MODIFY
# =========================================================================

# Model and data paths
MODEL_PATH = "/Users/alimran/Desktop/CSE465/best_DenseNet201.pt"  # Updated trained model
IMAGE_PATH = "/Users/alimran/Desktop/CSE465/Inference_Images/Copy of virus (42).jpg"  # Change to your test image

# Model architecture (for target layer selection)
TARGET_LAYER_NAME = "features"  # DenseNet201: features.denseblock4 is the last conv block

# Classification settings
TARGET_CLASS = None  # None = use predicted class; or specify int index (e.g., 0, 1, 2)

# LIME parameters
LIME_NUM_SAMPLES = 1000      # Number of perturbed samples
LIME_NUM_FEATURES = 10       # Top superpixels to highlight
LIME_HIDE_COLOR = 0          # Color for hidden regions (0=black)

# Output settings
OUTPUT_FILENAME = "xai_comparison.png"
FIGURE_DPI = 300
RANDOM_SEED = 42

# Visualization options
VISUALIZE_BOTH_TASKS = True  # True = show both species and health; False = species only

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =========================================================================
# DEVICE CONFIGURATION
# =========================================================================

def get_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# =========================================================================
# MODEL DEFINITION (Matching your architecture)
# =========================================================================

class MultiTaskDenseNet201(nn.Module):
    """Multi-task DenseNet-201 for species and disease classification"""
    def __init__(self, num_species=3, num_health=4, pretrained=False, dropout=0.3):
        super().__init__()
        from torchvision.models import densenet201, DenseNet201_Weights
        
        if pretrained:
            weights = DenseNet201_Weights.IMAGENET1K_V1
            self.backbone = densenet201(weights=weights)
        else:
            self.backbone = densenet201(weights=None)
        
        # DenseNet201 has classifier as a single Linear layer, not a Sequential
        in_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.head_species = nn.Linear(in_dim, num_species)
        self.head_health = nn.Linear(in_dim, num_health)
    
    def forward(self, x):
        feats = self.backbone(x)
        feats = self.dropout(feats)
        logits_species = self.head_species(feats)
        logits_health = self.head_health(feats)
        return logits_species, logits_health


# =========================================================================
# MODEL LOADING
# =========================================================================

def load_model(model_path, device):
    """
    Load trained model from checkpoint
    Falls back to pretrained DenseNet201 if model_path doesn't exist
    """
    model_path = Path(model_path)
    
    if model_path.exists():
        print(f"Loading model from: {model_path}")
        model = MultiTaskDenseNet201(num_species=3, num_health=4, pretrained=False, dropout=0.3)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"✓ DenseNet201 model loaded (Epoch: {checkpoint.get('epoch', 'N/A')})")
        if 'val_health' in checkpoint:
            print(f"  Val Health Accuracy: {checkpoint['val_health']:.4f}")
    else:
        print(f"⚠ Model path not found: {model_path}")
        print("Using pretrained DenseNet-201 as fallback")
        from torchvision.models import densenet201, DenseNet201_Weights
        model = MultiTaskDenseNet201(num_species=3, num_health=4, pretrained=True, dropout=0.3)
    
    model.to(device)
    model.eval()
    return model


def get_target_layer(model, layer_name="features"):
    """
    Get target convolutional layer for Grad-CAM++
    
    Args:
        model: PyTorch model
        layer_name: Name of the target layer (e.g., 'features' for DenseNet)
    
    Returns:
        Target layer module
    """
    # For our MultiTaskDenseNet201, access backbone.features.denseblock4
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        backbone = model
    
    if hasattr(backbone, layer_name):
        target = getattr(backbone, layer_name)
        # For DenseNet, features is a Sequential; we want denseblock4 (last dense block)
        if isinstance(target, nn.Sequential) and hasattr(target, 'denseblock4'):
            target = target.denseblock4
            print(f"✓ Target layer: backbone.{layer_name}.denseblock4")
        else:
            print(f"✓ Target layer: backbone.{layer_name}")
        return target
    else:
        raise ValueError(f"Layer '{layer_name}' not found in model")


# =========================================================================
# GRAD-CAM++ IMPLEMENTATION
# =========================================================================

class GradCAMPlusPlus:
    """
    Grad-CAM++ Implementation
    
    Reference:
    Chattopadhay, A., et al. (2018). Grad-CAM++: Generalized Gradient-Based 
    Visual Explanations for Deep Convolutional Networks. WACV 2018.
    
    Key improvement over Grad-CAM: Uses higher-order gradient terms (squared 
    and cubed) for better pixel-wise weighting, especially for multiple objects.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Forward hook: save activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None, task='species'):
        """
        Generate Grad-CAM++ heatmap with higher-order gradient weighting
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (None = predicted class)
            task: 'species' or 'health' for multi-task model
        
        Returns:
            cam: Normalized heatmap [0, 1] of shape (H, W)
        """
        self.model.eval()
        
        # Forward pass
        logits_species, logits_health = self.model(input_tensor)
        
        # Select task
        if task == 'species':
            logits = logits_species
        elif task == 'health':
            logits = logits_health
        else:
            logits = logits_species  # default
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients      # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # ============================================
        # Grad-CAM++ Weight Calculation
        # ============================================
        # Step 1: Compute gradient powers
        grad_squared = gradients.pow(2)
        grad_cubed = grad_squared * gradients
        
        # Step 2: Global sum for normalization
        spatial_sum = activations.sum(dim=(2, 3), keepdim=True)
        
        # Step 3: Alpha (pixel-wise importance weight)
        # α_kc = (∂²y^c/∂A_k²) / [2·(∂²y^c/∂A_k²) + Σ(∂³y^c/∂A_k³)·A_k]
        alpha_denom = grad_squared * 2 + grad_cubed * spatial_sum
        alpha_denom = torch.where(
            alpha_denom != 0, 
            alpha_denom, 
            torch.ones_like(alpha_denom)
        )
        alpha = grad_squared / alpha_denom
        
        # Step 4: Channel weights with ReLU-gated gradients
        # w_k = Σ_{i,j} α_k^{ij} · ReLU(∂y^c/∂A_k^{ij})
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Step 5: Weighted sum of activations
        # L_Grad-CAM++ = ReLU(Σ_k w_k · A_k)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_cam(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay Grad-CAM++ heatmap on image
        
        Args:
            image: Original image (H, W, 3) RGB or PIL Image
            cam: CAM heatmap (H', W')
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            overlay: RGB image with heatmap overlay
            cam_resized: Resized CAM matching image dimensions
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize CAM to match image
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay, cam_resized


# =========================================================================
# LIME IMPLEMENTATION
# =========================================================================

def run_lime(model, unified_image_pil, target_class=None, task='species',
             num_samples=1000, num_features=10, hide_color=0):
    """
    Apply LIME (Local Interpretable Model-agnostic Explanations)
    
    LIME is model-agnostic: it treats the model as a black box and explains
    predictions by perturbing superpixels and observing output changes.
    
    Args:
        model: PyTorch model
        unified_image_pil: PIL Image (unified format)
        target_class: Target class to explain (None = predicted)
        task: 'species' or 'health'
        num_samples: Number of perturbed samples
        num_features: Number of top superpixels to show
        hide_color: Color for hidden regions (0=black)
    
    Returns:
        lime_img: Image with decision boundaries
        explanation: LIME explanation object
    """
    print(f"Running LIME with {num_samples} samples...")
    
    # Convert PIL to numpy RGB
    unified_np = np.array(unified_image_pil)
    
    # Define prediction function for LIME
    def predict_fn(images):
        """
        Prediction function for LIME
        Takes batch of images (numpy RGB) and returns probabilities
        """
        batch_tensors = []
        for img in images:
            # Convert numpy to PIL
            pil_img = Image.fromarray(img.astype('uint8'))
            
            # Apply same preprocessing as inference
            # (LIME works on the unified image space)
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            batch_tensors.append(tensor)
        
        batch = torch.cat(batch_tensors, dim=0)
        
        with torch.no_grad():
            logits_species, logits_health = model(batch)
            
            if task == 'species':
                probs = F.softmax(logits_species, dim=1)
            else:
                probs = F.softmax(logits_health, dim=1)
        
        return probs.cpu().numpy()
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer(random_state=RANDOM_SEED)
    
    # Determine target class if not specified
    if target_class is None:
        probs = predict_fn([unified_np])
        target_class = probs[0].argmax()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        unified_np,
        predict_fn,
        top_labels=1,
        hide_color=hide_color,
        num_samples=num_samples,
        random_seed=RANDOM_SEED
    )
    
    # Get mask of important superpixels
    temp, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    
    # Mark boundaries in green
    lime_img = mark_boundaries(unified_np / 255.0, mask, color=(0, 1, 0), mode='thick')
    lime_img = (lime_img * 255).astype(np.uint8)
    
    print(f"✓ LIME explanation generated (class {target_class})")
    
    return lime_img, explanation


# =========================================================================
# VISUALIZATION
# =========================================================================

def visualize_interpretability(original_pil, unified_pil, gradcam_overlay, 
                               gradcam_heatmap, lime_img, output_path, 
                               task_name="Species", prediction="Unknown", confidence=0.0):
    """
    Create publication-quality 1×5 figure with all interpretability methods
    
    Layout:
        (a) Original Image
        (b) Unified Image  
        (c) GradCAM++ Attention Map
        (d) GradCAM++ Heatmap
        (e) LIME Decision Boundaries
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # (a) Original Image
    axes[0].imshow(original_pil)
    axes[0].set_title('(a) Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # (b) Unified Image
    axes[1].imshow(unified_pil)
    axes[1].set_title('(b) Unified Image', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # (c) GradCAM++ Attention Map (overlay)
    axes[2].imshow(gradcam_overlay)
    axes[2].set_title('(c) GradCAM++ Attention Map', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # (d) GradCAM++ Heatmap (standalone with colorbar)
    im = axes[3].imshow(gradcam_heatmap, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title('(d) GradCAM++ Heatmap', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
    
    # (e) LIME Decision Boundaries
    axes[4].imshow(lime_img)
    axes[4].set_title('(e) LIME Decision Boundaries', fontsize=12, fontweight='bold')
    axes[4].axis('off')
    
    # Add super title with prediction info
    fig.suptitle(f'{task_name} Task: {prediction} (Confidence: {confidence:.2%})', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved: {output_path}")
    plt.close()  # Close to free memory


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def main():
    """
    Main interpretability pipeline
    """
    print("="*70)
    print("EXPLAINABLE AI: Grad-CAM++ and LIME Visualization")
    print("="*70)
    
    # ===========================
    # 1. LOAD MODEL
    # ===========================
    print("\n[1/6] Loading model...")
    model = load_model(MODEL_PATH, DEVICE)
    target_layer = get_target_layer(model, TARGET_LAYER_NAME)
    
    # ===========================
    # 2. LOAD & UNIFY IMAGE
    # ===========================
    print("\n[2/6] Loading and unifying image...")
    image_path = Path(IMAGE_PATH)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Original image (for display only)
    original_pil = Image.open(image_path).convert('RGB')
    print(f"✓ Original image loaded: {original_pil.size}")
    
    # Background removal + Unification
    print("  → Removing background...")
    bg_removed = remove_background(str(image_path))
    
    print("  → Applying unification...")
    unified_np_bgr = unify_image_for_visualization(bg_removed)
    unified_pil = Image.fromarray(cv2.cvtColor(unified_np_bgr, cv2.COLOR_BGR2RGB))
    print(f"✓ Unified image ready: {unified_pil.size}")
    
    # ===========================
    # 3. MODEL INFERENCE
    # ===========================
    print("\n[3/6] Running inference on unified image...")
    input_tensor = process_bg_removed_image(bg_removed).to(DEVICE)
    
    with torch.no_grad():
        logits_species, logits_health = model(input_tensor)
    
    # Get predictions
    pred_species_idx = logits_species.argmax(dim=1).item()
    pred_health_idx = logits_health.argmax(dim=1).item()
    
    probs_species = F.softmax(logits_species, dim=1).cpu().numpy()[0]
    probs_health = F.softmax(logits_health, dim=1).cpu().numpy()[0]
    
    # Label mappings
    SPECIES_MAP = {0: "Eggplant", 1: "Potato", 2: "Tomato"}
    HEALTH_MAP = {0: "Bacterial", 1: "Fungal", 2: "Healthy", 3: "Virus"}
    
    print(f"✓ Species: {SPECIES_MAP[pred_species_idx]} ({probs_species[pred_species_idx]:.4f})")
    print(f"✓ Health: {HEALTH_MAP[pred_health_idx]} ({probs_health[pred_health_idx]:.4f})")
    
    # ===========================
    # 4. GENERATE VISUALIZATIONS FOR BOTH TASKS
    # ===========================
    
    if VISUALIZE_BOTH_TASKS:
        # ========== SPECIES TASK ==========
        print(f"\n[4/6] Generating Grad-CAM++ for SPECIES task...")
        target_task = 'species'
        target_cls = TARGET_CLASS if TARGET_CLASS is not None else pred_species_idx
        
        gradcam = GradCAMPlusPlus(model, target_layer)
        cam_species = gradcam.generate_cam(input_tensor, target_class=target_cls, task=target_task)
        
        gradcam_overlay_species, cam_resized_species = gradcam.overlay_cam(unified_pil, cam_species, alpha=0.4)
        print("✓ Grad-CAM++ generated for species")
        
        print(f"  → Running LIME for SPECIES...")
        lime_img_species, _ = run_lime(
            model, unified_pil, target_class=target_cls, task=target_task,
            num_samples=LIME_NUM_SAMPLES, num_features=LIME_NUM_FEATURES, hide_color=LIME_HIDE_COLOR
        )
        
        # ========== HEALTH TASK ==========
        print(f"\n[5/6] Generating Grad-CAM++ for HEALTH task...")
        target_task = 'health'
        target_cls = TARGET_CLASS if TARGET_CLASS is not None else pred_health_idx
        
        gradcam = GradCAMPlusPlus(model, target_layer)
        cam_health = gradcam.generate_cam(input_tensor, target_class=target_cls, task=target_task)
        
        gradcam_overlay_health, cam_resized_health = gradcam.overlay_cam(unified_pil, cam_health, alpha=0.4)
        print("✓ Grad-CAM++ generated for health")
        
        print(f"  → Running LIME for HEALTH...")
        lime_img_health, _ = run_lime(
            model, unified_pil, target_class=target_cls, task=target_task,
            num_samples=LIME_NUM_SAMPLES, num_features=LIME_NUM_FEATURES, hide_color=LIME_HIDE_COLOR
        )
        
        # ===========================
        # 6. CREATE VISUALIZATIONS
        # ===========================
        print(f"\n[6/6] Creating visualizations...")
        
        # Species visualization
        output_species = OUTPUT_FILENAME.replace('.png', '_species.png')
        visualize_interpretability(
            original_pil, unified_pil, gradcam_overlay_species, cam_resized_species, lime_img_species,
            output_species, 
            task_name="Species", 
            prediction=SPECIES_MAP[pred_species_idx],
            confidence=probs_species[pred_species_idx]
        )
        
        # Health visualization
        output_health = OUTPUT_FILENAME.replace('.png', '_health.png')
        visualize_interpretability(
            original_pil, unified_pil, gradcam_overlay_health, cam_resized_health, lime_img_health,
            output_health,
            task_name="Health",
            prediction=HEALTH_MAP[pred_health_idx],
            confidence=probs_health[pred_health_idx]
        )
        
    else:
        # Single task (original behavior)
        target_task = 'species'
        target_cls = TARGET_CLASS if TARGET_CLASS is not None else pred_species_idx
        
        print(f"\n[4/6] Generating Grad-CAM++ for {target_task} (class {target_cls})...")
        gradcam = GradCAMPlusPlus(model, target_layer)
        cam = gradcam.generate_cam(input_tensor, target_class=target_cls, task=target_task)
        
        gradcam_overlay, cam_resized = gradcam.overlay_cam(unified_pil, cam, alpha=0.4)
        print("✓ Grad-CAM++ generated")
        
        print(f"\n[5/6] Running LIME on unified image...")
        lime_img, _ = run_lime(
            model, unified_pil, target_class=target_cls, task=target_task,
            num_samples=LIME_NUM_SAMPLES, num_features=LIME_NUM_FEATURES, hide_color=LIME_HIDE_COLOR
        )
        
        print(f"\n[6/6] Creating visualization...")
        visualize_interpretability(
            original_pil, unified_pil, gradcam_overlay, cam_resized, lime_img,
            OUTPUT_FILENAME,
            task_name="Species",
            prediction=SPECIES_MAP[pred_species_idx],
            confidence=probs_species[pred_species_idx]
        )
    
    # ===========================
    # EXPLANATION
    # ===========================
    print("\n" + "="*70)
    print("INTERPRETABILITY METHODS SUMMARY")
    print("="*70)
    print("""
📊 Grad-CAM++ (Gradient-weighted Class Activation Mapping ++):
   • Uses HIGHER-ORDER gradient terms (squared & cubed) for pixel-wise weighting
   • Provides SHARPER localization compared to vanilla Grad-CAM
   • Particularly effective for multiple objects or fine-grained features
   • Gradient-based → model-specific but computationally efficient

🔍 LIME (Local Interpretable Model-agnostic Explanations):
   • MODEL-AGNOSTIC: treats classifier as black box
   • Perturbs SUPERPIXELS and observes prediction changes
   • Shows LOCAL decision regions via interpretable linear model
   • More intuitive for non-experts; complements gradient methods

✨ Pipeline: Both methods applied to UNIFIED IMAGE
   • Ensures interpretability matches training/serving distribution
   • Background removal + normalization applied BEFORE explanation
   • Consistent with actual model decision-making process
    """)
    print("="*70)
    if VISUALIZE_BOTH_TASKS:
        print(f"\n✅ Complete! Results saved:")
        print(f"   - Species: {OUTPUT_FILENAME.replace('.png', '_species.png')}")
        print(f"   - Health: {OUTPUT_FILENAME.replace('.png', '_health.png')}")
    else:
        print(f"\n✅ Complete! Results saved to: {OUTPUT_FILENAME}")
    print("="*70)


if __name__ == "__main__":
    main()
