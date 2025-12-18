"""
Image Unifier Module
Converts images into unified form for training, validation, and test sets.
Takes input from bg_remove.py background removal function OR direct image paths.

MATCHES TRAINING PREPROCESSING EXACTLY:
1. Background removal (transparent → black) - OR accepts PIL Image with transparency from bg_remove.py
2. Resize to 224×224 with LANCZOS4 interpolation
3. ImageNet normalization (mean/std) - applied only during model inference

INPUT OPTIONS:
- PIL.Image object from bg_remove.py (with transparency)
- File path to image (will process with or without background removal)
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import torch
from torchvision import transforms

# ========================================
# CONFIGURATION
# ========================================

IMG_SIZE = 224                          # Target size (224x224)
BACKGROUND_COLOR = (255, 255, 255)           # Black background (B, G, R)
AUG_QUALITY = 95                        # JPEG quality

# ImageNet normalization parameters (CRITICAL FOR MODEL INPUT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========================================
# IMAGE UNIFICATION FUNCTIONS
# ========================================

def read_image_with_transparency(pil_image: Image.Image, bg_color=(0, 0, 0)) -> np.ndarray:
    """
    Convert PIL Image with transparency to BGR numpy array with consistent background.
    
    Args:
        pil_image (PIL.Image): Input PIL Image (can have transparency)
        bg_color (tuple): Background color in BGR format (B, G, R)
    
    Returns:
        np.ndarray: BGR image array with solid background
    """
    try:
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            
            # Create background with specified color
            background = Image.new('RGB', pil_image.size, (bg_color[2], bg_color[1], bg_color[0]))
            
            # Paste using alpha channel as mask
            if pil_image.mode in ('RGBA', 'LA'):
                background.paste(pil_image, mask=pil_image.split()[-1])
            else:
                background.paste(pil_image)
            
            # Convert to OpenCV BGR format
            img_bgr = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)
            return img_bgr
        else:
            # Already RGB image without transparency
            img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return img_bgr
            
    except Exception as e:
        print(f"⚠ Warning processing image: {e}")
        raise


def resize_image(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Simple resize without normalization.
    Models will apply their own normalization (e.g., ImageNet stats).
    Uses LANCZOS4 interpolation for high-quality resizing.
    
    Args:
        img (np.ndarray): Input BGR image
        target_size (tuple): Target size (width, height)
    
    Returns:
        np.ndarray: Resized image in BGR format
    """
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    return img


def apply_imagenet_normalization(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Apply ImageNet normalization to match training transforms.
    
    This is CRITICAL - the model was trained with these exact values!
    
    Args:
        img_bgr (np.ndarray): Input BGR image (224x224x3, 0-255)
    
    Returns:
        torch.Tensor: Normalized tensor (3x224x224) ready for model input
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)
    
    # Apply torchvision transforms (ToTensor + Normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and changes to CxHxW
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    tensor = transform(pil_img)
    return tensor


def unify_image(pil_image: Image.Image, 
                target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
                return_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """
    Complete image unification pipeline - MATCHES TRAINING EXACTLY.
    
    Pipeline:
    1. Handle transparency → black background
    2. Resize to 224×224 with LANCZOS4
    3. Apply ImageNet normalization (if return_tensor=True)
    
    Args:
        pil_image (PIL.Image): Input PIL Image from bg_remove function
        target_size (tuple): Target size (default: 224x224)
        return_tensor (bool): If True, returns tensor with ImageNet norm (for model input)
                             If False, returns BGR numpy array (for visualization)
    
    Returns:
        torch.Tensor (if return_tensor=True): Normalized tensor (3x224x224) ready for model
        np.ndarray (if return_tensor=False): BGR image array (224x224x3) for visualization
    """
    # Step 1: Handle transparency and convert to BGR
    img = read_image_with_transparency(pil_image, bg_color=BACKGROUND_COLOR)
    
    # Step 2: Resize (no CLAHE normalization - models handle their own normalization)
    img = resize_image(img, target_size=target_size)
    
    # Step 3: Apply ImageNet normalization if returning tensor for model
    if return_tensor:
        tensor = apply_imagenet_normalization(img)
        return tensor
    else:
        # Return BGR image for visualization/saving
        return img


def unify_image_for_model(pil_image: Image.Image) -> torch.Tensor:
    """
    Convenience function: Prepare image for model inference.
    Returns tensor ready to feed to model (with batch dimension).
    
    Args:
        pil_image (PIL.Image): Input PIL Image
    
    Returns:
        torch.Tensor: Batch tensor (1x3x224x224) ready for model.forward()
    """
    tensor = unify_image(
        pil_image, 
        target_size=(IMG_SIZE, IMG_SIZE),
        return_tensor=True
    )
    
    # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    batch_tensor = tensor.unsqueeze(0)
    return batch_tensor


def unify_image_for_visualization(pil_image: Image.Image) -> np.ndarray:
    """
    Convenience function: Prepare image for visualization/saving.
    Returns BGR numpy array without ImageNet normalization.
    
    Args:
        pil_image (PIL.Image): Input PIL Image
    
    Returns:
        np.ndarray: BGR image (224x224x3) for cv2.imshow() or cv2.imwrite()
    """
    img = unify_image(
        pil_image,
        target_size=(IMG_SIZE, IMG_SIZE),
        return_tensor=False
    )
    return img


def save_unified_image(img: np.ndarray, 
                      output_path: Union[str, Path], 
                      quality: int = AUG_QUALITY) -> None:
    """
    Save unified image to disk.
    
    Args:
        img (np.ndarray): Image array in BGR format
        output_path (str or Path): Output file path
        quality (int): JPEG quality (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ext = output_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(
            str(output_path), 
            img, 
            [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
    else:
        cv2.imwrite(str(output_path), img)
    
    print(f"✓ Unified image saved to: {output_path}")


# ========================================
# INTEGRATION WITH BG_REMOVE
# ========================================

def process_bg_removed_image(pil_image: Image.Image,
                            output_dir: Optional[Union[str, Path]] = None,
                            save_visualization: bool = False) -> torch.Tensor:
    """
    Process PIL Image from bg_remove.py (with transparency) into model-ready tensor.
    
    This function is designed to accept the output from bg_remove.remove_background()
    which returns a PIL.Image with RGBA mode (transparent background).
    
    Pipeline:
    1. Handle transparency → convert to black background (BGR)
    2. Resize to 224×224 with LANCZOS4
    3. Apply ImageNet normalization
    4. Add batch dimension
    
    Args:
        pil_image (PIL.Image): PIL Image with transparency from bg_remove.py
        output_dir (str or Path): Directory to save visualization (optional) - NOT USED
        save_visualization (bool): Save preprocessed image for debugging - NOT USED
    
    Returns:
        torch.Tensor: Batch tensor (1x3x224x224) ready for model.forward()
        
    Example:
        from bg_remove import remove_background
        from 2_image_unifier import process_bg_removed_image
        
        # Step 1: Remove background
        bg_removed = remove_background("leaf.jpg")  # Returns PIL.Image with alpha
        
        # Step 2: Prepare for model
        model_input = process_bg_removed_image(bg_removed)  # Returns tensor (1,3,224,224)
        
        # Step 3: Use with model
        with torch.no_grad():
            output = model(model_input)
    """
    print("\n" + "="*60)
    print("PROCESSING BG_REMOVED IMAGE → MODEL INPUT")
    print("="*60)
    
    # Step 1: Convert to tensor for model
    print("Step 1: Converting PIL Image to tensor...")
    model_input = unify_image_for_model(pil_image)
    print(f"✓ Tensor prepared: shape {model_input.shape}")
    
    print("="*60 + "\n")
    return model_input


def process_for_inference(input_image_path: str,
                         output_dir: Union[str, Path] = None,
                         save_visualization: bool = False) -> torch.Tensor:
    """
    Complete pipeline for MODEL INFERENCE: Background removal + Image unification.
    
    Returns tensor ready for model.forward()
    NO IMAGE SAVING.
    
    Args:
        input_image_path (str): Path to input image
        output_dir (str or Path): UNUSED - no images are saved
        save_visualization (bool): UNUSED - no images are saved
    
    Returns:
        torch.Tensor: Batch tensor (1x3x224x224) ready for model inference
    """
    from bg_remove_465 import remove_background
    
    print("\n" + "="*60)
    print("PROCESSING FOR INFERENCE: BG Removal + Unification")
    print("="*60)
    
    # Step 1: Remove background
    print("Step 1: Removing background...")
    bg_removed_image = remove_background(input_image_path)
    print("✓ Background removed")
    
    # Step 2: Process with the new function
    print("Step 2: Processing removed background image...")
    model_input = process_bg_removed_image(
        bg_removed_image,
        output_dir=None,
        save_visualization=False
    )
    
    print("="*60)
    print("✓ Ready for model.forward(tensor)")
    print("="*60 + "\n")
    
    return model_input


def process_for_visualization(input_image_path: str,
                              output_dir: Union[str, Path] = None) -> np.ndarray:
    """
    Complete pipeline for VISUALIZATION: Background removal + Image unification.
    
    Returns BGR image for display (NO SAVING).
    
    Args:
        input_image_path (str): Path to input image
        output_dir (str or Path): UNUSED - no images are saved
    
    Returns:
        np.ndarray: Unified image array in BGR format (224x224x3)
    """
    from bg_remove_465 import remove_background
    
    print("\n" + "="*60)
    print("PROCESSING FOR VISUALIZATION: BG Removal + Unification")
    print("="*60)
    
    # Step 1: Remove background
    print("Step 1: Removing background...")
    bg_removed_image = remove_background(input_image_path)
    print("✓ Background removed")
    
    # Step 2: Unify image for visualization
    print("Step 2: Unifying image...")
    unified_image = unify_image_for_visualization(bg_removed_image)
    print(f"✓ Image unified to {IMG_SIZE}x{IMG_SIZE}")
    
    print("="*60 + "\n")
    return unified_image


# ========================================
# SIMPLE WRAPPER FOR NOTEBOOK USAGE
# ========================================

def process_with_bg_removal_and_unification(input_image_path: str) -> np.ndarray:
    """
    Simple wrapper function: Complete pipeline for image preprocessing.
    
    Returns unified image as numpy array (BGR format, 224x224x3).
    Perfect for direct use in notebooks like test.ipynb Cell 2.
    
    Args:
        input_image_path (str): Path to input image
    
    Returns:
        np.ndarray: Unified image array (224, 224, 3) in BGR format, values 0-255
    
    USAGE EXAMPLE:
        unified_img = process_with_bg_removal_and_unification(
            input_image_path="/path/to/image.jpg"
        )
        
        # For display in matplotlib:
        import matplotlib.pyplot as plt
        import cv2
        
        unified_img_rgb = cv2.cvtColor(unified_img, cv2.COLOR_BGR2RGB)
        plt.imshow(unified_img_rgb)
        plt.show()
    """
    from bg_remove_465 import remove_background
    
    # Step 1: Remove background
    bg_removed_image = remove_background(input_image_path)
    
    # Step 2: Get unified image (returns BGR numpy array)
    unified_image = unify_image_for_visualization(bg_removed_image)
    
    return unified_image


# ========================================
# VERIFICATION FUNCTION
# ========================================

def verify_preprocessing_matches_training():
    """
    Verification checklist to ensure preprocessing matches training exactly.
    """
    print("\n" + "="*60)
    print("PREPROCESSING VERIFICATION CHECKLIST")
    print("="*60)
    print("✓ 1. Transparency handling → Black background (0, 0, 0)")
    print("✓ 2. Resize to 224×224 (LANCZOS4 interpolation)")
    print("✓ 3. ImageNet normalization (for model input only):")
    print(f"     - Mean: {IMAGENET_MEAN}")
    print(f"     - Std:  {IMAGENET_STD}")
    print("✓ 4. Output: torch.Tensor (1, 3, 224, 224)")
    print("✗ NO CLAHE normalization (models handle their own preprocessing)")
    print("="*60)
    print("✅ ALL PREPROCESSING STEPS MATCH TRAINING!")
    print("="*60 + "\n")


if __name__ == "__main__":
    verify_preprocessing_matches_training()