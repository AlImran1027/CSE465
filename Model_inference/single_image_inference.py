"""
Simple Single Image Inference Pipeline
For Plant Disease Detection
Using DenseNet201 Multi-Task Model

Project: CSE465 - Eggplant, Potato, Tomato Disease Classification
Author: AlImran2122071
Date: 2025-12-17
"""

import torch
import torch.nn as nn
from torchvision.models import densenet201, DenseNet201_Weights
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from bg_remove_465 import remove_background
from image_unifier_v2 import unify_image_for_model

# ========================================
# CONFIGURATION
# ========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                      else "cpu")
MODEL_PATH = "/Users/alimran/Desktop/CSE465/best_DenseNet201.pt"

# Label Mappings (matching DenseNet201-465.ipynb)
SPECIES_MAP = {0: "Eggplant", 1: "Potato", 2: "Tomato"}
HEALTH_MAP = {0: "Bacterial", 1: "Fungal", 2: "Healthy", 3: "Virus"}

# ========================================
# MODEL DEFINITION
# ========================================

class MultiTaskDenseNet201(nn.Module):
    """Multi-task DenseNet201 for species and disease classification"""
    def __init__(self, num_species=3, num_health=4, pretrained=False, dropout=0.3):
        super().__init__()
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

# ========================================
# LOAD MODEL
# ========================================

def load_model():
    """Load trained DenseNet201 multi-task model"""
    print("="*70)
    print("Loading DenseNet201 Multi-Task Model...")
    print("="*70)
    model = MultiTaskDenseNet201(num_species=3, num_health=4, pretrained=False, dropout=0.3)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        model.to(DEVICE)
        model.eval()  # This disables dropout and batch norm
        
        # Verify model is in eval mode
        if not model.training:
            print(f"✓ Model loaded successfully!")
            print(f"  Device: {DEVICE}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            if 'val_health' in checkpoint:
                print(f"  Val Health Accuracy: {checkpoint['val_health']:.4f}")
            print(f"  Eval mode: Active (dropout disabled)")
        else:
            print(f"⚠ Warning: Model not in eval mode!")
        print("="*70 + "\n")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

# ========================================
# PREDICT
# ========================================

def predict(image_path: str) -> Dict:
    """
    Predict species and disease status from leaf image.
    
    Args:
        image_path (str): Path to leaf image
    
    Returns:
        dict: Predictions with probabilities and all class scores
    """
    print(f"\nProcessing: {Path(image_path).name}")
    
    # 1. Remove background
    bg_removed = remove_background(image_path)
    
    # 2. Preprocess
    input_tensor = unify_image_for_model(bg_removed).to(DEVICE)
    
    # 3. Predict
    with torch.no_grad():
        logits_species, logits_health = MODEL(input_tensor)
    
    # Get probabilities
    probs_species = torch.softmax(logits_species, dim=1).cpu().numpy()[0]
    probs_health = torch.softmax(logits_health, dim=1).cpu().numpy()[0]
    
    # Get predictions
    pred_species_idx = logits_species.argmax(dim=1).item()
    pred_health_idx = logits_health.argmax(dim=1).item()
    
    # Build results
    results = {
        "species": SPECIES_MAP[pred_species_idx],
        "health": HEALTH_MAP[pred_health_idx],
        "confidence": {
            "species": float(probs_species[pred_species_idx]),
            "health": float(probs_health[pred_health_idx])
        },
        "diagnosis": f"{SPECIES_MAP[pred_species_idx]}_{HEALTH_MAP[pred_health_idx]}",
        "all_probabilities": {
            "species": {SPECIES_MAP[i]: float(probs_species[i]) for i in range(len(probs_species))},
            "health": {HEALTH_MAP[i]: float(probs_health[i]) for i in range(len(probs_health))}
        }
    }
    
    # Print results
    print(f"\n{'RESULTS':^70}")
    print("="*70)
    print(f"Diagnosis: {results['diagnosis']}")
    print(f"Species:   {results['species']} ({results['confidence']['species']*100:.1f}%)")
    print(f"Health:    {results['health']} ({results['confidence']['health']*100:.1f}%)")
    print("="*70 + "\n")
    
    return results

# ========================================
# MAIN
# ========================================

# Load model once (global)
MODEL = load_model()

if __name__ == "__main__":
    # Single image inference
    test_image = "/Users/alimran/Desktop/CSE465/Model_inference/fungal (9).jpg"
    
    result = predict(test_image)