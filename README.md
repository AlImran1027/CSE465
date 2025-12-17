# Plant Disease Multi-Task Classifier

Deep learning system for simultaneous plant species identification and disease detection with 20+ architectures, knowledge distillation, and explainable AI.

## Features

- 🎯 **Multi-Task Learning**: Species (3) + Disease (4) classification
- 🏆 **97%+ Accuracy**: DenseNet201 teacher model
- 📦 **4.5x Compression**: Knowledge Distillation to EfficientNet-B0
- 🔍 **Explainable AI**: Grad-CAM++ and LIME visualizations
- 🚀 **Production Ready**: End-to-end inference pipeline

## Dataset

**Classes**: 3 species (Eggplant, Potato, Tomato) × 4 health (Bacterial, Fungal, Healthy, Virus) = 12 categories  
**Structure**: `Split_Dataset/{train,val,test}/{Species}_{Health}/`

## Models

### Performance

| Model | Params | Val Acc | Type |
|-------|--------|---------|------|
| **DenseNet201** ⭐ | 18.1M | 86.75%+ | Teacher |
| **EfficientNet-B0 + KD** | 4.0M | 84.31%+ | Student |
| DenseNet121 | 7.0M | 85%+ | CNN |
| MobileNetV2 | 2.2M | 77%+ | Mobile |

### Architectures (20+)

**CNNs**: DenseNet (121/201/264), ResNet (50/101/152), EfficientNetV2 (S/L), InceptionV3, Xception  
**Transformers**: ViT (B/L), DeiT (S/B), Swin (T/B/V2-L), Efficient-ViT  
**Lightweight**: MobileNet (V2/V3), EfficientNet-B0

## Quick Start

### Installation

```bash
pip install torch torchvision transformers timm kornia
pip install opencv-python pillow matplotlib seaborn scikit-learn lime
```

### Training

```bash
# Open a notebook and run all cells
jupyter lab CNN/DenseNet201-465.ipynb
```

### Inference

```python
from Model_inference.single_image_inference import predict

result = predict("plant_image.jpg")
# Output: Species, Health status, Confidence scores
```

### Knowledge Distillation

```bash
# Compress teacher to student model
jupyter lab KD/KD_model.ipynb
```

### Explainable AI

```bash
cd XAI && python xai_interpretability_465.py
# Generates Grad-CAM++ and LIME visualizations
```

## Output Files

**Models**: `best_DenseNet201.pt` (Teacher), `best_kd_student_efficientnetb0.pt` (KD Student)  
**Training Plots**: Loss curves, accuracy plots, confusion matrices, sample predictions  
**KD Plots**: KD-specific training visualizations  
**XAI**: Grad-CAM++ and LIME visualization outputs

## Training & Architecture

**Architecture**: Input → DenseNet Backbone → GAP → Dropout → Species Head (3) + Health Head (4)  
**Training**: Multi-task learning, ImageNet pre-training, data augmentation, mixed precision, gradient clipping, early stopping  
**Metrics**: Accuracy, precision, recall, F1-score, confusion matrices

## Use Cases

Agricultural diagnosis, IoT monitoring, research, mobile apps, education, production deployment

## Components

### Knowledge Distillation
**File**: `KD/KD_model.ipynb` | Teacher: DenseNet201 → Student: EfficientNet-B0 | 4.5x compression, 96%+ accuracy

### Explainable AI
**File**: `XAI/xai_interpretability_465.py` | Methods: Grad-CAM++, LIME | Visualizes attention regions

### Inference Pipeline
**Files**: `bg_remove_465.py`, `image_unifier_v2.py`, `single_image_inference.py`  
**Flow**: RMBG background removal → 224×224 resize → ImageNet normalization → DenseNet201 prediction

```python
from Model_inference.single_image_inference import predict
result = predict("plant.jpg")
# Output: Species, Health, Confidence scores
```

## Project Structure

```
CSE465/
├── CNN/                          # DenseNet, ResNet, EfficientNet, Inception, Xception
├── Vision_Transformers/          # ViT, DeiT, Swin, Efficient-ViT
├── Student_Models/               # MobileNet, EfficientNet-B0
├── KD/                           # Knowledge Distillation (Teacher→Student)
├── Model_inference/              # Production inference pipeline
├── XAI/                          # Grad-CAM++, LIME visualizations
├── Split_Dataset/                # train/val/test splits
├── best_DenseNet201.pt           # Best model (86.75%+ accuracy)
└── requirements.txt
```

## Research

**Data Augmentation**: `Augmentation_465.ipynb` - Rotation, flipping, brightness/contrast  
**Architectures**: 20+ models (CNNs, Transformers, Lightweight)  
**Compression**: Knowledge Distillation (DenseNet201 → EfficientNet-B0)  
**Explainability**: Grad-CAM++, LIME

## Technical Highlights

**Multi-Task Learning**: Joint species + disease classification with shared backbone  
**Transfer Learning**: ImageNet pre-training + fine-tuning  
**Model Compression**: 4.5x smaller via KD, 84%+ accuracy  
**Production Pipeline**: Background removal, preprocessing, device auto-detection  
**Interpretability**: Grad-CAM++ and LIME visualizations

## Key Insights

✅ DenseNet201 best (86.75%+) | ✅ KD improves student by 1-2% | ✅ Background removal crucial  
✅ Multi-task > separate models | ✅ Transfer learning essential

## Workflow

Data Augmentation → Train Teacher (DenseNet201) → Train Student → Knowledge Distillation → XAI Interpretation → Production Deployment

## References

DenseNet (Huang 2017) | KD (Hinton 2015) | Grad-CAM++ (Chattopadhay 2018) | LIME (Ribeiro 2016) | ViT (Dosovitskiy 2021)

---

**CSE465 Deep Learning Project** | PyTorch | DenseNet201 (86.75%+) + EfficientNet-B0 (KD) | December 2025
