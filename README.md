<div align="center">

# 🌱 PlantSense — Multitask Plant Disease Classifier

**A deep learning framework that simultaneously identifies plant species and diagnoses leaf diseases — benchmarked across 18 architectures, with knowledge distillation and explainable AI.**

*CSE465 · Pattern Recognition and Neural Network · North South University · Fall 2025*

[Results](#-results) · [Quick Start](#-quick-start) · [Architecture](#-model-architecture) · [Explainability](#-explainable-ai) · [Citation](#-citation)

</div>

---

## 🧭 Overview

Crop diseases caused by fungal, bacterial, and viral pathogens reduce yields by **20–40% annually**, costing billions in losses and threatening food security. Manual field scouting is slow, inconsistent, and inaccessible to smallholder farmers.

**PlantSense** addresses this with a single unified model that:

- Identifies **which crop** (Eggplant / Potato / Tomato) — **97.98% accuracy**
- Diagnoses **what's wrong** (Bacterial / Fungal / Healthy / Virus) — **86.33% accuracy**
- Explains **why** it made that decision using Grad-CAM++ and LIME
- Runs efficiently on constrained devices via Knowledge Distillation

> Developed as part of CSE465 at North South University under the supervision of **Dr. Sifat Momen**.

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| 🎯 Multitask Learning | Joint species + disease classification with a shared backbone |
| 🏆 Best Accuracy | DenseNet201 — 86.33% health, 97.98% species |
| 📦 Model Compression | 4.5× smaller via Knowledge Distillation |
| 🔍 Explainable AI | Grad-CAM++ heatmaps + LIME decision boundaries |
| ⚙️ 18 Architectures | 10 CNNs + 8 Vision Transformers benchmarked |
| 🌐 Web Deployment | End-to-end inference pipeline with background removal |

---

## 📂 Dataset

The dataset was aggregated from **multiple public sources** (Mendeley Data, Kaggle) covering three solanaceous crops under real-world and field conditions.

**12 Categories** = 3 Species × 4 Health States

| Task | Classes |
|------|---------|
| **Species** | Eggplant · Potato · Tomato |
| **Health** | Bacterial · Fungal · Healthy · Virus |

**Preprocessing Pipeline:**

```
Raw Images → RGB Conversion → Background Removal (BiRefNet RMBG v2.0)
           → Black Background → 224×224 Resize → ImageNet Normalization
```

**Dataset Splits:**

| Split | Ratio | Augmentation |
|-------|-------|--------------|
| Train | 70% | ✅ Adaptive per-class balancing to 1000 samples |
| Validation | 15% | ❌ |
| Test | 15% | ❌ |

Duplicate detection was performed using **perceptual hashing (pHash)** to prevent data leakage across splits.

**Augmentation Operations** *(Albumentations)*: Horizontal/vertical flip · Shift-scale-rotate · Brightness/contrast · Hue-saturation · Noise injection · Blur · Coarse dropout

---

## 🏗️ Model Architecture

The core model uses a **shared DenseNet backbone** with two task-specific classification heads:

```
Input Image (224×224)
       │
  ┌────▼────────────────────┐
  │   Shared DenseNet       │  ← ImageNet pretrained weights
  │   Backbone (1024-dim)   │
  └────────────┬────────────┘
               │
         Dropout (p=0.3)
               │
       ┌───────┴───────┐
       ▼               ▼
  Species Head     Health Head
  (3 classes)      (4 classes)
  Eggplant         Bacterial
  Potato           Fungal
  Tomato           Healthy
                   Virus
```

**Multitask Loss:**
```
L_total = L_species + λ · L_health      (λ = 1)
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1×10⁻⁴ |
| Weight Decay | 5×10⁻⁴ |
| Batch Size | 32 |
| Scheduler | CosineAnnealingLR |
| Gradient Clipping | 1.0 |
| Early Stopping | 3–5 epochs |
| Mixed Precision | ✅ AMP |

---

## 📊 Results

> All results reported as **mean ± std** across 3 random seeds (42, 123, 456).

### CNN Architectures — Health Classification

| Model | Accuracy (%) | Macro F1 (%) |
|-------|:-----------:|:------------:|
| **DenseNet201** ⭐ | **86.33 ± 0.42** | **87.24 ± 0.38** |
| EfficientNetV2-S | 86.25 ± 0.51 | 87.07 ± 0.47 |
| EfficientNetV2-L | 85.47 ± 0.49 | 86.31 ± 0.45 |
| DenseNet121 | 85.62 ± 0.45 | 86.52 ± 0.41 |
| InceptionV3 | 84.95 ± 0.63 | 85.88 ± 0.58 |
| Xception | 83.81 ± 0.55 | 84.75 ± 0.52 |
| ResNet152 | 83.56 ± 0.48 | 84.28 ± 0.44 |
| ResNet50 | 82.80 ± 0.67 | 83.54 ± 0.62 |
| ResNet101 | 82.30 ± 0.71 | 82.99 ± 0.65 |
| DenseNet264 | 82.21 ± 0.58 | 83.32 ± 0.54 |

### Vision Transformer Architectures — Health Classification

| Model | Accuracy (%) | Macro F1 (%) |
|-------|:-----------:|:------------:|
| **Swin-T** ⭐ | **84.74 ± 0.52** | **85.32 ± 0.48** |
| DeiT-S | 83.88 ± 0.61 | 84.13 ± 0.57 |
| Swin-V2-Large | 83.12 ± 0.55 | 83.89 ± 0.51 |
| Swin-B | 82.67 ± 0.58 | 83.29 ± 0.54 |
| DeiT-B | 82.51 ± 0.65 | 83.14 ± 0.59 |
| ViT-Large | 81.45 ± 0.69 | 82.23 ± 0.64 |
| ViT-Base | 80.99 ± 0.73 | 81.76 ± 0.68 |
| Efficient-ViT | 75.06 ± 0.82 | 75.83 ± 0.76 |

### CNN vs. ViT — Head-to-Head

| Paradigm | Best Model | Accuracy (%) | Macro F1 (%) | Parameters |
|----------|-----------|:------------:|:------------:|:----------:|
| **CNN** | DenseNet201 | **86.33** | **87.24** | ~20M |
| ViT | Swin-T | 84.74 | 85.32 | ~28M |

> CNNs outperform ViTs by ~1.6% with 28% fewer parameters. DenseNet's dense feature reuse excels at detecting localized, subtle disease patterns on leaves.

### Lightweight Student Models

| Model | Accuracy (%) | Macro F1 (%) |
|-------|:-----------:|:------------:|
| EfficientNet-B0 | 81.83 ± 0.18 | 81.29 ± 0.05 |
| MobileNetV2 | 77.63 ± 0.45 | 78.43 ± 0.43 |
| MobileNetV3-S | 75.53 ± 0.56 | 76.51 ± 0.45 |

### Ablation Study — EfficientNet-B0 with KD

| Configuration | Aug | Tuning | KD | Accuracy (%) | Macro F1 (%) |
|---------------|:---:|:------:|:--:|:------------:|:------------:|
| **Full Pipeline** | ✅ | ✅ | ✅ | **84.32 ± 0.11** | **84.47 ± 0.31** |
| No KD | ✅ | ✅ | ❌ | 81.83 ± 0.18 | 81.29 ± 0.05 |
| No Augmentation | ❌ | ✅ | ✅ | 78.41 ± 0.54 | 78.23 ± 0.67 |
| No HP Tuning | ✅ | ❌ | ✅ | 78.92 ± 0.29 | 78.95 ± 0.15 |
| Base Pipeline | ❌ | ❌ | ❌ | 71.89 ± 0.42 | 71.45 ± 0.35 |

---

## 🧠 Knowledge Distillation

DenseNet201 (teacher) transfers its learned knowledge to EfficientNet-B0 (student) via a composite training objective:

```
L_student = α · L_supervised + (1 − α) · T² · L_KD
```

- `L_KD` — KL Divergence between teacher and student soft distributions at temperature T
- `L_supervised` — Cross-entropy with ground-truth hard labels
- `α` — balances teacher guidance vs. ground-truth supervision
- `T²` — compensates for gradient scaling from temperature softening

**Result:** 4.5× model compression, retaining 84.47% accuracy (vs. teacher's 86.33%).

---

## 🔍 Explainable AI

The model's decisions are validated using two complementary XAI techniques:

**Grad-CAM++** highlights what the model looks at:
- **Bacterial** → lesion spots and necrotic areas
- **Fungal** → discoloration and spreading patches
- **Viral** → mosaic and mottling patterns
- **Healthy** → uniform green tissue

**LIME** provides local, model-agnostic explanations showing which superpixels most influenced each individual prediction.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch 2.0+

### Installation

```bash
git clone https://github.com/AlImran1027/CSE465.git
cd CSE465

pip install torch torchvision transformers timm kornia
pip install opencv-python pillow matplotlib seaborn scikit-learn
pip install lime albumentations
```

### Inference

```python
from Model_inference.single_image_inference import predict

result = predict("leaf.jpg")
# Output:
# {
#   "species": "Tomato",
#   "health":  "Bacterial",
#   "confidence": { "species": 0.98, "health": 0.91 }
# }
```

### Training

```bash
# Best teacher model
jupyter lab CNN/DenseNet201-465.ipynb

# Knowledge distillation
jupyter lab KD/KD_model.ipynb
```

### Explainability

```bash
cd XAI && python xai_interpretability_465.py
# Saves Grad-CAM++ heatmaps and LIME visualizations to ./outputs/
```

---

## 📁 Project Structure

```
CSE465/
├── CNN/                        # 10 CNN architectures
│   └── DenseNet201-465.ipynb   # Best performing model
├── Vision_Transformers/        # 8 ViT architectures (ViT, DeiT, Swin, Efficient-ViT)
├── Student_Models/             # Lightweight models (MobileNet, EfficientNet-B0)
├── KD/                         # Knowledge Distillation
│   └── KD_model.ipynb
├── Model_inference/            # Production inference pipeline
│   ├── bg_remove_465.py        # BiRefNet background removal
│   ├── image_unifier_v2.py     # Preprocessing
│   └── single_image_inference.py
├── XAI/                        # Explainability
│   └── xai_interpretability_465.py
└── Augmentation_465.ipynb      # Data augmentation experiments
```

---

## ⚠️ Limitations

- Covers three solanaceous crops only — generalization to other plant families is untested
- Performance may degrade on heavily occluded or very low-resolution field images
- Background removal adds ~1–2s latency to the inference pipeline
- ViT models require significantly more compute for comparable accuracy on this dataset size

---

## 🔭 Future Work

- Expand coverage to more crop species and disease categories
- Export to TFLite / ONNX for real-time mobile inference
- Integrate an active learning loop for continuous improvement with field-collected data
- Add disease **severity estimation** (mild / moderate / severe) alongside detection

---

## 👥 Team

| Name | Student ID |
|------|-----------|
| Al Imran | 2122071042 |
| Shoumik Sarker | 2211320042 |
| Md Rafiqul Islam Rana | 2132344642 |
| Sumon Das | 2211834642 |

**Faculty Advisor:** Dr. Sifat Momen, Professor — Dept. of Electrical & Computer Engineering, North South University

---

## 📄 Citation

```bibtex
@misc{imran2025plantsense,
  title  = {A Multitask Deep Learning Framework for Plant Species Identification
            and Leaf Disease Classification},
  author = {Al Imran and Shoumik Sarker and Md Rafiqul Islam Rana and Sumon Das},
  year   = {2025},
  note   = {CSE465 Course Project, North South University},
  url    = {https://github.com/AlImran1027/CSE465}
}
```

---

## 📚 References

- Mohanty et al. (2016) — Deep learning for image-based plant disease detection
- Hinton et al. (2015) — Distilling the knowledge in a neural network
- Chattopadhay et al. (2018) — Grad-CAM++: Generalized gradient-based visual explanations
- Ribeiro et al. (2016) — LIME: Why should I trust you?
- Dosovitskiy et al. (2021) — An image is worth 16×16 words (ViT)
- Huang et al. (2017) — Densely connected convolutional networks

---

<div align="center">
<sub>Built with PyTorch · North South University · Fall 2025</sub>
</div>
