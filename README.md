# Plant Disease Classification using Multi-Task Learning

This project implements multi-task deep learning models for simultaneous plant species identification and disease classification using DenseNet architectures.

## 📋 Project Overview

The system performs two classification tasks simultaneously:
1. **Species Classification**: Identifies the plant type (Eggplant, Potato, Tomato)
2. **Disease Classification**: Detects disease status (Bacterial, Fungal, Healthy, Virus)

## 🗂️ Dataset Structure

The `Split_Dataset` directory contains images organized by plant species and health status:

```
Split_Dataset/
├── train/
│   ├── Eggplant_Bacterial/
│   ├── Eggplant_Fungal/
│   ├── Eggplant_Healthy/
│   ├── Eggplant_Virus/
│   ├── Potato_Bacterial/
│   ├── Potato_Fungal/
│   ├── Potato_Healthy/
│   ├── Potato_Virus/
│   ├── Tomato_Bacterial/
│   ├── Tomato_Fungal/
│   ├── Tomato_Healthy/
│   └── Tomato_Virus/
├── val/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

**Label Mappings:**
- **Species**: `{eggplant: 0, potato: 1, tomato: 2}`
- **Health**: `{bacterial: 0, fungal: 1, healthy: 2, virus: 3}`

## 🧠 Models

### DenseNet121
- **File**: `Code files/DenseNet121-465.ipynb`
- **Architecture**: DenseNet121 with multi-task heads
- **Parameters**: ~7M trainable parameters
- **Model Output**: `best_multitask_DenseNet121.pt`

### DenseNet201
- **File**: `Code files/DenseNet201-465.ipynb`
- **Architecture**: DenseNet201 with multi-task heads
- **Parameters**: ~18M trainable parameters
- **Model Output**: `best_multitask_DenseNet201.pt`

## ⚙️ Configuration

Both models use the following configuration:

| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.3 |
| Epochs | 10 (max) |
| Early Stopping | 3 epochs patience |
| Loss Function | CrossEntropyLoss (both tasks) |

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install pandas numpy
pip install matplotlib seaborn
pip install scikit-learn
pip install pillow opencv-python
pip install tqdm
```

### Running the Models

1. **Open Jupyter Notebook/Lab or VS Code**
   ```bash
   jupyter lab
   # or
   code .
   ```

2. **Select a model notebook**:
   - `DenseNet121-465.ipynb` for the lighter model
   - `DenseNet201-465.ipynb` for the heavier model

3. **Run cells sequentially**:
   - Cell 1: Imports
   - Cell 2: Visualization setup
   - Cell 3: Data loading and configuration
   - Cell 4: Model definition
   - Cell 5: Training loop
   - Cell 6: Testing and evaluation
   - Cell 7: Plot generation
   - Cell 8: Comprehensive metrics
   - Cell 9: Sample visualizations

## 📊 Output Files

After training, the following files will be generated:

### Model Checkpoints
- `best_multitask_DenseNet121.pt` / `best_multitask_DenseNet201.pt`
- `final_multitask_DenseNet121.pt` / `final_multitask_DenseNet201.pt`

### Training Plots
- `plot_train_loss.png` - Training loss over epochs
- `plot_val_loss.png` - Validation loss over epochs
- `plot_loss_comparison.png` - Train vs Val loss comparison
- `plot_species_accuracy.png` - Species classification accuracy
- `plot_health_accuracy.png` - Disease classification accuracy
- `plot_all_metrics.png` - All metrics combined

### Evaluation Results
- `confusion_matrix_species.png` - Species classification confusion matrix
- `confusion_matrix_health.png` - Disease classification confusion matrix
- `sample_predictions.png` - Sample predictions visualization (30 images)

## 📈 Model Architecture

```
Input (224×224×3)
    ↓
DenseNet Backbone (121 or 201 layers)
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
    ├─→ Species Head → [3 classes]
    └─→ Health Head → [4 classes]
```

## 🔬 Training Features

- **Multi-Task Learning**: Joint training for species and disease classification
- **Data Augmentation**: Pre-applied offline augmentation
- **ImageNet Normalization**: Standard normalization for pretrained models
- **Mixed Precision Training**: Faster training with AMP (if CUDA available)
- **Gradient Clipping**: Prevents gradient explosion (max norm: 1.0)
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Cosine annealing for better convergence

## 📋 Evaluation Metrics

The notebooks compute and display:
- Overall accuracy (species and health separately)
- Per-class precision, recall, F1-score
- Confusion matrices with visualizations
- Classification reports

## 🎯 Use Cases

- Agricultural disease diagnosis
- Automated plant health monitoring
- Research in plant pathology
- Educational tools for plant disease identification

## 📝 Notes

- The notebooks use **CPU/MPS/CUDA** automatically based on availability
- Set `PRETRAINED = True` in configuration cell to use ImageNet pretrained weights
- Adjust `EPOCHS` and `BATCH_SIZE` based on your computational resources
- The dataset includes augmented images (identified by `_aug_` in filenames)

## 🔧 Customization

To modify the training:

1. **Change hyperparameters**: Edit the configuration cell (Cell 3)
2. **Adjust model architecture**: Modify the model definition cell (Cell 4)
3. **Add new metrics**: Update the comprehensive testing function (Cell 8)
4. **Change visualization style**: Modify the TrainingLogger class (Cell 2)

## 📄 Files

```
CSE465/
├── README.md                          # This file
├── Code files/
│   ├── DenseNet121-465.ipynb         # DenseNet121 implementation
│   └── DenseNet201-465.ipynb         # DenseNet201 implementation
├── Split_Dataset/                     # Dataset directory
│   ├── train/                        # Training set
│   ├── val/                          # Validation set
│   └── test/                         # Test set
└── [Generated outputs after training]
```

## 🤝 Contributing

This is a course project (CSE465). Feel free to experiment with different architectures, hyperparameters, or augmentation strategies.

## 📧 Contact

For questions or issues related to this project, please refer to the course materials or contact the instructor.

---

**Last Updated**: December 2025  
**Course**: CSE465  
**Models**: DenseNet121, DenseNet201  
**Framework**: PyTorch
