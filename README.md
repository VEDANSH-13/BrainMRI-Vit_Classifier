# Brain Tumor Classification using Vision Transformer

A deep learning project that classifies brain tumor types using a Vision Transformer (ViT) model. The system can identify four different tumor types from MRI images.

## 🧠 Tumor Classes

- **Glioma Tumor**
- **Meningioma Tumor** 
- **No Tumor**
- **Pituitary Tumor**

## 📁 Project Structure

```
Transformer/
├── dataset/
│   ├── Training/
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── no_tumor/
│   │   └── pituitary_tumor/
│   └── Testing/
│       ├── glioma_tumor/
│       ├── meningioma_tumor/
│       ├── no_tumor/
│       └── pituitary_tumor/
├── brain_tumor_classification.py    # Training script
├── best_model.pth                   # Trained model weights
├── confusion_matrix.png             # Evaluation visualization
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn pillow tqdm
```

### 2. Prepare Dataset

Organize your MRI images in the following structure:

```
dataset/
├── Training/
│   ├── glioma_tumor/     # Training images for glioma
│   ├── meningioma_tumor/ # Training images for meningioma
│   ├── no_tumor/         # Training images for no tumor
│   └── pituitary_tumor/  # Training images for pituitary
└── Testing/
    ├── glioma_tumor/     # Testing images for glioma
    ├── meningioma_tumor/ # Testing images for meningioma
    ├── no_tumor/         # Testing images for no tumor
    └── pituitary_tumor/  # Testing images for pituitary
```

### 3. Train the Model

```bash
python brain_tumor_classification.py
```

The training will:
- Load and preprocess the dataset
- Train a Vision Transformer model
- Save the best model as `best_model.pth`
- Generate a confusion matrix

## 🛠️ Model Architecture

### Vision Transformer (ViT) Configuration
- **Base Model**: `google/vit-base-patch16-224`
- **Input Size**: 224x224 pixels
- **Patch Size**: 16x16 pixels
- **Hidden Size**: 768
- **Number of Layers**: 12
- **Number of Attention Heads**: 12

### Custom Classification Head
- **Linear Layer**: Maps ViT output to 4 tumor classes
- **Output**: 4-dimensional logits for tumor classification

## 📊 Training Configuration

```python
# Hyperparameters
batch_size = 16
num_epochs = 20
learning_rate = 3e-5
weight_decay = 0.01
img_size = 224
```

### Data Augmentation
- Random horizontal flip
- Random rotation (±15 degrees)
- ImageNet normalization

## 📈 Performance Metrics

The model evaluation includes:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1-Score**: Weighted F1 score
- **Confusion Matrix**: Visual representation of predictions

## 🔧 Technical Details

### Image Preprocessing
1. **Resize**: 224x224 pixels
2. **Normalization**: ImageNet standards
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

### Device Support
- **CUDA**: Automatically uses GPU if available
- **CPU**: Fallback to CPU processing

### Model Saving
- Best model saved based on test accuracy
- PyTorch `.pth` format
- Contains model state dict only

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'torch'**
   ```bash
   pip install torch torchvision
   ```

2. **CUDA out of memory**
   - Reduce batch size in Config class
   - Use CPU instead: `device = torch.device("cpu")`

3. **Corrupted model file**
   - Delete `best_model.pth` and retrain
   - Check if training completed successfully

4. **Dataset not found**
   - Verify dataset directory structure
   - Check image file formats (.jpg, .png, .bmp)

### Python Version Compatibility
- **Recommended**: Python 3.8+
- **Tested**: Python 3.12
- **Not compatible**: Python 3.13 (use 3.12 instead)

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=8.0.0
tqdm>=4.60.0
numpy>=1.21.0
```

## 🎯 Model Performance

Expected performance metrics (based on training):
- **Test Accuracy**: ~90-95%
- **Precision**: ~0.90-0.95
- **Recall**: ~0.90-0.95
- **F1-Score**: ~0.90-0.95

*Note: Actual performance may vary based on dataset quality and size*

## 🔬 Medical Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📝 License

This project is provided for educational purposes. Please ensure compliance with relevant medical data regulations and ethical guidelines when using patient data.

## 🤝 Contributing

Feel free to contribute improvements:
- Add new tumor types
- Implement different architectures
- Improve data augmentation
- Add evaluation metrics

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify dataset structure
3. Ensure all dependencies are installed
4. Check Python version compatibility
