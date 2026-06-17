# Dataset Setup

## Download Full Dataset

The complete brain tumor MRI dataset is too large for GitHub. Please download it from:

### Option 1: Kaggle Dataset
[Link to your Kaggle dataset]

### Option 2: Google Drive
[Link to your Google Drive folder]

### Option 3: Direct Download
```
wget [dataset_url]
unzip brain_tumor_dataset.zip
```

## Directory Structure

After downloading, organize as follows:

```
dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Sample Images

This repository includes a few sample images in `dataset_samples/` for testing the model structure.

## Dataset Statistics

- **Total Images**: ~7,000 MRI scans
- **Training**: ~5,500 images
- **Testing**: ~1,500 images
- **Image Format**: JPEG/PNG
- **Image Size**: Variable (resized to 224x224)
- **Classes**: 4 tumor types

## Data Source

[Information about where the dataset came from]
