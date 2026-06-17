# brain_tumor_classification.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTModel, ViTConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
class Config:
    # Dataset paths
    data_dir = "dataset"
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")
    
    # Model configuration
    model_name = "google/vit-base-patch16-224"
    num_classes = 4
    img_size = 224
    
    # Training hyperparameters
    batch_size = 16
    num_epochs = 20
    learning_rate = 3e-5
    weight_decay = 0.01
    num_workers = 2
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform

# Create data loaders
def create_dataloaders():
    train_transform, test_transform = get_transforms()
    
    # Load datasets
    train_dataset = ImageFolder(Config.train_dir, transform=train_transform)
    test_dataset = ImageFolder(Config.test_dir, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.classes

# Vision Transformer Model
class ViTForImageClassification(nn.Module):
    def __init__(self, num_classes=4):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(Config.model_name)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate_model(model, test_loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(test_loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return {
        'loss': epoch_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Main function
def main():
    print(f"Using device: {Config.device}")
    
    # Create data loaders
    train_loader, test_loader, class_names = create_dataloaders()
    print(f"Class names: {class_names}")
    
    # Initialize model
    model = ViTForImageClassification(num_classes=len(class_names))
    model = model.to(Config.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    # Training loop
    best_accuracy = 0.0
    
    print("Starting training...")
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
        print("-" * 10)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.device)
        
        # Evaluate on test set
        test_metrics = evaluate_model(
            model, test_loader, criterion, Config.device, class_names)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # Save the best model
        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved with accuracy: {best_accuracy:.4f}")
    
    print("Training complete!")
    print(f"Best Test Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()