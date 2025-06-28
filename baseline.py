import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train and Compare Baseline CNN Models')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of fish classes (default: 5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Dataset/EnhancedDataset',
                        help='Path to the dataset directory (default: Dataset/EnhancedDataset)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for training (default: 256)')
    
    # Normalization parameters
    parser.add_argument('--mean', type=float, nargs=3, default=[0.5, 0.5, 0.5],
                        help='Mean values for normalization (default: [0.5, 0.5, 0.5])')
    parser.add_argument('--std', type=float, nargs=3, default=[0.5, 0.5, 0.5],
                        help='Std values for normalization (default: [0.5, 0.5, 0.5])')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='baseline_results',
                        help='Directory to save results (default: baseline_results)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda) (default: auto)')
    
    return parser.parse_args()

# ======================
# Model Definitions
# ======================

class LinearBaseline(nn.Module):
    """Simple linear classifier baseline"""
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3*256*256, num_classes)

    def forward(self, x):
        return self.linear(self.flatten(x))

class ShallowVGG(nn.Module):
    """Shallow VGG-like architecture"""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(16*128*128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class MNISTLeNet(nn.Module):
    """LeNet-5 inspired architecture adapted for RGB images"""
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*61*61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ======================
# Utility Functions
# ======================

def create_transforms(args):
    """Create training and validation transforms"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ])
    
    return transform_train, transform_valid

def load_datasets(args, transform_train, transform_valid):
    """Load training, validation, and test datasets"""
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    test_dir = os.path.join(args.data_dir, 'test')
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}. Using validation set for testing.")
        test_dir = valid_dir
    
    # Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform_valid)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_valid)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(valid_dataset)} validation samples")
    print(f"Loaded {len(test_dataset)} test samples")
    print(f"Class names: {train_dataset.classes}")
    
    return train_loader, valid_loader, test_loader, train_dataset.classes

def get_device(device_arg):
    """Get the appropriate device based on argument"""
    if device_arg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    return device

def validate_model(model, valid_loader, criterion, device):
    """Validation function"""
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    avg_loss = valid_loss / len(valid_loader)
    
    return avg_loss, accuracy

def test_model(model, test_loader, device):
    """Test function"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    return accuracy

def train_model(model, optimizer, train_loader, valid_loader, test_loader, device, num_epochs=50, model_name="Model"):
    """Universal training function for all baseline models"""
    criterion = nn.CrossEntropyLoss()
    
    train_accuracies = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        # Calculate training metrics
        train_accuracy = 100 * correct / total
        train_loss = epoch_loss / len(train_loader)
        
        # Validation phase
        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
        
        # Store metrics
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Print progress every 10 epochs or last epoch
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_accuracy:.2f}%, Valid Acc: {valid_accuracy:.2f}%")
    
    # Final test evaluation
    test_accuracy = test_model(model, test_loader, device)
    
    print(f"\n{model_name} Training Complete!")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Validation Accuracy: {valid_accuracies[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    return {
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'test_accuracy': test_accuracy,
        'model_name': model_name
    }

def plot_comparison_results(results, save_dir):
    """Plot comparison of all models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['blue', 'red', 'green']
    epochs = range(1, len(results[0]['train_accuracies']) + 1)
    
    # Plot training accuracies
    for i, result in enumerate(results):
        ax1.plot(epochs, result['train_accuracies'], 
                color=colors[i], linestyle='-', linewidth=2,
                label=f"{result['model_name']} (Final: {result['train_accuracies'][-1]:.1f}%)")
    ax1.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot validation accuracies
    for i, result in enumerate(results):
        ax2.plot(epochs, result['valid_accuracies'], 
                color=colors[i], linestyle='-', linewidth=2,
                label=f"{result['model_name']} (Final: {result['valid_accuracies'][-1]:.1f}%)")
    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Plot training losses
    for i, result in enumerate(results):
        ax3.plot(epochs, result['train_losses'], 
                color=colors[i], linestyle='-', linewidth=2,
                label=f"{result['model_name']}")
    ax3.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot test accuracy comparison (bar chart)
    model_names = [result['model_name'] for result in results]
    test_accuracies = [result['test_accuracy'] for result in results]
    
    bars = ax4.bar(model_names, test_accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'baseline_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plots saved to: {plot_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create transforms
    transform_train, transform_valid = create_transforms(args)
    
    # Load datasets
    try:
        train_loader, valid_loader, test_loader, class_names = load_datasets(
            args, transform_train, transform_valid
        )
        print(f"Dataset loaded from: {args.data_dir}")
        
        # Verify number of classes matches
        if len(class_names) != args.num_classes:
            print(f"Warning: Found {len(class_names)} classes but expecting {args.num_classes}")
            print(f"Classes found: {class_names}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure your dataset structure is:")
        print("Dataset/EnhancedDataset/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   └── ...")
        print("├── valid/")
        print("│   ├── class1/")
        print("│   └── ...")
        print("└── test/ (optional)")
        print("    ├── class1/")
        print("    └── ...")
        return
    
    # Initialize models and optimizers
    models_config = [
        {
            'model': LinearBaseline(num_classes=args.num_classes).to(device),
            'optimizer_class': optim.SGD,
            'optimizer_params': {'lr': 0.1},
            'name': 'Linear Baseline'
        },
        {
            'model': ShallowVGG(num_classes=args.num_classes).to(device),
            'optimizer_class': optim.Adam,
            'optimizer_params': {'lr': 0.01},
            'name': 'Shallow VGG'
        },
        {
            'model': MNISTLeNet(num_classes=args.num_classes).to(device),
            'optimizer_class': optim.SGD,
            'optimizer_params': {'lr': 0.01, 'momentum': 0.9},
            'name': 'MNIST LeNet'
        }
    ]
    
    # Train all models and store results
    results = []
    
    for config in models_config:
        model = config['model']
        optimizer = config['optimizer_class'](model.parameters(), **config['optimizer_params'])
        
        result = train_model(
            model, optimizer, train_loader, valid_loader, test_loader, 
            device, args.epochs, config['name']
        )
        results.append(result)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Plot comparison results
    plot_comparison_results(results, args.save_dir)
    
    # Save detailed results
    results_path = os.path.join(args.save_dir, 'baseline_results.txt')
    with open(results_path, 'w') as f:
        f.write("BASELINE MODELS COMPARISON RESULTS\n")
        f.write("="*50 + "\n\n")
        
        for result in results:
            f.write(f"{result['model_name']}:\n")
            f.write(f"  Final Training Accuracy:   {result['train_accuracies'][-1]:.2f}%\n")
            f.write(f"  Final Validation Accuracy: {result['valid_accuracies'][-1]:.2f}%\n")
            f.write(f"  Final Test Accuracy:       {result['test_accuracy']:.2f}%\n")
            f.write(f"  Best Training Accuracy:    {max(result['train_accuracies']):.2f}%\n")
            f.write(f"  Best Validation Accuracy:  {max(result['valid_accuracies']):.2f}%\n")
            f.write("\n")
    
    print(f"Detailed results saved to: {results_path}")
    
    # Print final summary - just accuracies
    print("\n" + "="*50)
    print("FINAL MODEL ACCURACIES")
    print("="*50)
    
    for result in results:
        print(f"{result['model_name']}: {result['test_accuracy']:.2f}%")

if __name__ == "__main__":
    main()