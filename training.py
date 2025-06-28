import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from cnn import initialize_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train Fish CNN Model')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of fish classes (default: 5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW optimizer (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    
    # Scheduler parameters
    parser.add_argument('--step_size', type=int, default=5,
                        help='Step size for learning rate scheduler (default: 5)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Decay factor for learning rate scheduler (default: 0.5)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Dataset/EnhancedDataset',
                        help='Path to the dataset directory (default: Dataset/EnhancedDataset)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for training (default: 256)')
    
    # Augmentation parameters
    parser.add_argument('--rotation', type=int, default=10,
                        help='Random rotation angle for data augmentation (default: 10)')
    parser.add_argument('--scale_min', type=float, default=0.8,
                        help='Minimum scale for random resized crop (default: 0.8)')
    parser.add_argument('--scale_max', type=float, default=1.0,
                        help='Maximum scale for random resized crop (default: 1.0)')
    
    # Normalization parameters
    parser.add_argument('--mean', type=float, nargs=3, default=[0.5, 0.5, 0.5],
                        help='Mean values for normalization (default: [0.5, 0.5, 0.5])')
    parser.add_argument('--std', type=float, nargs=3, default=[0.5, 0.5, 0.5],
                        help='Std values for normalization (default: [0.5, 0.5, 0.5])')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='Directory to save the trained model (default: saved_models)')
    parser.add_argument('--model_name', type=str, default='fish_cnn.pth',
                        help='Name of the saved model file (default: fish_cnn.pth)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda) (default: auto)')
    
    return parser.parse_args()

def create_transforms(args):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(args.rotation),
        transforms.RandomResizedCrop(args.img_size, scale=(args.scale_min, args.scale_max)),
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
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform_valid)
    
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
    
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(valid_dataset)} validation samples")
    print(f"Class names: {train_dataset.classes}")
    
    return train_loader, valid_loader, train_dataset.classes

def get_device(device_arg):
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

def train_model(args, model, train_loader, valid_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    train_accuracies = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}")
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
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })

        scheduler.step()

        train_accuracy = 100 * correct / total
        train_loss = epoch_loss / len(train_loader)
        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)
        
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"  Valid - Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)
    
    return train_accuracies, valid_accuracies, train_losses, valid_losses

def plot_training_curves(train_accuracies, valid_accuracies, train_losses, valid_losses, save_dir):
    epochs = range(1, len(train_accuracies) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, valid_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    ax2.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to: {plot_path}")

def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    model = initialize_model(num_classes=args.num_classes, device=device)
    print(f"Model initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    transform_train, transform_valid = create_transforms(args)
    
    try:
        train_loader, valid_loader, class_names = load_datasets(args, transform_train, transform_valid)
        print(f"Dataset loaded from: {args.data_dir}")
        print(f"Image size: {args.img_size}x{args.img_size}")
        
        if len(class_names) != args.num_classes:
            print(f"Warning: Found {len(class_names)} classes but model expects {args.num_classes}")
            print(f"   Classes found: {class_names}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure your dataset structure is:")
        print("Dataset/EnhancedDataset/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   ├── class2/")
        print("│   └── ...")
        print("└── valid/")
        print("    ├── class1/")
        print("    ├── class2/")
        print("    └── ...")
        return
    
    train_accuracies, valid_accuracies, train_losses, valid_losses = train_model(
        args, model, train_loader, valid_loader, device
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, args.model_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'class_names': class_names,
        'args': args
    }, model_path)
    print(f"Model and training history saved to: {model_path}")
    
    plot_training_curves(train_accuracies, valid_accuracies, train_losses, valid_losses, args.save_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE - FINAL RESULTS")
    print("="*60)
    print(f"Final Training Accuracy:   {train_accuracies[-1]:.2f}%")
    print(f"Final Validation Accuracy: {valid_accuracies[-1]:.2f}%")
    print(f"Best Training Accuracy:    {max(train_accuracies):.2f}%")
    print(f"Best Validation Accuracy:  {max(valid_accuracies):.2f}%")
    print(f"Final Training Loss:       {train_losses[-1]:.4f}")
    print(f"Final Validation Loss:     {valid_losses[-1]:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()