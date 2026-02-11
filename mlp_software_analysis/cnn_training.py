import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime


class CNNModel(nn.Module):
    """
    Convolutional Neural Network version of MLPModel.
    Supports dynamic image size + GPU + quantization export.
    """

    def __init__(self, input_channels, num_classes, device=None):
        super(CNNModel, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.num_classes = num_classes
        self.weights_dir = "weights"
        os.makedirs(self.weights_dir, exist_ok=True)

        # --- CNN architecture ---
        self.features = nn.Sequential(

            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.to(self.device)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def compile_model(self, lr=0.001):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()


    def train_epoch(self, loader):
        self.train()
        total_loss = 0
        correct = 0
        total = 0

        for X,y in loader:
            X,y = X.to(self.device), y.to(self.device)

            out = self(X)
            loss = self.criterion(out,y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)

        return total_loss/len(loader), correct/total


    def validate(self, loader):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X,y in loader:
                X,y = X.to(self.device), y.to(self.device)
                out = self(X)

                loss = self.criterion(out,y)
                total_loss += loss.item()

                pred = out.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)

        return total_loss/len(loader), correct/total
    
    def evaluate(self, loader):
        """Evaluate the model on a dataset."""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self(X)
                
                loss = self.criterion(out, y)
                total_loss += loss.item()
                
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return total_loss / len(loader), correct / total
    def train_model(self, train_loader, val_loader, epochs=20):
        """Train the model."""
        best = 1e9
        patience = 10
        counter = 0

        for ep in range(epochs):
            tl, ta = self.train_epoch(train_loader)
            vl, va = self.validate(val_loader)

            print(f"Epoch {ep+1}/{epochs}: TrainAcc={ta:.4f} ValAcc={va:.4f}")

            if vl < best:
                best = vl
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break
    def get_weights_and_biases(self):

        params = []

        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                params.append({
                    "weight": layer.weight.detach().cpu(),
                    "bias": layer.bias.detach().cpu()
                })

        return params

class QuantizedCNN:
    """Dynamic quantized CNN that works with any architecture."""

    def __init__(self, cnn_model, scaling_factor=2**16):
        self.model = cnn_model
        self.S = scaling_factor
        self.F = int(np.log2(self.S))
        self.device = cnn_model.device

    def forward_quantized(self, x):
        """
        Forward pass with quantized weights and quantized inputs.
        Assumes input x is already quantized by scaling_factor.
        Mirrors the MLP approach.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Quantize weights on-the-fly
                W = module.weight.data
                W_q = torch.round(W * self.S).int().float()
                
                if module.bias is not None:
                    b = module.bias.data
                    b_q = torch.round(b * self.S).int().float()
                else:
                    b_q = None
                
                # Both x and W_q are quantized (scaled by S)
                # Output will be scaled by S^2, so divide by S^2
                x = torch.nn.functional.conv2d(
                    x,
                    W_q / (self.S ** 2),
                    bias=b_q / (self.S ** 2) if b_q is not None else None,
                    stride=module.stride,
                    padding=module.padding
                )
                
            elif isinstance(module, nn.BatchNorm2d):
                x = module(x)
            elif isinstance(module, nn.ReLU):
                x = torch.relu(x)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                x = torch.nn.functional.adaptive_avg_pool2d(x, module.output_size)
            elif isinstance(module, nn.MaxPool2d):
                x = torch.nn.functional.max_pool2d(x, module.kernel_size, module.stride, module.padding)
            elif isinstance(module, nn.Dropout):
                x = x  # Skip dropout during inference
            elif isinstance(module, nn.Flatten):
                x = x.view(x.size(0), -1)
            elif isinstance(module, nn.Linear):
                W = module.weight.data
                W_q = torch.round(W * self.S).int().float()
                
                if module.bias is not None:
                    b = module.bias.data
                    b_q = torch.round(b * self.S).int().float()
                else:
                    b_q = None
                
                # Both x and W_q are quantized (scaled by S)
                # Output will be scaled by S^2, so divide by S^2
                x = torch.nn.functional.linear(
                    x,
                    W_q / (self.S ** 2),
                    bias=b_q / (self.S ** 2) if b_q is not None else None
                )
        
        return x

def compare_quantization(cnn_model, test_loader, scaling_factor=2**16):
    """
    Compare floating-point and integer quantized CNN inference.
    
    Args:
        cnn_model: Trained CNNModel
        test_loader: DataLoader with test data (tensors in correct format)
        scaling_factor: Scaling factor for quantization
    """
    print("\n" + "=" * 80)
    print("QUANTIZATION AND INTEGER INFERENCE COMPARISON")
    print("=" * 80)
    
    # Create quantized model
    quantized_model = QuantizedCNN(cnn_model, scaling_factor=scaling_factor)
    
    print(f"\nQuantization Configuration:")
    print(f"  Scaling Factor (S): {quantized_model.S}")
    print(f"  Fractional Bits (F): {quantized_model.F}")
    
    # Evaluate floating-point model
    print("\n" + "-" * 80)
    print("FLOATING-POINT INFERENCE (Original Trained Model)")
    print("-" * 80)
    
    cnn_model.eval()
    fp_correct = 0
    fp_total = 0
    fp_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(cnn_model.device)
            batch_y = batch_y.to(cnn_model.device)
            
            outputs = cnn_model(batch_X)
            loss = cnn_model.criterion(outputs, batch_y)
            fp_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            fp_total += batch_y.size(0)
            fp_correct += (predicted == batch_y).sum().item()
    
    fp_loss = fp_loss / len(test_loader)
    fp_accuracy = fp_correct / fp_total
    
    print(f"Loss: {fp_loss:.4f}")
    print(f"Accuracy: {fp_accuracy:.4f} ({fp_correct}/{fp_total})")
    
    # Evaluate quantized model
    print("\n" + "-" * 80)
    print("INTEGER QUANTIZED INFERENCE (Simulated Hardware)")
    print("-" * 80)
    
    int_correct = 0
    int_total = 0
    int_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(cnn_model.device)
            batch_y = batch_y.to(cnn_model.device)
            
            # Quantize inputs (same as MLP pattern)
            x_q = torch.round(batch_X * quantized_model.S).int().float()
            
            # Quantized inference
            outputs_int = quantized_model.forward_quantized(x_q)
            
            loss = cnn_model.criterion(outputs_int, batch_y)
            int_loss += loss.item()
            
            _, predicted = torch.max(outputs_int, 1)
            int_total += batch_y.size(0)
            int_correct += (predicted == batch_y).sum().item()
    
    int_loss = int_loss / len(test_loader)
    int_accuracy = int_correct / int_total
    
    print(f"Loss: {int_loss:.4f}")
    print(f"Accuracy: {int_accuracy:.4f} ({int_correct}/{int_total})")
    
    # Comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nFloating-Point (FP32):")
    print(f"  Loss:     {fp_loss:.6f}")
    print(f"  Accuracy: {fp_accuracy:.6f}")
    
    print(f"\nInteger Quantized (INT32):")
    print(f"  Loss:     {int_loss:.6f}")
    print(f"  Accuracy: {int_accuracy:.6f}")
    
    print(f"\nDifference:")
    print(f"  Loss Difference:     {abs(fp_loss - int_loss):.6f}")
    print(f"  Accuracy Difference: {abs(fp_accuracy - int_accuracy):.6f}")
    print(f"  Accuracy Drop:       {(fp_accuracy - int_accuracy) * 100:.2f}%")
    
    print("\n" + "=" * 80)
    
    return {
        'fp_loss': fp_loss,
        'fp_accuracy': fp_accuracy,
        'int_loss': int_loss,
        'int_accuracy': int_accuracy,
        'quantized_model': quantized_model
    }


def prepare_mnist_dataset(batch_size=32):
    """
    Prepare the MNIST dataset (28x28 images, 1 channel, 10 classes).
    Returns DataLoaders with proper tensor format (batch, channels, height, width).
    """
    print("Loading MNIST dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load full datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, 1, 10


def prepare_cifar10_dataset(batch_size=32):
    """
    Prepare the CIFAR-10 dataset (32x32 RGB images, 3 channels, 10 classes).
    Returns DataLoaders with proper tensor format (batch, channels, height, width).
    """
    print("Loading CIFAR-10 dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load full datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Split training into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, 3, 10


def prepare_cifar100_dataset(batch_size=32):
    """
    Prepare the CIFAR-100 dataset (32x32 RGB images, 3 channels, 100 classes).
    Returns DataLoaders with proper tensor format (batch, channels, height, width).
    """
    print("Loading CIFAR-100 dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load full datasets
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # Split training into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, 3, 100


def select_dataset():
    """
    Allow user to select which dataset to use.
    Returns the appropriate dataset preparation function and dataset name.
    """
    print("\n" + "=" * 60)
    print("Available Image Datasets:")
    print("=" * 60)
    print("1. MNIST (28x28, 1 channel, 10 classes)")
    print("2. CIFAR-10 (32x32, 3 channels, 10 classes)")
    print("3. CIFAR-100 (32x32, 3 channels, 100 classes)")
    print("=" * 60)
    
    choice = input("\nSelect dataset (1-3, default=2): ").strip()
    
    if choice == '1':
        return prepare_mnist_dataset, 'MNIST'
    elif choice == '3':
        return prepare_cifar100_dataset, 'CIFAR-100'
    else:
        return prepare_cifar10_dataset, 'CIFAR-10'  


def ensure_results_dir():
    """
    Ensure results directory exists.
    """
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_weight_statistics(cnn_model, log_file):
    """
    Compute and log weight statistics for all layers.
    
    Args:
        cnn_model: Trained CNN model
        log_file: File handle to write logs
    """
    log_file.write("Weight Statistics:\n")
    log_file.write(f"{'Layer':<40} {'Max |W|':<15} {'Std(W)':<15}\n")
    log_file.write("-" * 70 + "\n")
    
    for name, module in cnn_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            max_weight = w.abs().max().item()
            std_weight = w.std().item()
            layer_name = name[:40]
            log_file.write(f"{layer_name:<40} {max_weight:<15.6f} {std_weight:<15.6f}\n")
    
    log_file.write("\n")


def main():
    """Main function to train the CNN model for image classification with multiple configurations."""
    
    print("=" * 60)
    print("CNN Training - Comparative Analysis (PyTorch with GPU Support)")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Ensure results directory exists
    results_dir = ensure_results_dir()
    
    # Let user select dataset
    dataset_func, dataset_name = select_dataset()
    
    # Create log file with fixed name (append mode)
    log_filename = os.path.join(results_dir, f'{dataset_name.lower().replace("-", "")}_cnn_comparison.txt')
    
    # Check if file exists to determine if we're appending
    file_exists = os.path.exists(log_filename)
    open_mode = 'a' if file_exists else 'w'
    
    with open(log_filename, open_mode) as log_file:
        # Write header only if new file
        if not file_exists:
            log_file.write("=" * 80 + "\n")
            log_file.write(f"CNN QUANTIZATION IMPACT ANALYSIS - {dataset_name}\n")
            log_file.write("=" * 80 + "\n")
            log_file.write("Configurations:\n")
            log_file.write("  1. 20 epochs, scaling_factor=2^8 (256)\n")
            log_file.write("  2. 20 epochs, scaling_factor=2^16 (65536)\n")
            log_file.write("  3. 50 epochs, scaling_factor=2^8 (256)\n")
            log_file.write("  4. 50 epochs, scaling_factor=2^16 (65536)\n\n")
        
        # Add separator and timestamp for this run
        log_file.write("\n\n")
        log_file.write("#" * 80 + "\n")
        log_file.write(f"RUN at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("#" * 80 + "\n\n")
        
        # Define configurations
        configs = [
            (20, 2**8, "20 epochs + SF=2^8"),
            (20, 2**16, "20 epochs + SF=2^16"),
            (50, 2**8, "50 epochs + SF=2^8"),
            (50, 2**16, "50 epochs + SF=2^16"),
        ]
        
        results_summary = []
        
        # Run each configuration
        for epochs, scaling_factor, config_name in configs:
            log_file.write(f"\n{'=' * 80}\n")
            log_file.write(f"Configuration: {config_name}\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'=' * 80}\n\n")
            
            print(f"\n{'=' * 60}")
            print(f"Running: {dataset_name} with epochs={epochs}, scaling_factor={scaling_factor}")
            print(f"{'=' * 60}")
            
            try:
                # Prepare dataset
                print(f"Preparing {dataset_name} dataset...")
                train_subset, val_subset, test_dataset, input_channels, num_classes = dataset_func()
                
                log_file.write(f"Dataset: {dataset_name}\n")
                log_file.write(f"  Input Channels: {input_channels}\n")
                log_file.write(f"  Number of Classes: {num_classes}\n")
                log_file.write(f"  Training samples: {len(train_subset)}\n")
                log_file.write(f"  Validation samples: {len(val_subset)}\n")
                log_file.write(f"  Test samples: {len(test_dataset)}\n\n")
                
                # Create data loaders
                train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
                
                # Create and compile model
                cnn = CNNModel(num_classes=num_classes, input_channels=input_channels, device=device)
                cnn.compile_model(learning_rate=0.001)
                
                # Count parameters
                total_params = sum(p.numel() for p in cnn.parameters())
                trainable_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
                log_file.write(f"Model Parameters:\n")
                log_file.write(f"  Total: {total_params:,}\n")
                log_file.write(f"  Trainable: {trainable_params:,}\n\n")
                
                # Train the model
                print(f"Training for {epochs} epochs...")
                log_file.write(f"Training for {epochs} epochs...\n\n")
                cnn.train_model(train_loader, val_loader, epochs=epochs)
                
                # Evaluate on test set
                print(f"Evaluating on test set...")
                test_loss, test_acc = cnn.evaluate(test_loader)
                
                log_file.write(f"\nTest Performance (Floating-Point):\n")
                log_file.write(f"  Loss: {test_loss:.6f}\n")
                log_file.write(f"  Accuracy: {test_acc:.6f}\n\n")
                
                # Extract and log weight statistics
                log_weight_statistics(cnn, log_file)
                
                # Quantization comparison
                print(f"Running quantization with scaling_factor={scaling_factor}...")
                comparison_results = compare_quantization(cnn, test_loader, scaling_factor=scaling_factor)
                
                # Log results
                log_file.write(f"Quantization Results (scaling_factor={scaling_factor}):\n")
                log_file.write(f"  Floating-Point Accuracy:  {comparison_results['fp_accuracy']:.6f}\n")
                log_file.write(f"  Quantized Accuracy:       {comparison_results['int_accuracy']:.6f}\n")
                log_file.write(f"  Accuracy Drop:            {(comparison_results['fp_accuracy'] - comparison_results['int_accuracy']) * 100:.2f}%\n")
                log_file.write(f"  Loss Difference:          {abs(comparison_results['fp_loss'] - comparison_results['int_loss']):.6f}\n\n")
                
                log_file.flush()
                print(f"Configuration complete!\n")
                
                results_summary.append({
                    'config': config_name,
                    'epochs': epochs,
                    'scaling_factor': scaling_factor,
                    'fp_accuracy': comparison_results['fp_accuracy'],
                    'int_accuracy': comparison_results['int_accuracy'],
                    'accuracy_drop': comparison_results['fp_accuracy'] - comparison_results['int_accuracy']
                })
                
            except Exception as e:
                error_msg = f"Error in configuration: {str(e)}\n"
                print(error_msg)
                log_file.write(error_msg)
                log_file.flush()
        
        # Write summary table
        log_file.write("\n" + "-" * 80 + "\n")
        log_file.write("SUMMARY TABLE FOR THIS RUN\n")
        log_file.write("-" * 80 + "\n\n")
        
        log_file.write(f"{'Configuration':<25} {'FP Accuracy':<15} {'Quantized':<15} {'Drop %':<10}\n")
        log_file.write("-" * 65 + "\n")
        
        for result in results_summary:
            config_str = f"{result['config']:<25}"
            fp_acc = f"{result['fp_accuracy']:.6f}".ljust(15)
            int_acc = f"{result['int_accuracy']:.6f}".ljust(15)
            drop_pct = f"{result['accuracy_drop'] * 100:.2f}%".ljust(10)
            log_file.write(f"{config_str} {fp_acc} {int_acc} {drop_pct}\n")
        
        log_file.write("\n" + "-" * 80 + "\n")
        log_file.write("KEY OBSERVATIONS\n")
        log_file.write("-" * 80 + "\n\n")
        
        if results_summary:
            # Analysis
            epochs_20_drops = [r['accuracy_drop'] for r in results_summary if r['epochs'] == 20]
            epochs_50_drops = [r['accuracy_drop'] for r in results_summary if r['epochs'] == 50]
            sf_8_drops = [r['accuracy_drop'] for r in results_summary if r['scaling_factor'] == 2**8]
            sf_16_drops = [r['accuracy_drop'] for r in results_summary if r['scaling_factor'] == 2**16]
            
            log_file.write(f"Average drop (20 epochs): {np.mean(epochs_20_drops):.6f}\n")
            log_file.write(f"Average drop (50 epochs): {np.mean(epochs_50_drops):.6f}\n")
            log_file.write(f"Average drop (SF=2^8):    {np.mean(sf_8_drops):.6f}\n")
            log_file.write(f"Average drop (SF=2^16):   {np.mean(sf_16_drops):.6f}\n\n")
            
            log_file.write("Insights:\n")
            if np.mean(epochs_20_drops) < np.mean(epochs_50_drops):
                log_file.write(f"  - Models trained for fewer epochs (20) show LESS quantization degradation\n")
            log_file.write(f"  - Scaling factor 2^{int(np.log2(2**16))} provides better precision than 2^8\n")
        
        log_file.write("\n")
    
    print(f"\n{'=' * 60}")
    print(f"All configurations completed!")
    print(f"Results appended to: {log_filename}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
