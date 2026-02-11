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


class CNNModel(nn.Module):
    """
    Convolutional Neural Network for image classification.
    Designed for CIFAR-10, MNIST, and similar datasets.
    GPU support enabled.
    """
    
    def __init__(self, num_classes, input_channels=3, device=None):
        """
        Initialize the CNN model.
        
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            device: Device to run on (cpu or cuda)
        """
        super(CNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.losses = []
        self.weights_dir = 'weights'
        
        # Create weights directory if it doesn't exist
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        
        # Build the model
        self._build_model()
        
        # Move model to device
        self.to(self.device)
    
    def _build_model(self):
        """Build the CNN model architecture using PyTorch."""
        # Convolutional block 1
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Convolutional block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Convolutional block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Calculate flattened size (depends on input image size)
        # For 32x32 images: after 3 pooling layers -> 4x4x256 = 4096
        # For 28x28 images: after 3 pooling layers -> 3x3x256 = 2304
        self.flatten_size = 256 * 4 * 4  # Default for 32x32
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, self.num_classes)
    
    def forward(self, x):
        """Forward pass."""
        # Block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout4(x)
        
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout5(x)
        
        x = self.fc3(x)
        
        return x
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model (setup optimizer and loss)."""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            outputs = self(batch_X)
            
            # Convert one-hot to class indices if needed
            if batch_y.dim() > 1 and batch_y.size(1) > 1:
                loss = self.criterion(outputs, torch.argmax(batch_y, dim=1))
                target = torch.argmax(batch_y, dim=1)
            else:
                loss = self.criterion(outputs, batch_y.squeeze())
                target = batch_y.squeeze()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self(batch_X)
                
                # Convert one-hot to class indices if needed
                if batch_y.dim() > 1 and batch_y.size(1) > 1:
                    loss = self.criterion(outputs, torch.argmax(batch_y, dim=1))
                    target = torch.argmax(batch_y, dim=1)
                else:
                    loss = self.criterion(outputs, batch_y.squeeze())
                    target = batch_y.squeeze()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train_model(self, train_loader, val_loader, epochs=50):
        """
        Train the CNN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs (int): Number of training epochs
        """
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return train_loss, train_acc
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self(batch_X)
                
                if batch_y.dim() > 1 and batch_y.size(1) > 1:
                    loss = self.criterion(outputs, torch.argmax(batch_y, dim=1))
                    target = torch.argmax(batch_y, dim=1)
                else:
                    loss = self.criterion(outputs, batch_y.squeeze())
                    target = batch_y.squeeze()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        print(f"\nTest Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def predict(self, X):
        """Make predictions on new data."""
        X = torch.FloatTensor(X).to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
        return outputs.cpu().numpy()
    
    def save_weights(self, model_name='cnn_model'):
        """
        Save the trained weights and model configuration.
        
        Args:
            model_name (str): Name of the model for saving weights
        """
        # Save model weights
        weights_path = os.path.join(self.weights_dir, f'{model_name}_weights.pth')
        torch.save(self.state_dict(), weights_path)
        print(f"Weights saved to: {weights_path}")
        
        # Save entire model
        model_path = os.path.join(self.weights_dir, f'{model_name}_model.pth')
        torch.save(self, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save model configuration as JSON
        config = {
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'device': str(self.device)
        }
        config_path = os.path.join(self.weights_dir, f'{model_name}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Model config saved to: {config_path}")
    
    def load_weights(self, model_name='cnn_model'):
        """
        Load previously saved weights.
        
        Args:
            model_name (str): Name of the model to load
        """
        weights_path = os.path.join(self.weights_dir, f'{model_name}_weights.pth')
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        print(f"Weights loaded from: {weights_path}")
    
    def summary(self):
        """Print model summary."""
        print(self.network if hasattr(self, 'network') else "CNN Model")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def get_weights_and_biases(self):
        """
        Extract all weights and biases from the model.
        Returns a list of (weight, bias) tuples for each layer.
        """
        weights_biases = []
        for name, layer in self.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                weights_biases.append({
                    'name': name,
                    'weight': layer.weight.data.clone().detach().cpu(),
                    'bias': layer.bias.data.clone().detach().cpu() if layer.bias is not None else None
                })
        return weights_biases


class QuantizedCNNModel:
    """
    Quantized version of CNN for integer inference.
    Simulates fixed-point integer arithmetic used in hardware.
    """
    
    def __init__(self, cnn_model, scaling_factor=None):
        """
        Initialize quantized model from trained CNN.
        
        Args:
            cnn_model: Trained CNNModel
            scaling_factor: Scaling factor S for quantization.
                          If None, auto-computed from weight statistics
        """
        self.cnn_model = cnn_model
        self.weights_biases = cnn_model.get_weights_and_biases()
        self.num_layers = len(self.weights_biases)
        
        # Auto-compute scaling factor if not provided
        if scaling_factor is None:
            self.scaling_factor = self._compute_scaling_factor()
        else:
            self.scaling_factor = scaling_factor
        
        # Quantize weights and biases
        self.quantized_weights_biases = self._quantize_weights()
        
        # Compute fractional bits (F) used in integer arithmetic
        self.F = int(torch.log2(torch.tensor(self.scaling_factor)).item())
    
    def _compute_scaling_factor(self):
        """
        Compute scaling factor automatically based on weight statistics.
        Uses the maximum absolute weight value to determine scaling.
        """
        max_weight = 0.0
        for wb in self.weights_biases:
            max_weight = max(max_weight, wb['weight'].abs().max().item())
        
        # Choose scaling factor to fit in int32 range while maintaining precision
        scaling_factor = 2 ** 8
        
        print(f"Auto-computed scaling factor: {scaling_factor}")
        return scaling_factor
    
    def _quantize_weights(self):
        """Quantize all weights and biases to integers."""
        quantized = []
        for wb in self.weights_biases:
            W_q = torch.round(wb['weight'] * self.scaling_factor).int()
            b_q = torch.round(wb['bias'] * self.scaling_factor).int() if wb['bias'] is not None else None
            quantized.append({
                'name': wb['name'],
                'weight': W_q,
                'bias': b_q
            })
        return quantized
    
    def forward_quantized(self, x):
        """
        Forward pass with quantized weights.
        
        Args:
            x: Input tensor (float32, will be quantized internally)
            
        Returns:
            Output logits (float32)
        """
        # Quantize input
        x_q = torch.round(x * self.scaling_factor).int().float()
        
        # Forward pass through CNN layers
        x = x_q
        layer_idx = 0
        
        for name, module in self.cnn_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Get quantized weights
                W_q = self.quantized_weights_biases[layer_idx]['weight']
                b_q = self.quantized_weights_biases[layer_idx]['bias']
                
                # Reshape weights for conv2d operation
                W_q_reshaped = W_q.reshape(module.weight.shape).float()
                b_q_float = b_q.float() / self.scaling_factor if b_q is not None else None
                
                # Conv2d operation with quantized weights
                x = torch.nn.functional.conv2d(
                    x / self.scaling_factor,
                    W_q_reshaped / self.scaling_factor,
                    bias=b_q_float,
                    stride=module.stride,
                    padding=module.padding
                )
                layer_idx += 1
                
            elif isinstance(module, nn.BatchNorm2d):
                x = module(x)
            elif isinstance(module, nn.ReLU) or name.endswith('relu'):
                x = torch.relu(x)
            elif isinstance(module, nn.MaxPool2d):
                x = torch.nn.functional.max_pool2d(x, module.kernel_size, module.stride)
            elif isinstance(module, nn.Dropout):
                x = x  # Skip dropout during inference
            elif isinstance(module, nn.Linear):
                # Flatten if needed
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                
                W_q = self.quantized_weights_biases[layer_idx]['weight']
                b_q = self.quantized_weights_biases[layer_idx]['bias']
                
                x = torch.matmul(x, W_q.float().t() / self.scaling_factor)
                if b_q is not None:
                    x = x + b_q.float() / self.scaling_factor
                
                layer_idx += 1
        
        return x


def compare_quantization_cnn(cnn_model, test_loader, scaling_factor=None):
    """
    Compare floating-point and integer quantized CNN inference.
    
    Args:
        cnn_model: Trained CNNModel
        test_loader: Test data loader
        scaling_factor: Scaling factor for quantization
    """
    print("\n" + "=" * 80)
    print("CNN QUANTIZATION AND INTEGER INFERENCE COMPARISON")
    print("=" * 80)
    
    # Create quantized model
    quantized_model = QuantizedCNNModel(cnn_model, scaling_factor=scaling_factor)
    
    print(f"\nQuantization Configuration:")
    print(f"  Scaling Factor (S): {quantized_model.scaling_factor}")
    print(f"  Fractional Bits (F): {quantized_model.F}")
    print(f"  Number of layers: {quantized_model.num_layers}")
    
    # Evaluate floating-point model
    print("\n" + "-" * 80)
    print("FLOATING-POINT INFERENCE (Original Trained CNN)")
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
            
            # Calculate loss
            if batch_y.dim() > 1 and batch_y.size(1) > 1:
                loss = cnn_model.criterion(outputs, torch.argmax(batch_y, dim=1))
                target = torch.argmax(batch_y, dim=1)
            else:
                loss = cnn_model.criterion(outputs, batch_y.squeeze())
                target = batch_y.squeeze()
            
            fp_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            fp_total += target.size(0)
            fp_correct += (predicted == target).sum().item()
    
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
    int_loss_total = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(cnn_model.device)
            batch_y = batch_y.to(cnn_model.device)
            
            # Quantized inference
            outputs_int = quantized_model.forward_quantized(batch_X)
            
            # Calculate loss
            if batch_y.dim() > 1 and batch_y.size(1) > 1:
                loss = cnn_model.criterion(outputs_int, torch.argmax(batch_y, dim=1))
                target = torch.argmax(batch_y, dim=1)
            else:
                loss = cnn_model.criterion(outputs_int, batch_y.squeeze())
                target = batch_y.squeeze()
            
            int_loss_total += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs_int, 1)
            int_total += target.size(0)
            int_correct += (predicted == target).sum().item()
    
    int_accuracy = int_correct / int_total
    int_loss = int_loss_total / len(test_loader)
    
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


def prepare_mnist_dataset_cnn(test_size=0.2):
    """
    Prepare the MNIST dataset for CNN training.
    Returns images as 1x28x28 tensors.
    """
    print("Loading MNIST dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training data
    train_indices = list(range(len(train_dataset)))
    val_indices = train_indices[int(0.8*len(train_dataset)):]
    train_indices = train_indices[:int(0.8*len(train_dataset))]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    return train_subset, val_subset, test_dataset, 1, 10


def prepare_cifar10_dataset_cnn(test_size=0.2):
    """
    Prepare the CIFAR-10 dataset for CNN training.
    Returns images as 3x32x32 tensors.
    """
    print("Loading CIFAR-10 dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Split training data
    train_indices = list(range(len(train_dataset)))
    val_indices = train_indices[int(0.8*len(train_dataset)):]
    train_indices = train_indices[:int(0.8*len(train_dataset))]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    return train_subset, val_subset, test_dataset, 3, 10


def prepare_cifar100_dataset_cnn(test_size=0.2):
    """
    Prepare the CIFAR-100 dataset for CNN training.
    Returns images as 3x32x32 tensors.
    """
    print("Loading CIFAR-100 dataset...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # Split training data
    train_indices = list(range(len(train_dataset)))
    val_indices = train_indices[int(0.8*len(train_dataset)):]
    train_indices = train_indices[:int(0.8*len(train_dataset))]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    return train_subset, val_subset, test_dataset, 3, 100


def select_dataset_cnn():
    """
    Allow user to select which dataset to use for CNN.
    Returns the appropriate dataset preparation function and dataset name.
    """
    print("\n" + "=" * 60)
    print("Available Image Datasets for CNN:")
    print("=" * 60)
    print("1. MNIST (28x28, 1 channel, 10 classes) - Handwritten Digits")
    print("2. CIFAR-10 (32x32, 3 channels, 10 classes) - Common Objects")
    print("3. CIFAR-100 (32x32, 3 channels, 100 classes) - Fine-grained")
    print("=" * 60)
    
    choice = input("\nSelect dataset (1-3, default=2): ").strip()
    
    if choice == '1':
        return prepare_mnist_dataset_cnn, 'MNIST'
    elif choice == '3':
        return prepare_cifar100_dataset_cnn, 'CIFAR-100'
    else:
        return prepare_cifar10_dataset_cnn, 'CIFAR-10'


def main():
    """Main function to train the CNN model for image classification."""
    
    print("=" * 60)
    print("CNN Training - Image Classification (PyTorch with GPU Support)")
    print("=" * 60)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Let user select dataset
    dataset_func, dataset_name = select_dataset_cnn()
    
    # Prepare dataset
    print(f"\nPreparing {dataset_name} dataset...")
    train_subset, val_subset, test_dataset, input_channels, num_classes = dataset_func()
    
    print(f"\nDataset Information ({dataset_name}):")
    print(f"  Input Channels: {input_channels}")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Training samples: {len(train_subset)}")
    print(f"  Validation samples: {len(val_subset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create and compile model
    cnn = CNNModel(num_classes=num_classes, input_channels=input_channels, device=device)
    cnn.compile_model(learning_rate=0.001)
    
    print(f"\nCNN Architecture:")
    print(f"  Input Channels: {input_channels}")
    print(f"  Output Classes: {num_classes}")
    print(f"  Device: {device}")
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in cnn.parameters())
    trainable_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train the model
    print("\n" + "=" * 60)
    print("Training the CNN...")
    print("=" * 60)
    cnn.train_model(train_loader, val_loader, epochs=50)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    cnn.evaluate(test_loader)
    
    # Save the weights and model
    print("\n" + "=" * 60)
    print("Saving Model and Weights...")
    print("=" * 60)
    model_name = f'{dataset_name.lower().replace("-", "")}_cnn_classifier'
    cnn.save_weights(model_name=model_name)
    
    # Quantization and Integer Inference Comparison
    print("\n" + "=" * 60)
    print("Performing Quantization and Comparison...")
    print("=" * 60)
    comparison_results = compare_quantization_cnn(cnn, test_loader, scaling_factor=2**16)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
