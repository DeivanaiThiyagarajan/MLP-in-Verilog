import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import os
import json

class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron with PyTorch backend.
    Dynamic input and output layers, fixed hidden layers (4-5 hidden layers).
    GPU support enabled.
    """
    
    def __init__(self, input_size, num_classes, hidden_layers=None, device=None):
        """
        Initialize the MLP model.
        
        Args:
            input_size (int): Size of input features (dynamic based on image input)
            num_classes (int): Number of output classes (dynamic)
            hidden_layers (list): List of hidden layer sizes. Default: [128, 64, 32, 16]
            device: Device to run on (cpu or cuda)
        """
        super(MLPModel, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Default hidden layer configuration: 4 layers
        if hidden_layers is None:
            self.hidden_layers = [128, 64, 32, 16]
        else:
            self.hidden_layers = hidden_layers
        
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
        """Build the MLP model architecture using PyTorch."""
        layers_list = []
        
        # Input layer to first hidden layer
        layers_list.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.BatchNorm1d(self.hidden_layers[0]))
        layers_list.append(nn.Dropout(0.2))
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            layers_list.append(nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.BatchNorm1d(self.hidden_layers[i]))
            layers_list.append(nn.Dropout(0.2))
        
        # Output layer
        layers_list.append(nn.Linear(self.hidden_layers[-1], self.num_classes))
        
        self.network = nn.Sequential(*layers_list)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model (setup optimizer and loss)."""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Use CrossEntropyLoss for multi-class classification
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
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the MLP model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience = 10
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
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate the model on test data."""
        # Convert to PyTorch tensors
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Create DataLoader
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
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
    
    def save_weights(self, model_name='mlp_model'):
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
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'hidden_layers': self.hidden_layers,
            'device': str(self.device)
        }
        config_path = os.path.join(self.weights_dir, f'{model_name}_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Model config saved to: {config_path}")
    
    def load_weights(self, model_name='mlp_model'):
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
        print(self.network)
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
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                weights_biases.append({
                    'weight': layer.weight.data.clone().detach().cpu(),
                    'bias': layer.bias.data.clone().detach().cpu()
                })
        return weights_biases


class QuantizedMLPModel:
    """
    Quantized version of MLP for integer inference.
    Simulates fixed-point integer arithmetic used in hardware.
    """
    
    def __init__(self, weights_biases, scaling_factor=None):
        """
        Initialize quantized model from trained weights.
        
        Args:
            weights_biases: List of dicts with 'weight' and 'bias' tensors
            scaling_factor: Scaling factor S for quantization. 
                          If None, auto-computed from weight statistics
        """
        self.weights_biases = weights_biases
        self.num_layers = len(weights_biases)
        
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
        # For int32: max value ~2^31, we use 2^16 as scaling factor for reasonable precision
        scaling_factor = 2 ** 16
        
        print(f"Auto-computed scaling factor: {scaling_factor}")
        return scaling_factor
    
    def _quantize_weights(self):
        """Quantize all weights and biases to integers."""
        quantized = []
        for wb in self.weights_biases:
            W_q = torch.round(wb['weight'] * self.scaling_factor).int()
            b_q = torch.round(wb['bias'] * self.scaling_factor).int()
            quantized.append({
                'weight': W_q,
                'bias': b_q
            })
        return quantized
    
    def linear_int(self, x_q, layer_idx):
        """
        Integer linear operation (simulates hardware).
        
        Args:
            x_q: Quantized input tensor
            layer_idx: Layer index
            
        Returns:
            Output after linear operation and bit shift
        """
        W_q = self.quantized_weights_biases[layer_idx]['weight']
        b_q = self.quantized_weights_biases[layer_idx]['bias']
        
        # Matrix multiplication with quantized weights
        acc = torch.matmul(W_q.float(), x_q.float())
        
        # Right shift to remove scaling (simulates fixed-point division)
        acc = torch.div(acc, self.scaling_factor).int().float()
        
        # Add bias
        acc = acc + b_q.float() / self.scaling_factor
        
        return acc
    
    def forward_int(self, x_q):
        """
        Forward pass using integer arithmetic and quantized weights.
        
        Args:
            x_q: Quantized input
            
        Returns:
            Output logits
        """
        x = x_q
        
        for i in range(self.num_layers - 1):
            # Integer linear operation
            z = self.linear_int(x, i)
            
            # ReLU activation
            x = torch.clamp(z, min=0.0)
        
        # Output layer (no activation)
        output = self.linear_int(x, self.num_layers - 1)
        
        return output


def compare_quantization(mlp_model, X_test, y_test, scaling_factor=None, batch_size=32):
    """
    Compare floating-point and integer quantized inference.
    
    Args:
        mlp_model: Trained MLPModel
        X_test: Test features (float)
        y_test: Test labels (one-hot)
        scaling_factor: Scaling factor for quantization
        batch_size: Batch size for evaluation
    """
    print("\n" + "=" * 80)
    print("QUANTIZATION AND INTEGER INFERENCE COMPARISON")
    print("=" * 80)
    
    # Extract weights from trained model
    weights_biases = mlp_model.get_weights_and_biases()
    
    # Create quantized model
    quantized_model = QuantizedMLPModel(weights_biases, scaling_factor=scaling_factor)
    
    print(f"\nQuantization Configuration:")
    print(f"  Scaling Factor (S): {quantized_model.scaling_factor}")
    print(f"  Fractional Bits (F): {quantized_model.F}")
    print(f"  Weight ranges:")
    for i, wb in enumerate(weights_biases):
        w_min = wb['weight'].min().item()
        w_max = wb['weight'].max().item()
        print(f"    Layer {i}: [{w_min:.6f}, {w_max:.6f}]")
    
    # Prepare test data
    X_test_tensor = torch.FloatTensor(X_test).to(mlp_model.device)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Convert to PyTorch for evaluation
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate floating-point model
    print("\n" + "-" * 80)
    print("FLOATING-POINT INFERENCE (Original Trained Model)")
    print("-" * 80)
    
    mlp_model.eval()
    fp_correct = 0
    fp_total = 0
    fp_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(mlp_model.device)
            batch_y = batch_y.to(mlp_model.device)
            
            outputs = mlp_model(batch_X)
            
            # Calculate loss
            if batch_y.dim() > 1 and batch_y.size(1) > 1:
                loss = mlp_model.criterion(outputs, torch.argmax(batch_y, dim=1))
                target = torch.argmax(batch_y, dim=1)
            else:
                loss = mlp_model.criterion(outputs, batch_y.squeeze())
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
    int_outputs_all = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.cpu()
            batch_y = batch_y.cpu()
            
            # Quantize inputs
            x_q = torch.round(batch_X * quantized_model.scaling_factor).int().float()
            
            # Integer inference
            outputs_int = []
            for i in range(len(batch_X)):
                output = quantized_model.forward_int(x_q[i:i+1].squeeze(0))
                outputs_int.append(output)
            
            outputs_int = torch.stack(outputs_int)
            int_outputs_all.append(outputs_int)
            
            # Calculate accuracy
            target = torch.argmax(batch_y, dim=1)
            _, predicted = torch.max(outputs_int, 1)
            int_total += target.size(0)
            int_correct += (predicted == target).sum().item()
    
    int_accuracy = int_correct / int_total
    
    # For loss calculation, convert back to float and rescale
    all_outputs_int = torch.cat(int_outputs_all, dim=0)
    logits_float = all_outputs_int.float() / quantized_model.scaling_factor
    all_targets = torch.argmax(y_test_tensor, dim=1)
    int_loss = mlp_model.criterion(logits_float, all_targets).item()
    
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


def prepare_digit_dataset(test_size=0.2):
    """
    Prepare the digit dataset (8x8 images = 64 features).
    Image classification task.
    """
    # Load digits dataset (64 features from 8x8 images)
    data = load_digits()
    X = data.data
    y = data.target
    
    # Normalize the data
    X = X / 16.0  # Pixel values are 0-16
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y))
    y_onehot = np.eye(num_classes)[y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X.shape[1], num_classes


def prepare_mnist_dataset(test_size=0.2):
    """
    Prepare the MNIST dataset (28x28 images = 784 features).
    Handwritten digit classification (0-9).
    """
    print("Loading MNIST dataset...")
    
    # Define transformation to convert to tensor and normalize
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Convert to numpy for preprocessing
    X_train = []
    y_train = []
    for img, label in train_dataset:
        X_train.append(img.numpy().flatten())
        y_train.append(label)
    
    X_test = []
    y_test = []
    for img, label in test_dataset:
        X_test.append(img.numpy().flatten())
        y_test.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_onehot, test_size=0.2, random_state=42
    )
    
    num_features = X_train.shape[1]
    
    return X_train, X_val, X_test, y_train, y_val, y_test_onehot, num_features, num_classes


def prepare_cifar10_dataset(test_size=0.2):
    """
    Prepare the CIFAR-10 dataset (32x32 RGB images = 3072 features).
    10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
    """
    print("Loading CIFAR-10 dataset...")
    
    # Define transformation to convert to tensor and normalize
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Convert to numpy for preprocessing
    X_train = []
    y_train = []
    for img, label in train_dataset:
        X_train.append(img.numpy().flatten())
        y_train.append(label)
    
    X_test = []
    y_test = []
    for img, label in test_dataset:
        X_test.append(img.numpy().flatten())
        y_test.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_onehot, test_size=0.2, random_state=42
    )
    
    num_features = X_train.shape[1]
    
    return X_train, X_val, X_test, y_train, y_val, y_test_onehot, num_features, num_classes


def prepare_cifar100_dataset(test_size=0.2):
    """
    Prepare the CIFAR-100 dataset (32x32 RGB images = 3072 features).
    100 classes for fine-grained classification.
    """
    print("Loading CIFAR-100 dataset...")
    
    # Define transformation to convert to tensor and normalize
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # Convert to numpy for preprocessing
    X_train = []
    y_train = []
    for img, label in train_dataset:
        X_train.append(img.numpy().flatten())
        y_train.append(label)
    
    X_test = []
    y_test = []
    for img, label in test_dataset:
        X_test.append(img.numpy().flatten())
        y_test.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Convert labels to one-hot encoding
    num_classes = 100
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_onehot, test_size=0.2, random_state=42
    )
    
    num_features = X_train.shape[1]
    
    return X_train, X_val, X_test, y_train, y_val, y_test_onehot, num_features, num_classes


def select_dataset():
    """
    Allow user to select which dataset to use.
    Returns the appropriate dataset preparation function and dataset name.
    """
    print("\n" + "=" * 60)
    print("Available Image Datasets:")
    print("=" * 60)
    print("1. Digits (8x8, 10 classes) - Small & Fast")
    print("2. MNIST (28x28, 10 classes) - Handwritten Digits")
    print("3. CIFAR-10 (32x32, 10 classes) - Common Objects")
    print("4. CIFAR-100 (32x32, 100 classes) - Fine-grained Classification")
    print("=" * 60)
    
    choice = input("\nSelect dataset (1-4, default=1): ").strip()
    
    if choice == '2':
        return prepare_mnist_dataset, 'MNIST'
    elif choice == '3':
        return prepare_cifar10_dataset, 'CIFAR-10'
    elif choice == '4':
        return prepare_cifar100_dataset, 'CIFAR-100'
    else:
        return prepare_digit_dataset, 'Digits'


def main():
    """Main function to train the MLP model for image classification."""
    
    print("=" * 60)
    print("MLP Training - Image Classification (PyTorch with GPU Support)")
    print("=" * 60)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Let user select dataset
    dataset_func, dataset_name = select_dataset()
    
    # Prepare image dataset
    print(f"\nPreparing {dataset_name} dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, input_size, num_classes = dataset_func()
    
    print(f"\nDataset Information ({dataset_name}):")
    print(f"  Input Features (dynamic): {input_size}")
    print(f"  Number of Classes (dynamic): {num_classes}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    
    # Create and build model
    mlp = MLPModel(input_size=input_size, num_classes=num_classes, device=device)
    mlp.compile_model()
    
    print(f"\nModel Architecture:")
    print(f"  Input Layer: {input_size} neurons")
    print(f"  Hidden Layers: {mlp.hidden_layers}")
    print(f"  Output Layer: {num_classes} neurons")
    print(f"  Device: {device}")
    print()
    
    mlp.summary()
    
    # Train the model
    print("\n" + "=" * 60)
    print("Training the MLP...")
    print("=" * 60)
    mlp.train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    mlp.evaluate(X_test, y_test)
    
    # Save the weights and model
    print("\n" + "=" * 60)
    print("Saving Model and Weights...")
    print("=" * 60)
    model_name = f'{dataset_name.lower().replace("-", "")}_classifier'
    mlp.save_weights(model_name=model_name)
    
    # Quantization and Integer Inference Comparison
    print("\n" + "=" * 60)
    print("Performing Quantization and Comparison...")
    print("=" * 60)
    comparison_results = compare_quantization(mlp, X_test, y_test, scaling_factor=2**16, batch_size=32)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
