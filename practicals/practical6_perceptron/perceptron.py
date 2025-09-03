import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class LinearPerceptron:
    """
    Linear Perceptron Learning Algorithm Implementation
    
    A comprehensive implementation of the classic Perceptron algorithm
    with various learning rules, visualization capabilities, and analysis tools.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000, 
                 random_state: Optional[int] = None, tolerance: float = 1e-6):
        """
        Initialize the Linear Perceptron
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for weight updates
        max_epochs : int, default=1000
            Maximum number of training epochs
        random_state : int, optional
            Random seed for reproducibility
        tolerance : float, default=1e-6
            Convergence tolerance for early stopping
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.tolerance = tolerance
        
        # Training history
        self.weights_history = []
        self.bias_history = []
        self.error_history = []
        self.accuracy_history = []
        self.converged = False
        self.convergence_epoch = None
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.n_features = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_weights(self, n_features: int) -> None:
        """
        Initialize weights and bias
        
        Parameters:
        -----------
        n_features : int
            Number of input features
        """
        self.n_features = n_features
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        # Reset history
        self.weights_history = [self.weights.copy()]
        self.bias_history = [self.bias]
        self.error_history = []
        self.accuracy_history = []
    
    def _activation(self, z: np.ndarray) -> np.ndarray:
        """
        Step activation function
        
        Parameters:
        -----------
        z : np.ndarray
            Linear combination of inputs
            
        Returns:
        --------
        np.ndarray
            Binary predictions (0 or 1)
        """
        return np.where(z >= 0, 1, 0)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute linear combination (before activation)
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        np.ndarray
            Linear combination values
        """
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        np.ndarray
            Binary predictions
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        z = self._predict_proba(X)
        return self._activation(z)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'LinearPerceptron':
        """
        Train the perceptron using the perceptron learning rule
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels (0 or 1)
        verbose : bool, default=False
            Print training progress
            
        Returns:
        --------
        self : LinearPerceptron
            Trained perceptron
        """
        # Ensure binary labels
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("Perceptron requires exactly 2 classes")
        
        # Convert labels to 0 and 1 if necessary
        if not np.array_equal(unique_labels, [0, 1]):
            y = np.where(y == unique_labels[0], 0, 1)
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        if verbose:
            print(f"Training Perceptron with {n_samples} samples, {n_features} features")
            print(f"Learning rate: {self.learning_rate}, Max epochs: {self.max_epochs}")
        
        for epoch in range(self.max_epochs):
            # Make predictions
            predictions = self.predict(X)
            
            # Calculate errors
            errors = y - predictions
            n_errors = np.sum(errors != 0)
            
            # Calculate accuracy
            accuracy = 1.0 - (n_errors / n_samples)
            
            # Store history
            self.error_history.append(n_errors)
            self.accuracy_history.append(accuracy)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: Errors = {n_errors:3d}, Accuracy = {accuracy:.4f}")
            
            # Check for convergence
            if n_errors == 0:
                self.converged = True
                self.convergence_epoch = epoch
                if verbose:
                    print(f"Converged at epoch {epoch}!")
                break
            
            # Update weights and bias using perceptron learning rule
            for i in range(n_samples):
                if errors[i] != 0:
                    # Update rule: w = w + η * error * x
                    self.weights += self.learning_rate * errors[i] * X[i]
                    self.bias += self.learning_rate * errors[i]
            
            # Store updated weights
            self.weights_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            
            # Check for convergence based on weight changes
            if len(self.weights_history) > 1:
                weight_change = np.linalg.norm(self.weights_history[-1] - self.weights_history[-2])
                if weight_change < self.tolerance:
                    self.converged = True
                    self.convergence_epoch = epoch
                    if verbose:
                        print(f"Converged at epoch {epoch} (weight change < tolerance)!")
                    break
        
        if not self.converged and verbose:
            print(f"Did not converge after {self.max_epochs} epochs")
        
        return self
    
    def plot_training_history(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot training history including errors, accuracy, and weight evolution
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 5)
            Figure size for the plots
        """
        if not self.error_history:
            raise ValueError("No training history available. Train the model first.")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        epochs = range(len(self.error_history))
        
        # Plot errors
        axes[0].plot(epochs, self.error_history, 'b-', linewidth=2, label='Training Errors')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Number of Errors')
        axes[0].set_title('Training Errors Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        if self.converged:
            axes[0].axvline(x=self.convergence_epoch, color='red', linestyle='--', 
                           label=f'Convergence (Epoch {self.convergence_epoch})')
            axes[0].legend()
        
        # Plot accuracy
        axes[1].plot(epochs, self.accuracy_history, 'g-', linewidth=2, label='Training Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy Over Time')
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        if self.converged:
            axes[1].axvline(x=self.convergence_epoch, color='red', linestyle='--', 
                           label=f'Convergence (Epoch {self.convergence_epoch})')
            axes[1].legend()
        
        # Plot weight evolution (for 2D case)
        if self.n_features == 2:
            weight_history = np.array(self.weights_history)
            axes[2].plot(weight_history[:, 0], weight_history[:, 1], 'ro-', 
                        markersize=3, linewidth=1, alpha=0.7)
            axes[2].scatter(weight_history[0, 0], weight_history[0, 1], 
                           c='green', s=100, marker='s', label='Start', zorder=5)
            axes[2].scatter(weight_history[-1, 0], weight_history[-1, 1], 
                           c='red', s=100, marker='*', label='End', zorder=5)
            axes[2].set_xlabel('Weight 1')
            axes[2].set_ylabel('Weight 2')
            axes[2].set_title('Weight Evolution in 2D')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
        else:
            # Plot weight magnitudes for higher dimensions
            weight_norms = [np.linalg.norm(w) for w in self.weights_history]
            axes[2].plot(range(len(weight_norms)), weight_norms, 'purple', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Weight Norm')
            axes[2].set_title('Weight Vector Magnitude')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, 
                              title: str = "Perceptron Decision Boundary",
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot decision boundary for 2D data
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (must be 2D)
        y : np.ndarray
            True labels
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plotting only supports 2D data")
        
        if self.weights is None:
            raise ValueError("Model must be trained before plotting decision boundary")
        
        plt.figure(figsize=figsize)
        
        # Create a mesh for plotting decision boundary
        h = 0.02  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        
        # Plot decision line
        if abs(self.weights[1]) > 1e-6:  # Avoid division by zero
            x_line = np.linspace(x_min, x_max, 100)
            y_line = -(self.weights[0] * x_line + self.bias) / self.weights[1]
            plt.plot(x_line, y_line, 'k-', linewidth=2, label='Decision Boundary')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
        --------
        dict
            Model information and statistics
        """
        if self.weights is None:
            return {"status": "Model not trained"}
        
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "n_features": self.n_features,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "converged": self.converged,
            "convergence_epoch": self.convergence_epoch,
            "final_errors": self.error_history[-1] if self.error_history else None,
            "final_accuracy": self.accuracy_history[-1] if self.accuracy_history else None,
            "total_epochs": len(self.error_history),
            "weight_norm": np.linalg.norm(self.weights)
        }

def generate_linearly_separable_data(n_samples: int = 200, n_features: int = 2, 
                                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate linearly separable binary classification data
    
    Parameters:
    -----------
    n_samples : int, default=200
        Number of samples
    n_features : int, default=2
        Number of features
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    tuple
        Features and labels
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_redundant=0, n_informative=n_features,
                              n_clusters_per_class=1, random_state=random_state,
                              class_sep=2.0)  # Ensure good separation
    return X, y

def generate_non_linearly_separable_data(n_samples: int = 200, noise: float = 0.3,
                                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate non-linearly separable data (XOR-like pattern)
    
    Parameters:
    -----------
    n_samples : int, default=200
        Number of samples
    noise : float, default=0.3
        Noise level
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    tuple
        Features and labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create XOR-like pattern
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    
    # Add noise
    noise_mask = np.random.random(n_samples) < noise
    y[noise_mask] = 1 - y[noise_mask]
    
    return X, y

def compare_learning_rates(X: np.ndarray, y: np.ndarray, 
                          learning_rates: List[float] = [0.001, 0.01, 0.1, 1.0],
                          max_epochs: int = 500) -> None:
    """
    Compare perceptron performance with different learning rates
    
    Parameters:
    -----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training labels
    learning_rates : list
        Learning rates to compare
    max_epochs : int
        Maximum training epochs
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    results = []
    
    for i, lr in enumerate(learning_rates):
        # Train perceptron
        perceptron = LinearPerceptron(learning_rate=lr, max_epochs=max_epochs, random_state=42)
        perceptron.fit(X, y)
        
        # Store results
        info = perceptron.get_model_info()
        results.append({
            'learning_rate': lr,
            'converged': info['converged'],
            'convergence_epoch': info['convergence_epoch'],
            'final_accuracy': info['final_accuracy'],
            'total_epochs': info['total_epochs']
        })
        
        # Plot training history
        epochs = range(len(perceptron.error_history))
        axes[i].plot(epochs, perceptron.accuracy_history, 'b-', linewidth=2)
        axes[i].set_title(f'Learning Rate: {lr}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Accuracy')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1.05)
        
        if perceptron.converged:
            axes[i].axvline(x=perceptron.convergence_epoch, color='red', linestyle='--',
                           label=f'Convergence (Epoch {perceptron.convergence_epoch})')
            axes[i].legend()
    
    plt.suptitle('Learning Rate Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("LEARNING RATE COMPARISON RESULTS")
    print("="*80)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Find best learning rate
    converged_results = df[df['converged'] == True]
    if not converged_results.empty:
        best_lr = converged_results.loc[converged_results['convergence_epoch'].idxmin(), 'learning_rate']
        print(f"\nBest learning rate (fastest convergence): {best_lr}")
    else:
        best_lr = df.loc[df['final_accuracy'].idxmax(), 'learning_rate']
        print(f"\nBest learning rate (highest accuracy): {best_lr}")

def analyze_dataset_separability(X: np.ndarray, y: np.ndarray, dataset_name: str) -> None:
    """
    Analyze whether a dataset is linearly separable using perceptron
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    dataset_name : str
        Name of the dataset
    """
    print(f"\n" + "="*60)
    print(f"ANALYZING DATASET: {dataset_name.upper()}")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train perceptron
    perceptron = LinearPerceptron(learning_rate=0.1, max_epochs=1000, random_state=42)
    perceptron.fit(X_scaled, y, verbose=True)
    
    # Make predictions
    y_pred = perceptron.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    # Print results
    info = perceptron.get_model_info()
    print(f"\nResults:")
    print(f"  Final Accuracy: {accuracy:.4f}")
    print(f"  Converged: {info['converged']}")
    if info['converged']:
        print(f"  Convergence Epoch: {info['convergence_epoch']}")
        print(f"  Dataset is LINEARLY SEPARABLE")
    else:
        print(f"  Total Epochs: {info['total_epochs']}")
        print(f"  Dataset is NOT LINEARLY SEPARABLE (or needs more epochs)")
    
    print(f"  Final Weights: {info['weights']}")
    print(f"  Final Bias: {info['bias']:.4f}")
    
    # Plot results if 2D
    if X.shape[1] == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot decision boundary
        plt.sca(axes[0])
        perceptron.plot_decision_boundary(X_scaled, y, f"{dataset_name} - Decision Boundary")
        
        # Plot training history
        plt.sca(axes[1])
        perceptron.plot_training_history(figsize=(5, 6))
    else:
        # Just plot training history for higher dimensions
        perceptron.plot_training_history()
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred))
    
    return perceptron, accuracy

def demonstrate_perceptron_variants() -> None:
    """
    Demonstrate different perceptron scenarios and datasets
    """
    print("\n" + "="*80)
    print("PERCEPTRON LEARNING ALGORITHM DEMONSTRATION")
    print("="*80)
    
    # 1. Linearly Separable Data
    print("\n1. LINEARLY SEPARABLE DATA")
    print("-" * 40)
    X_sep, y_sep = generate_linearly_separable_data(n_samples=200, random_state=42)
    analyze_dataset_separability(X_sep, y_sep, "Linearly Separable")
    
    # 2. Non-linearly Separable Data
    print("\n2. NON-LINEARLY SEPARABLE DATA (XOR-like)")
    print("-" * 50)
    X_nonsep, y_nonsep = generate_non_linearly_separable_data(n_samples=200, noise=0.2, random_state=42)
    analyze_dataset_separability(X_nonsep, y_nonsep, "Non-linearly Separable")
    
    # 3. Real Dataset - Iris (first 2 classes)
    print("\n3. REAL DATASET - IRIS (Setosa vs Versicolor)")
    print("-" * 50)
    iris = load_iris()
    # Use only first 2 classes and first 2 features for visualization
    mask = iris.target < 2
    X_iris = iris.data[mask][:, :2]  # First 2 features
    y_iris = iris.target[mask]
    analyze_dataset_separability(X_iris, y_iris, "Iris (Setosa vs Versicolor)")
    
    # 4. Learning Rate Comparison
    print("\n4. LEARNING RATE COMPARISON")
    print("-" * 35)
    compare_learning_rates(X_sep, y_sep)
    
    # 5. High-dimensional data
    print("\n5. HIGH-DIMENSIONAL DATA")
    print("-" * 30)
    X_high, y_high = generate_linearly_separable_data(n_samples=300, n_features=10, random_state=42)
    analyze_dataset_separability(X_high, y_high, "High-dimensional (10D)")

def demonstrate_convergence_analysis() -> None:
    """
    Demonstrate convergence analysis with different scenarios
    """
    print("\n" + "="*80)
    print("PERCEPTRON CONVERGENCE ANALYSIS")
    print("="*80)
    
    scenarios = [
        {"name": "Easy Separation", "class_sep": 3.0, "n_samples": 100},
        {"name": "Moderate Separation", "class_sep": 1.5, "n_samples": 200},
        {"name": "Difficult Separation", "class_sep": 0.8, "n_samples": 300},
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario: {scenario['name']}")
        print("-" * 30)
        
        # Generate data
        X, y = make_classification(
            n_samples=scenario['n_samples'], 
            n_features=2, 
            n_redundant=0, 
            n_informative=2,
            n_clusters_per_class=1, 
            class_sep=scenario['class_sep'],
            random_state=42
        )
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train perceptron
        perceptron = LinearPerceptron(learning_rate=0.1, max_epochs=500, random_state=42)
        perceptron.fit(X_scaled, y)
        
        # Plot convergence
        epochs = range(len(perceptron.error_history))
        axes[i].plot(epochs, perceptron.error_history, 'b-', linewidth=2, label='Errors')
        axes[i].set_title(f'{scenario["name"]}\n(Class Sep: {scenario["class_sep"]})')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Number of Errors')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        if perceptron.converged:
            axes[i].axvline(x=perceptron.convergence_epoch, color='red', linestyle='--',
                           label=f'Convergence (Epoch {perceptron.convergence_epoch})')
            axes[i].legend()
            print(f"  Converged at epoch: {perceptron.convergence_epoch}")
        else:
            print(f"  Did not converge in 500 epochs")
        
        info = perceptron.get_model_info()
        print(f"  Final accuracy: {info['final_accuracy']:.4f}")
        print(f"  Final errors: {info['final_errors']}")
    
    plt.suptitle('Convergence Analysis: Effect of Class Separation', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run demonstrations
    demonstrate_perceptron_variants()
    demonstrate_convergence_analysis()
    
    print("\n" + "="*80)
    print("PERCEPTRON DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Perceptron can only learn linearly separable patterns")
    print("2. Learning rate affects convergence speed")
    print("3. Class separation difficulty affects convergence")
    print("4. Perceptron provides interpretable linear decision boundaries")
    print("5. Feature scaling can improve convergence")