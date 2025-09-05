import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_diabetes, fetch_california_housing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionGD:
    """
    Linear Regression implementation using Gradient Descent
    
    Supports:
    - Multiple features
    - Regularization (Ridge, Lasso, Elastic Net)
    - Different gradient descent variants
    - Comprehensive visualization
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, 
                 regularization=None, lambda_reg=0.01, l1_ratio=0.5):
        """
        Initialize Linear Regression with Gradient Descent
        
        Args:
            learning_rate (float): Step size for gradient descent
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            regularization (str): 'ridge', 'lasso', 'elastic_net', or None
            lambda_reg (float): Regularization strength
            l1_ratio (float): L1 ratio for elastic net (0=ridge, 1=lasso)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.l1_ratio = l1_ratio
        
        # Model parameters
        self.weights = None
        self.bias = None
        
        # Training history
        self.cost_history = []
        self.weight_history = []
        
    def _add_bias(self, X):
        """Add bias column to feature matrix"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_cost(self, X, y, weights):
        """Compute cost function with regularization"""
        m = X.shape[0]
        
        # Predictions
        predictions = X @ weights
        
        # Mean squared error
        mse = np.sum((predictions - y) ** 2) / (2 * m)
        
        # Add regularization
        reg_term = 0
        if self.regularization == 'ridge':
            reg_term = self.lambda_reg * np.sum(weights[1:] ** 2) / 2
        elif self.regularization == 'lasso':
            reg_term = self.lambda_reg * np.sum(np.abs(weights[1:]))
        elif self.regularization == 'elastic_net':
            l1_term = self.l1_ratio * np.sum(np.abs(weights[1:]))
            l2_term = (1 - self.l1_ratio) * np.sum(weights[1:] ** 2) / 2
            reg_term = self.lambda_reg * (l1_term + l2_term)
        
        return mse + reg_term
    
    def _compute_gradients(self, X, y, weights):
        """Compute gradients with regularization"""
        m = X.shape[0]
        
        # Predictions
        predictions = X @ weights
        
        # Basic gradient
        gradients = X.T @ (predictions - y) / m
        
        # Add regularization gradients
        if self.regularization == 'ridge':
            gradients[1:] += self.lambda_reg * weights[1:]
        elif self.regularization == 'lasso':
            gradients[1:] += self.lambda_reg * np.sign(weights[1:])
        elif self.regularization == 'elastic_net':
            l1_grad = self.l1_ratio * np.sign(weights[1:])
            l2_grad = (1 - self.l1_ratio) * weights[1:]
            gradients[1:] += self.lambda_reg * (l1_grad + l2_grad)
        
        return gradients
    
    def fit(self, X, y, verbose=False):
        """Train the linear regression model"""
        # Add bias term
        X_with_bias = self._add_bias(X)
        m, n = X_with_bias.shape
        
        # Initialize weights
        self.weights = np.random.normal(0, 0.01, n)
        
        # Initialize history
        self.cost_history = []
        self.weight_history = []
        
        prev_cost = float('inf')
        
        for iteration in range(self.max_iterations):
            # Compute cost
            cost = self._compute_cost(X_with_bias, y, self.weights)
            self.cost_history.append(cost)
            self.weight_history.append(self.weights.copy())
            
            # Compute gradients
            gradients = self._compute_gradients(X_with_bias, y, self.weights)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Check convergence
            if abs(prev_cost - cost) < self.tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            prev_cost = cost
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}, Cost: {cost:.6f}")
        
        # Store final parameters
        self.bias = self.weights[0]
        self.weights = self.weights[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def plot_training_progress(self):
        """Plot cost function during training"""
        plt.figure(figsize=(12, 4))
        
        # Cost history
        plt.subplot(1, 2, 1)
        plt.plot(self.cost_history, 'b-', linewidth=2)
        plt.title('Training Cost Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
        
        # Weight evolution (for first few weights)
        plt.subplot(1, 2, 2)
        weight_history = np.array(self.weight_history)
        
        # Plot evolution of first 5 weights (including bias)
        n_weights_to_plot = min(5, weight_history.shape[1])
        for i in range(n_weights_to_plot):
            label = 'bias' if i == 0 else f'w{i}'
            plt.plot(weight_history[:, i], label=label, linewidth=2)
        
        plt.title('Weight Evolution During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, X, y, title="Predictions vs Actual"):
        """Plot predictions vs actual values"""
        predictions = self.predict(X)
        
        plt.figure(figsize=(12, 4))
        
        # Scatter plot of predictions vs actual
        plt.subplot(1, 2, 1)
        plt.scatter(y, predictions, alpha=0.6, color='blue')
        
        # Perfect prediction line
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{title} - Scatter Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y - predictions
        plt.scatter(predictions, residuals, alpha=0.6, color='red')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
        
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{title} - Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

def generate_synthetic_data(n_samples=100, n_features=1, noise=10, random_state=42):
    """Generate synthetic regression data"""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=noise, random_state=random_state)
    return X, y

def load_real_dataset(dataset_name='diabetes'):
    """Load real-world datasets"""
    if dataset_name == 'diabetes':
        data = load_diabetes()
        return data.data, data.target
    elif dataset_name == 'boston' or dataset_name == 'california':
        # Boston dataset is deprecated, using California housing dataset instead
        data = fetch_california_housing()
        return data.data, data.target
    else:
        raise ValueError("Unknown dataset name")

def compare_regularization_methods(X, y):
    """Compare different regularization methods"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Different regularization methods
    methods = {
        'No Regularization': {'regularization': None},
        'Ridge (L2)': {'regularization': 'ridge', 'lambda_reg': 0.1},
        'Lasso (L1)': {'regularization': 'lasso', 'lambda_reg': 0.1},
        'Elastic Net': {'regularization': 'elastic_net', 'lambda_reg': 0.1, 'l1_ratio': 0.5}
    }
    
    results = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, params) in enumerate(methods.items()):
        # Train model
        model = LinearRegressionGD(learning_rate=0.01, max_iterations=1000, **params)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'model': model
        }
        
        # Plot cost history
        plt.subplot(2, 2, i+1)
        plt.plot(model.cost_history, linewidth=2)
        plt.title(f'{name}\nTrain R²: {train_r2:.3f}, Test R²: {test_r2:.3f}')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\nRegularization Method Comparison:")
    print("-" * 80)
    print(f"{'Method':<15} {'Train R²':<10} {'Test R²':<10} {'Train MSE':<12} {'Test MSE':<12}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['train_r2']:<10.4f} {metrics['test_r2']:<10.4f} "
              f"{metrics['train_mse']:<12.2f} {metrics['test_mse']:<12.2f}")
    
    return results

def learning_rate_analysis(X, y):
    """Analyze effect of different learning rates"""
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        model = LinearRegressionGD(learning_rate=lr, max_iterations=1000)
        model.fit(X_scaled, y)
        
        plt.subplot(2, 2, i+1)
        plt.plot(model.cost_history, linewidth=2, color='blue')
        plt.title(f'Learning Rate = {lr}')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True, alpha=0.3)
        
        # Add final metrics
        final_r2 = model.score(X_scaled, y)
        plt.text(0.7, 0.9, f'Final R²: {final_r2:.3f}', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def demonstrate_linear_regression():
    """Comprehensive demonstration of Linear Regression with Gradient Descent"""
    print("=" * 70)
    print("LINEAR REGRESSION WITH GRADIENT DESCENT DEMONSTRATION")
    print("=" * 70)
    
    # Example 1: Simple 1D regression
    print("\n1. Simple 1D Linear Regression")
    print("-" * 50)
    
    X_1d, y_1d = generate_synthetic_data(n_samples=100, n_features=1, noise=10)
    
    model_1d = LinearRegressionGD(learning_rate=0.01, max_iterations=1000)
    model_1d.fit(X_1d, y_1d, verbose=True)
    
    print(f"Final weight: {model_1d.weights[0]:.4f}")
    print(f"Final bias: {model_1d.bias:.4f}")
    print(f"R² Score: {model_1d.score(X_1d, y_1d):.4f}")
    
    # Plot 1D regression
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_1d, y_1d, alpha=0.6, color='blue', label='Data')
    
    # Plot regression line
    X_line = np.linspace(X_1d.min(), X_1d.max(), 100).reshape(-1, 1)
    y_line = model_1d.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='Regression Line')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('1D Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(model_1d.cost_history, 'b-', linewidth=2)
    plt.title('Cost Function Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Example 2: Multi-dimensional regression with real data
    print("\n2. Multi-dimensional Regression (Diabetes Dataset)")
    print("-" * 50)
    
    X_diabetes, y_diabetes = load_real_dataset('diabetes')
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, 
                                                        test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_multi = LinearRegressionGD(learning_rate=0.01, max_iterations=2000)
    model_multi.fit(X_train_scaled, y_train, verbose=True)
    
    print(f"Number of features: {X_diabetes.shape[1]}")
    print(f"Training R² Score: {model_multi.score(X_train_scaled, y_train):.4f}")
    print(f"Test R² Score: {model_multi.score(X_test_scaled, y_test):.4f}")
    
    # Plot training progress and predictions
    model_multi.plot_training_progress()
    model_multi.plot_predictions(X_test_scaled, y_test, "Test Set")
    
    # Example 3: Regularization comparison
    print("\n3. Regularization Methods Comparison")
    print("-" * 50)
    
    reg_results = compare_regularization_methods(X_diabetes, y_diabetes)
    
    # Example 4: Learning rate analysis
    print("\n4. Learning Rate Analysis")
    print("-" * 50)
    
    learning_rate_analysis(X_diabetes, y_diabetes)
    
    # Example 5: Feature importance (weight magnitude)
    print("\n5. Feature Importance Analysis")
    print("-" * 50)
    
    # Use the no-regularization model for feature importance
    feature_names = [f'Feature_{i+1}' for i in range(X_diabetes.shape[1])]
    weights = reg_results['No Regularization']['model'].weights
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    indices = np.argsort(np.abs(weights))[::-1]
    
    plt.bar(range(len(weights)), np.abs(weights)[indices])
    plt.title('Feature Importance (Absolute Weight Values)')
    plt.xlabel('Features')
    plt.ylabel('Absolute Weight')
    plt.xticks(range(len(weights)), [feature_names[i] for i in indices], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nTop 5 Most Important Features:")
    for i in range(min(5, len(weights))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]}: {weights[idx]:.4f}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    demonstrate_linear_regression()