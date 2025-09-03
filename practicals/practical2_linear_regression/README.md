# Practical 2: Linear Regression using Gradient Descent

## Overview
This practical implements Linear Regression using Gradient Descent from scratch, providing a comprehensive understanding of how gradient-based optimization works in machine learning. The implementation includes multiple features, regularization techniques, and extensive visualization capabilities.

## Implementation Features

### Core LinearRegressionGD Class
- **Multiple Feature Support**: Handles both single and multi-dimensional regression
- **Regularization Options**: Ridge (L2), Lasso (L1), and Elastic Net regularization
- **Convergence Monitoring**: Tracks cost function and weight evolution during training
- **Comprehensive Metrics**: MSE, MAE, and R² score calculations

### Key Methods
- `fit(X, y, verbose=False)`: Train the model using gradient descent
- `predict(X)`: Make predictions on new data
- `score(X, y)`: Calculate R² score
- `plot_training_progress()`: Visualize cost and weight evolution
- `plot_predictions(X, y)`: Show predictions vs actual values with residual plots

## Mathematical Foundation

### Cost Function
```
J(θ) = (1/2m) * Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + Regularization Term
```

### Gradient Update Rule
```
θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ
```

### Regularization Terms
- **Ridge (L2)**: λ * Σθⱼ²
- **Lasso (L1)**: λ * Σ|θⱼ|
- **Elastic Net**: λ * [ρ * Σ|θⱼ| + (1-ρ) * Σθⱼ²]

## Usage Examples

### Basic Linear Regression
```python
from linear_regression_gd import LinearRegressionGD, generate_synthetic_data

# Generate sample data
X, y = generate_synthetic_data(n_samples=100, n_features=1, noise=10)

# Create and train model
model = LinearRegressionGD(learning_rate=0.01, max_iterations=1000)
model.fit(X, y, verbose=True)

# Make predictions
predictions = model.predict(X)
print(f"R² Score: {model.score(X, y):.4f}")

# Visualize results
model.plot_training_progress()
model.plot_predictions(X, y)
```

### Multi-dimensional Regression with Regularization
```python
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load real dataset
X, y = load_diabetes(return_X_y=True)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with Ridge regularization
model = LinearRegressionGD(
    learning_rate=0.01,
    max_iterations=2000,
    regularization='ridge',
    lambda_reg=0.1
)
model.fit(X_train_scaled, y_train)

# Evaluate
print(f"Test R² Score: {model.score(X_test_scaled, y_test):.4f}")
```

## Demonstration Functions

### 1. `demonstrate_linear_regression()`
Comprehensive demonstration including:
- 1D regression with visualization
- Multi-dimensional regression on diabetes dataset
- Regularization method comparison
- Learning rate analysis
- Feature importance analysis

### 2. `compare_regularization_methods(X, y)`
Compares different regularization techniques:
- No regularization
- Ridge (L2) regularization
- Lasso (L1) regularization
- Elastic Net regularization

### 3. `learning_rate_analysis(X, y)`
Analyzes the effect of different learning rates on convergence

## Key Learning Outcomes

### Understanding Gradient Descent
- How gradient descent optimizes the cost function
- Impact of learning rate on convergence speed and stability
- Convergence criteria and stopping conditions

### Regularization Effects
- **Ridge**: Shrinks coefficients, handles multicollinearity
- **Lasso**: Can zero out coefficients, performs feature selection
- **Elastic Net**: Combines benefits of both Ridge and Lasso

### Feature Scaling Importance
- Why standardization is crucial for gradient descent
- Impact on convergence speed and final results

### Model Evaluation
- Training vs. test performance
- Overfitting detection through regularization
- Residual analysis for model diagnostics

## Parameters and Tuning

### Learning Rate (α)
- **Too small**: Slow convergence
- **Too large**: Oscillation or divergence
- **Optimal**: Fast, stable convergence

### Regularization Strength (λ)
- **λ = 0**: No regularization (may overfit)
- **Small λ**: Light regularization
- **Large λ**: Strong regularization (may underfit)

### Maximum Iterations
- Should be sufficient for convergence
- Monitor cost function to determine adequacy

## Visualization Features

### Training Progress Plots
- Cost function evolution over iterations
- Weight parameter evolution during training

### Prediction Analysis
- Scatter plot of predictions vs. actual values
- Residual plots for error analysis
- Perfect prediction reference line

### Comparative Analysis
- Side-by-side regularization method comparison
- Learning rate effect visualization
- Feature importance bar charts

## Dependencies
```
numpy
matplotlib
scikit-learn
```

## Installation
```bash
pip install numpy matplotlib scikit-learn
```

## Running the Demonstration
```bash
python linear_regression_gd.py
```

## Expected Output
The demonstration will show:
1. 1D regression with line fitting
2. Multi-dimensional regression performance metrics
3. Regularization method comparison table
4. Learning rate analysis plots
5. Feature importance rankings

## Extensions and Improvements

### Possible Enhancements
1. **Adaptive Learning Rate**: Implement learning rate scheduling
2. **Momentum**: Add momentum to gradient descent
3. **Mini-batch GD**: Implement stochastic and mini-batch variants
4. **Cross-validation**: Add k-fold cross-validation for hyperparameter tuning
5. **Polynomial Features**: Extend to polynomial regression
6. **Early Stopping**: Implement early stopping based on validation loss

### Advanced Topics
1. **Coordinate Descent**: Alternative optimization for Lasso
2. **Proximal Gradient**: For non-smooth regularization
3. **Feature Engineering**: Automatic feature creation and selection
4. **Robust Regression**: Handle outliers with robust loss functions

## Mathematical Insights

### Gradient Computation
For linear regression with regularization:
```
∂J/∂θ₀ = (1/m) * Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)  [bias term]
∂J/∂θⱼ = (1/m) * Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾ + regularization_gradient
```

### Convergence Analysis
- **Convex optimization**: Linear regression has a global minimum
- **Learning rate bounds**: For convergence, α < 2/λmax(XᵀX)
- **Condition number**: Affects convergence speed

This implementation provides a solid foundation for understanding gradient-based optimization in machine learning and serves as a stepping stone to more advanced optimization algorithms.