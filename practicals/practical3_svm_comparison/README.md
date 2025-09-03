# Practical 3: SVM Classification Accuracy Comparison

## Overview
This practical provides a comprehensive comparison of Support Vector Machine (SVM) classification performance across different kernels, hyperparameters, and datasets. The implementation includes extensive analysis tools, visualization capabilities, and performance evaluation metrics.

## Implementation Features

### Core SVMComparison Class
- **Multiple Kernel Support**: Linear, Polynomial, RBF, and Sigmoid kernels
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Validation Curves**: Parameter sensitivity analysis
- **Multi-Dataset Analysis**: Comparison across various datasets
- **Comprehensive Visualization**: Performance plots and confusion matrices

### Key Methods
- `compare_kernels(X, y)`: Compare different SVM kernels
- `hyperparameter_tuning(X, y, kernel)`: Optimize hyperparameters
- `validation_curves_analysis(X, y, kernel, param_name)`: Parameter sensitivity
- `comprehensive_analysis(dataset_name)`: Complete analysis pipeline
- `plot_kernel_comparison(results)`: Visualize kernel performance
- `plot_validation_curves(results)`: Plot parameter validation curves

## Mathematical Foundation

### SVM Optimization Problem
```
Minimize: (1/2)||w||² + C∑ξᵢ
Subject to: yᵢ(w·φ(xᵢ) + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

### Kernel Functions

#### Linear Kernel
```
K(xᵢ, xⱼ) = xᵢ · xⱼ
```

#### Polynomial Kernel
```
K(xᵢ, xⱼ) = (γ(xᵢ · xⱼ) + r)^d
```

#### RBF (Radial Basis Function) Kernel
```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
```

#### Sigmoid Kernel
```
K(xᵢ, xⱼ) = tanh(γ(xᵢ · xⱼ) + r)
```

## Supported Datasets

### Built-in Datasets
1. **Iris Dataset**: 150 samples, 4 features, 3 classes
2. **Wine Dataset**: 178 samples, 13 features, 3 classes
3. **Breast Cancer Dataset**: 569 samples, 30 features, 2 classes
4. **Synthetic Dataset**: 1000 samples, 20 features, customizable classes

### Dataset Characteristics
- **Iris**: Classic multi-class classification, well-separated classes
- **Wine**: Chemical analysis features, moderate complexity
- **Breast Cancer**: Medical diagnosis, high-dimensional features
- **Synthetic**: Controlled complexity for experimentation

## Usage Examples

### Basic Kernel Comparison
```python
from svm_comparison import SVMComparison

# Initialize comparison tool
svm_comp = SVMComparison(random_state=42)

# Load dataset
X, y, feature_names, target_names = svm_comp.load_dataset('iris')

# Compare different kernels
kernel_results = svm_comp.compare_kernels(X, y)

# Visualize results
svm_comp.plot_kernel_comparison(kernel_results)
```

### Hyperparameter Tuning
```python
# Tune RBF kernel hyperparameters
tuning_results = svm_comp.hyperparameter_tuning(X, y, kernel='rbf')

print(f"Best parameters: {tuning_results['best_params']}")
print(f"Best CV score: {tuning_results['best_score']:.4f}")
print(f"Test accuracy: {tuning_results['test_accuracy']:.4f}")
```

### Validation Curve Analysis
```python
# Analyze C parameter sensitivity
c_validation = svm_comp.validation_curves_analysis(
    X, y, kernel='rbf', param_name='C', 
    param_range=[0.1, 1, 10, 100, 1000]
)

# Plot validation curves
svm_comp.plot_validation_curves(c_validation, "(RBF Kernel)")
```

### Comprehensive Analysis
```python
# Complete analysis pipeline
results = svm_comp.comprehensive_analysis('iris')

# Results include:
# - Kernel comparison
# - Hyperparameter tuning
# - Validation curves
# - Performance metrics
```

## Hyperparameter Grids

### Linear Kernel
- **C**: [0.1, 1, 10, 100]

### Polynomial Kernel
- **C**: [0.1, 1, 10, 100]
- **degree**: [2, 3, 4]
- **gamma**: ['scale', 'auto', 0.001, 0.01, 0.1, 1]

### RBF Kernel
- **C**: [0.1, 1, 10, 100, 1000]
- **gamma**: ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]

### Sigmoid Kernel
- **C**: [0.1, 1, 10, 100]
- **gamma**: ['scale', 'auto', 0.001, 0.01, 0.1, 1]

## Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

### Cross-Validation
- **K-Fold CV**: Robust performance estimation
- **Stratified CV**: Maintains class distribution
- **Mean ± Standard Deviation**: Statistical significance

## Visualization Features

### Kernel Comparison Plots
1. **Accuracy Comparison**: Train vs. test accuracy by kernel
2. **Cross-Validation Scores**: Mean accuracy with error bars
3. **Confusion Matrix**: Best performing kernel
4. **Performance Metrics**: Precision, recall, F1-score comparison

### Validation Curves
1. **Parameter Sensitivity**: Training vs. validation scores
2. **Optimal Parameter**: Identification of best values
3. **Overfitting Detection**: Gap between training and validation

### Multi-Dataset Analysis
1. **Accuracy by Dataset**: Best performance comparison
2. **Dataset Characteristics**: Size vs. accuracy relationship
3. **Kernel Distribution**: Most effective kernels
4. **Complexity Analysis**: Features vs. classes visualization

## Key Learning Outcomes

### Understanding SVM Kernels
- **Linear**: Best for linearly separable data
- **Polynomial**: Captures polynomial relationships
- **RBF**: Most versatile, handles non-linear patterns
- **Sigmoid**: Neural network-like decision boundaries

### Hyperparameter Effects
- **C Parameter**: Controls regularization strength
  - Small C: More regularization, simpler model
  - Large C: Less regularization, complex model
- **Gamma Parameter**: Controls kernel coefficient
  - Small gamma: Wider influence, smoother boundaries
  - Large gamma: Narrow influence, complex boundaries
- **Degree Parameter**: Polynomial complexity
  - Higher degree: More complex polynomial relationships

### Model Selection Insights
- **Cross-validation**: Essential for reliable performance estimation
- **Validation curves**: Identify optimal hyperparameters
- **Overfitting detection**: Monitor training vs. validation gap
- **Dataset characteristics**: Influence kernel choice

## Advanced Analysis Features

### Multi-Dataset Comparison
```python
# Compare SVM performance across multiple datasets
from svm_comparison import compare_multiple_datasets

summary_df = compare_multiple_datasets()
print(summary_df)
```

### Custom Dataset Integration
```python
# Add your own dataset
def load_custom_dataset():
    # Load your data
    X, y = load_your_data()
    feature_names = ['feature_1', 'feature_2', ...]
    target_names = ['class_1', 'class_2', ...]
    return X, y, feature_names, target_names

# Integrate with SVMComparison
svm_comp = SVMComparison()
X, y, features, targets = load_custom_dataset()
results = svm_comp.compare_kernels(X, y)
```

## Performance Optimization Tips

### Feature Scaling
- **Always scale features** for SVM (except linear kernel with normalized data)
- **StandardScaler**: Zero mean, unit variance
- **MinMaxScaler**: Alternative for bounded features

### Kernel Selection Guidelines
- **Linear**: High-dimensional data, text classification
- **RBF**: General-purpose, unknown data distribution
- **Polynomial**: Known polynomial relationships
- **Sigmoid**: Rare, specific neural network-like problems

### Hyperparameter Tuning Strategy
1. **Coarse grid search**: Wide parameter range
2. **Fine grid search**: Narrow range around optimal values
3. **Random search**: Alternative for high-dimensional parameter space
4. **Bayesian optimization**: Advanced optimization technique

## Dependencies
```
numpy
matplotlib
seaborn
scikit-learn
pandas
```

## Installation
```bash
pip install numpy matplotlib seaborn scikit-learn pandas
```

## Running the Demonstration
```bash
python svm_comparison.py
```

## Expected Output
The demonstration will show:
1. **Iris dataset analysis**: Complete SVM comparison
2. **Kernel performance**: Accuracy and metrics comparison
3. **Hyperparameter tuning**: Optimal RBF parameters
4. **Validation curves**: C and gamma parameter analysis
5. **Multi-dataset summary**: Performance across datasets
6. **Visualization plots**: Comprehensive performance analysis

## Extensions and Improvements

### Possible Enhancements
1. **Custom Kernels**: Implement domain-specific kernels
2. **Feature Selection**: Integrate feature selection methods
3. **Ensemble Methods**: Combine multiple SVM models
4. **Online Learning**: Implement incremental SVM
5. **Probability Calibration**: Add probability estimates
6. **Multi-class Strategies**: Compare one-vs-one vs. one-vs-rest

### Advanced Topics
1. **Support Vector Regression**: Extend to regression problems
2. **Novelty Detection**: One-class SVM implementation
3. **Large-scale SVM**: Handle big data with SGD
4. **Kernel PCA**: Dimensionality reduction with kernels
5. **Semi-supervised SVM**: Utilize unlabeled data

## Troubleshooting

### Common Issues
1. **Convergence warnings**: Increase max_iter or scale features
2. **Poor performance**: Check feature scaling and parameter tuning
3. **Memory issues**: Use smaller parameter grids or sample data
4. **Slow training**: Consider linear kernel or feature reduction

### Performance Tips
1. **Feature scaling**: Essential for non-linear kernels
2. **Parameter bounds**: Use reasonable ranges for grid search
3. **Cross-validation**: Balance between accuracy and computation time
4. **Parallel processing**: Use n_jobs=-1 for faster computation

This implementation provides a comprehensive foundation for understanding SVM classification and serves as a practical tool for comparing different SVM configurations across various datasets.