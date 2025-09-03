# Practical 4: Feature Engineering and Representation

## Overview

This practical demonstrates comprehensive feature engineering and representation techniques in machine learning. Feature engineering is the process of creating, transforming, and selecting features to improve model performance and interpretability.

## Implementation Features

### FeatureEngineering Class
- **Feature Creation**: Polynomial features, statistical features, binning
- **Feature Transformation**: Scaling, normalization, mathematical transformations
- **Feature Selection**: Univariate selection, RFE, importance-based selection
- **Dimensionality Reduction**: PCA, ICA, t-SNE
- **Evaluation Framework**: Cross-validation, multiple classifiers
- **Visualization**: Feature importance, correlation matrices, dimensionality reduction plots

### Key Methods

#### Feature Creation
- `create_polynomial_features()`: Generate polynomial and interaction terms
- `create_statistical_features()`: Row-wise statistics, ratios, differences, transformations
- `create_binning_features()`: Discretization using uniform, quantile, or k-means strategies

#### Feature Selection
- `feature_selection_univariate()`: Statistical tests (F-test, mutual information)
- `feature_selection_rfe()`: Recursive Feature Elimination
- `feature_selection_importance()`: Tree-based feature importance

#### Dimensionality Reduction
- `dimensionality_reduction_pca()`: Principal Component Analysis
- `dimensionality_reduction_ica()`: Independent Component Analysis
- `dimensionality_reduction_tsne()`: t-Distributed Stochastic Neighbor Embedding

#### Scaling and Normalization
- `scale_features()`: StandardScaler, MinMaxScaler, RobustScaler

## Mathematical Foundations

### Polynomial Features
For features x₁, x₂, ..., xₙ, polynomial features of degree d include:
- **Linear terms**: x₁, x₂, ..., xₙ
- **Interaction terms**: x₁x₂, x₁x₃, ..., xᵢxⱼ
- **Higher-order terms**: x₁², x₂², ..., x₁x₂x₃, ...

### Statistical Features
- **Row-wise statistics**: mean, std, min, max, range, skewness, kurtosis
- **Pairwise operations**: ratios (xᵢ/xⱼ), differences (xᵢ - xⱼ)
- **Mathematical transformations**: log(x), √x, x²

### Principal Component Analysis (PCA)
PCA finds orthogonal components that maximize variance:
```
X_centered = X - mean(X)
C = (1/n) * X_centered^T * X_centered  # Covariance matrix
λ, v = eigendecomposition(C)           # Eigenvalues and eigenvectors
X_pca = X_centered * v                 # Transform to PC space
```

### Feature Selection Metrics
- **F-statistic**: Measures linear relationship between feature and target
- **Mutual Information**: Measures non-linear dependencies
- **Feature Importance**: Tree-based models provide importance scores

## Supported Datasets

1. **Iris**: 4 features, 3 classes (flower species)
2. **Wine**: 13 features, 3 classes (wine cultivars)
3. **Breast Cancer**: 30 features, 2 classes (malignant/benign)
4. **Synthetic**: Customizable features and classes

## Usage Examples

### Basic Feature Engineering
```python
from feature_engineering import FeatureEngineering

# Initialize
fe = FeatureEngineering(random_state=42)

# Load dataset
X, y, feature_names, target_names = fe.load_dataset('iris')

# Create polynomial features
X_poly, poly_names, poly_transformer = fe.create_polynomial_features(X, degree=2)

# Create statistical features
X_stat, stat_names = fe.create_statistical_features(X, feature_names)

# Combine features
X_combined = np.hstack([X, X_stat])
combined_names = list(feature_names) + stat_names

# Evaluate feature set
results = fe.evaluate_feature_set(X_combined, y, combined_names)
```

### Feature Selection
```python
# Univariate selection
X_selected, selector, scores, indices = fe.feature_selection_univariate(
    X_combined, y, k=10
)

# Importance-based selection
X_important, importances, imp_indices = fe.feature_selection_importance(
    X_combined, y, threshold=0.01
)

# Recursive Feature Elimination
X_rfe, rfe_selector, rankings, rfe_indices = fe.feature_selection_rfe(
    X_combined, y, n_features=8
)
```

### Dimensionality Reduction
```python
# Scale features first
X_scaled, scaler = fe.scale_features(X_combined, method='standard')

# PCA
X_pca, pca_model, explained_variance = fe.dimensionality_reduction_pca(
    X_scaled, explained_variance_ratio=0.95
)

# t-SNE for visualization
X_tsne, tsne_model = fe.dimensionality_reduction_tsne(X_scaled)
```

### Comprehensive Analysis
```python
# Run complete feature engineering pipeline
results = fe.comprehensive_feature_engineering('iris')
```

## Demonstration Functions

### `comprehensive_feature_engineering(dataset_name)`
Performs complete feature engineering analysis:
1. **Original Features**: Baseline performance evaluation
2. **Polynomial Features**: Create interaction and higher-order terms
3. **Statistical Features**: Generate statistical transformations
4. **Feature Selection**: Apply multiple selection methods
5. **Dimensionality Reduction**: PCA and t-SNE analysis
6. **Visualization**: Feature importance, correlations, dimensionality reduction plots
7. **Performance Comparison**: Comprehensive results summary

### `demonstrate_feature_engineering()`
Main demonstration function that:
- Runs comprehensive analysis on Iris dataset
- Performs quick analysis on Wine dataset
- Compares different feature engineering approaches

## Key Learning Outcomes

### Feature Engineering Concepts
1. **Feature Creation**: Understanding how to generate new meaningful features
2. **Feature Transformation**: Mathematical transformations and their effects
3. **Feature Selection**: Different approaches to identify relevant features
4. **Dimensionality Reduction**: Techniques to reduce feature space while preserving information
5. **Feature Scaling**: Importance of normalization for different algorithms

### Practical Skills
1. **Pipeline Design**: Building comprehensive feature engineering pipelines
2. **Performance Evaluation**: Comparing different feature sets systematically
3. **Visualization**: Creating informative plots for feature analysis
4. **Method Selection**: Choosing appropriate techniques for different scenarios

### Advanced Concepts
1. **Curse of Dimensionality**: Understanding when more features hurt performance
2. **Feature Interaction**: Capturing non-linear relationships
3. **Regularization**: How feature engineering affects regularization needs
4. **Interpretability**: Balancing performance with model interpretability

## Visualization Features

### Feature Importance Plots
- Bar charts showing feature importance scores
- Ranked visualization of most important features
- Comparison across different selection methods

### Correlation Analysis
- Heatmaps showing feature correlations
- Identification of redundant features
- Understanding feature relationships

### Dimensionality Reduction Plots
- 2D scatter plots of reduced features
- Comparison of original vs. reduced feature spaces
- Class separation visualization

### Performance Comparison
- Summary tables comparing different approaches
- Cross-validation results with confidence intervals
- Method ranking and selection guidance

## Parameters and Configuration

### Feature Creation Parameters
- **Polynomial degree**: Controls complexity of polynomial features
- **Interaction only**: Whether to include only interaction terms
- **Binning strategy**: 'uniform', 'quantile', or 'kmeans'
- **Number of bins**: Discretization granularity

### Feature Selection Parameters
- **k**: Number of features to select (univariate)
- **Score function**: f_classif, mutual_info_classif
- **Importance threshold**: Minimum importance for selection
- **Estimator**: Base model for RFE and importance

### Dimensionality Reduction Parameters
- **n_components**: Number of components to keep
- **Explained variance ratio**: Minimum variance to preserve (PCA)
- **Perplexity**: t-SNE neighborhood parameter

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Required Libraries
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical visualizations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **SciPy**: Statistical functions

## Installation and Running

1. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy
   ```

2. **Run the demonstration**:
   ```bash
   python feature_engineering.py
   ```

3. **Import for custom use**:
   ```python
   from feature_engineering import FeatureEngineering
   fe = FeatureEngineering()
   ```

## Expected Output

The demonstration will show:

1. **Dataset Information**: Original dataset characteristics
2. **Feature Engineering Steps**: Progress through different techniques
3. **Performance Metrics**: Cross-validation and test accuracies
4. **Visualizations**: Feature importance, correlations, dimensionality reduction
5. **Summary Table**: Comparison of all methods
6. **Best Method**: Recommendation based on performance

### Sample Output
```
======================================================================
COMPREHENSIVE FEATURE ENGINEERING - IRIS DATASET
======================================================================

Original Dataset:
- Samples: 150
- Features: 4
- Classes: 3

1. Evaluating Original Features...
Original feature performance:
  Logistic Regression: CV=0.9583±0.0372, Test=1.0000
  Random Forest: CV=0.9500±0.0500, Test=1.0000

2. Creating Polynomial Features...
Polynomial features: 15 features
  Logistic Regression: CV=0.9667±0.0471, Test=1.0000
  Random Forest: CV=0.9583±0.0372, Test=1.0000

...

7. Performance Summary:
================================================================================
   Method  Features              LR_CV  LR_Test              RF_CV  RF_Test
 Original         4  0.9583±0.0372   1.0000  0.9500±0.0500   1.0000
Polynomial        15  0.9667±0.0471   1.0000  0.9583±0.0372   1.0000
 Combined        25  0.9750±0.0395   1.0000  0.9667±0.0471   1.0000
...

Best performing method: Combined (Average test accuracy: 1.0000)
```

## Potential Extensions

### Advanced Feature Engineering
1. **Domain-specific features**: Custom features for specific domains
2. **Time-series features**: Lag features, rolling statistics, seasonality
3. **Text features**: TF-IDF, word embeddings, n-grams
4. **Image features**: HOG, SIFT, CNN features

### Advanced Selection Methods
1. **Wrapper methods**: Forward/backward selection
2. **Embedded methods**: LASSO, Ridge regularization
3. **Stability selection**: Robust feature selection
4. **Multi-objective optimization**: Balancing performance and complexity

### Advanced Dimensionality Reduction
1. **Non-linear methods**: Kernel PCA, Autoencoders
2. **Manifold learning**: Isomap, Locally Linear Embedding
3. **Supervised reduction**: Linear Discriminant Analysis
4. **Sparse methods**: Sparse PCA, Dictionary Learning

### Performance Optimization
1. **Feature caching**: Store computed features
2. **Parallel processing**: Multiprocessing for feature creation
3. **Memory optimization**: Efficient data structures
4. **Incremental learning**: Online feature engineering

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce polynomial degree or use feature selection
2. **Slow performance**: Use smaller datasets or reduce feature complexity
3. **Poor results**: Check feature scaling and selection parameters
4. **Visualization errors**: Ensure matplotlib backend is properly configured

### Performance Tips
1. **Start simple**: Begin with basic transformations
2. **Use cross-validation**: Avoid overfitting to specific train/test splits
3. **Monitor complexity**: Balance performance gains with interpretability
4. **Domain knowledge**: Incorporate domain expertise in feature design

## References

1. **Feature Engineering for Machine Learning** - Alice Zheng & Amanda Casari
2. **Hands-On Machine Learning** - Aurélien Géron
3. **The Elements of Statistical Learning** - Hastie, Tibshirani & Friedman
4. **Pattern Recognition and Machine Learning** - Christopher Bishop
5. **Scikit-learn Documentation**: Feature selection and preprocessing guides