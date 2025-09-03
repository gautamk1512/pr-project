# Practical 5: Normal Distribution Sample Generation

## Overview

This practical demonstrates comprehensive normal distribution sample generation and analysis. It covers univariate and multivariate normal distributions, statistical testing, parameter estimation, and the Central Limit Theorem with extensive visualization capabilities.

## Implementation Features

### NormalDistributionGenerator Class
- **Univariate Normal Generation**: Single-variable normal distributions
- **Multivariate Normal Generation**: Multi-dimensional normal distributions with covariance
- **Statistical Testing**: Multiple normality tests and hypothesis testing
- **Parameter Estimation**: Maximum likelihood estimation from samples
- **Visualization**: Comprehensive plotting for analysis
- **Central Limit Theorem**: Interactive demonstration
- **Distribution Comparison**: Side-by-side analysis tools

### Key Methods

#### Sample Generation
- `generate_univariate_normal()`: Generate 1D normal samples
- `generate_multivariate_normal()`: Generate multi-dimensional normal samples
- `estimate_parameters()`: Estimate distribution parameters from data

#### Statistical Testing
- `test_normality()`: Multiple normality tests (Shapiro-Wilk, D'Agostino-Pearson, KS)
- Hypothesis testing with configurable significance levels
- Comprehensive test result reporting

#### Visualization
- `plot_univariate_distribution()`: Histograms, Q-Q plots, box plots, statistics
- `plot_multivariate_distribution()`: Scatter plots, contours, marginals, correlations
- `compare_distributions()`: Side-by-side distribution comparison
- `demonstrate_central_limit_theorem()`: CLT visualization

#### Advanced Analysis
- `comprehensive_normal_analysis()`: Complete analysis pipeline
- Parameter estimation accuracy analysis
- High-dimensional distribution handling

## Mathematical Foundations

### Univariate Normal Distribution
The probability density function of a normal distribution:

```
f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```

Where:
- μ (mu): mean parameter
- σ (sigma): standard deviation parameter
- σ² (sigma squared): variance parameter

### Multivariate Normal Distribution
For a k-dimensional multivariate normal distribution:

```
f(x) = (1/√((2π)^k |Σ|)) * exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

Where:
- μ: k-dimensional mean vector
- Σ: k×k covariance matrix
- |Σ|: determinant of covariance matrix
- Σ⁻¹: inverse of covariance matrix

### Covariance and Correlation
- **Covariance**: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
- **Correlation**: ρ = Cov(X,Y)/(σₓσᵧ)
- **Covariance Matrix**: Σᵢⱼ = Cov(Xᵢ, Xⱼ)
- **Correlation Matrix**: Rᵢⱼ = Corr(Xᵢ, Xⱼ)

### Central Limit Theorem
For independent random variables X₁, X₂, ..., Xₙ with mean μ and variance σ²:

```
(X̄ - μ) / (σ/√n) → N(0,1) as n → ∞
```

Where X̄ = (X₁ + X₂ + ... + Xₙ)/n

### Parameter Estimation
- **Sample Mean**: x̄ = (1/n)Σxᵢ
- **Sample Variance**: s² = (1/(n-1))Σ(xᵢ - x̄)²
- **Sample Covariance**: S = (1/(n-1))Σ(xᵢ - x̄)(xᵢ - x̄)ᵀ

## Normality Tests

### Shapiro-Wilk Test
- **Best for**: Small to medium sample sizes (n ≤ 5000)
- **Null Hypothesis**: Data comes from normal distribution
- **Test Statistic**: W = (Σaᵢx₍ᵢ₎)² / Σ(xᵢ - x̄)²

### D'Agostino-Pearson Test
- **Tests**: Skewness and kurtosis jointly
- **Null Hypothesis**: Data has normal skewness and kurtosis
- **Test Statistic**: K² = Z₁² + Z₂² (chi-square with 2 df)

### Kolmogorov-Smirnov Test
- **Tests**: Goodness of fit to any distribution
- **Null Hypothesis**: Data follows specified distribution
- **Test Statistic**: D = max|F(x) - S(x)|

## Usage Examples

### Basic Univariate Generation
```python
from normal_distribution import NormalDistributionGenerator

# Initialize generator
gen = NormalDistributionGenerator(random_state=42)

# Generate standard normal samples
samples, params = gen.generate_univariate_normal(
    mean=0, std=1, n_samples=1000, name="standard_normal"
)

# Plot and analyze
gen.plot_univariate_distribution(samples, params, "Standard Normal")

# Test normality
normality_results = gen.test_normality(samples)
print(normality_results)
```

### Multivariate Normal Generation
```python
# Define parameters
mean = [2, 3]
covariance = [[2, 1.5], [1.5, 3]]  # Positive correlation

# Generate samples
samples, params = gen.generate_multivariate_normal(
    mean, covariance, n_samples=1000, name="bivariate_normal"
)

# Visualize
gen.plot_multivariate_distribution(samples, params, "Bivariate Normal")

# Estimate parameters
estimated = gen.estimate_parameters(samples)
print(f"Estimated mean: {estimated['estimated_mean']}")
print(f"Estimated covariance:\n{estimated['estimated_covariance']}")
```

### Central Limit Theorem Demonstration
```python
# Demonstrate CLT with different population distributions
sample_means_uniform = gen.demonstrate_central_limit_theorem(
    population_dist='uniform', 
    sample_sizes=[1, 5, 10, 30], 
    n_samples=1000
)

sample_means_exponential = gen.demonstrate_central_limit_theorem(
    population_dist='exponential', 
    sample_sizes=[1, 5, 10, 30], 
    n_samples=1000
)
```

### Distribution Comparison
```python
# Generate different distributions
samples1, _ = gen.generate_univariate_normal(0, 1, 1000, "standard")
samples2, _ = gen.generate_univariate_normal(5, 2, 1000, "shifted")
samples3, _ = gen.generate_univariate_normal(0, 0.5, 1000, "narrow")

# Compare them
gen.compare_distributions(
    [samples1, samples2, samples3],
    ["Standard (0,1)", "Shifted (5,2)", "Narrow (0,0.5)"],
    "Normal Distribution Comparison"
)
```

### Comprehensive Analysis
```python
# Run complete analysis pipeline
results = gen.comprehensive_normal_analysis()
```

## Demonstration Functions

### `comprehensive_normal_analysis()`
Performs complete normal distribution analysis:
1. **Univariate Analysis**: Different parameter combinations
2. **Bivariate Analysis**: Independent and correlated cases
3. **High-Dimensional Analysis**: 5D normal distribution with PCA
4. **Central Limit Theorem**: Multiple population distributions
5. **Parameter Estimation**: Accuracy vs sample size analysis

### `demonstrate_central_limit_theorem()`
Interactive CLT demonstration:
- Multiple population distributions (uniform, exponential, binomial)
- Various sample sizes showing convergence to normality
- Theoretical vs empirical comparison
- Normality testing of sample means

### `demonstrate_normal_distributions()`
Main demonstration function that runs the complete analysis pipeline

## Key Learning Outcomes

### Statistical Concepts
1. **Normal Distribution Properties**: Understanding mean, variance, and shape
2. **Multivariate Relationships**: Covariance, correlation, and dependence
3. **Central Limit Theorem**: Convergence to normality regardless of population
4. **Parameter Estimation**: Maximum likelihood and method of moments
5. **Hypothesis Testing**: Normality tests and their interpretation

### Practical Skills
1. **Sample Generation**: Creating realistic synthetic datasets
2. **Statistical Testing**: Applying and interpreting normality tests
3. **Visualization**: Creating informative statistical plots
4. **Parameter Validation**: Checking estimation accuracy
5. **Distribution Comparison**: Systematic comparison methods

### Advanced Topics
1. **Positive Definite Matrices**: Ensuring valid covariance matrices
2. **High-Dimensional Visualization**: PCA for dimensionality reduction
3. **Convergence Analysis**: Understanding asymptotic behavior
4. **Robustness**: Handling edge cases and numerical issues

## Visualization Features

### Univariate Plots
- **Histograms**: With fitted normal curves and true parameters
- **Q-Q Plots**: Quantile-quantile plots against normal distribution
- **Box Plots**: Showing quartiles, outliers, and distribution shape
- **Statistics Summary**: Comprehensive parameter comparison

### Multivariate Plots
- **Scatter Plots**: 2D and 3D visualizations
- **Contour Plots**: Probability density contours
- **Marginal Distributions**: Individual dimension analysis
- **Correlation Heatmaps**: Correlation matrix visualization
- **PCA Projections**: High-dimensional data visualization

### Comparison Plots
- **Overlaid Histograms**: Multiple distributions on same axes
- **Box Plot Arrays**: Side-by-side distribution comparison
- **Statistics Tables**: Tabular parameter comparison
- **Q-Q Plot Arrays**: Multiple normality assessments

### Central Limit Theorem Plots
- **Convergence Visualization**: Sample size vs normality
- **Theoretical Overlay**: Expected vs observed distributions
- **Population vs Sample Means**: Distribution transformation
- **Normality Testing**: Statistical test results

## Parameters and Configuration

### Generation Parameters
- **mean**: Distribution center (scalar or vector)
- **std/covariance**: Distribution spread (scalar, matrix)
- **n_samples**: Number of samples to generate
- **random_state**: Reproducibility seed

### Testing Parameters
- **alpha**: Significance level for hypothesis tests (default: 0.05)
- **test_selection**: Which normality tests to apply

### Visualization Parameters
- **bins**: Histogram bin count
- **alpha**: Transparency for overlays
- **colors**: Color schemes for multiple distributions
- **figure_size**: Plot dimensions

## Dependencies

```bash
pip install numpy matplotlib seaborn scipy scikit-learn pandas
```

### Required Libraries
- **NumPy**: Numerical computations and random number generation
- **Matplotlib**: Basic plotting and 3D visualization
- **Seaborn**: Statistical plotting enhancements
- **SciPy**: Statistical functions and hypothesis tests
- **Scikit-learn**: PCA and preprocessing utilities
- **Pandas**: Data manipulation and tabular display

## Installation and Running

1. **Install dependencies**:
   ```bash
   pip install numpy matplotlib seaborn scipy scikit-learn pandas
   ```

2. **Run the demonstration**:
   ```bash
   python normal_distribution.py
   ```

3. **Import for custom use**:
   ```python
   from normal_distribution import NormalDistributionGenerator
   gen = NormalDistributionGenerator()
   ```

## Expected Output

The demonstration will show:

1. **Univariate Analysis**: Standard, shifted, and narrow normal distributions
2. **Multivariate Analysis**: Independent, positively correlated, and negatively correlated bivariate normals
3. **High-Dimensional Analysis**: 5D normal distribution with PCA visualization
4. **Central Limit Theorem**: Convergence demonstrations for uniform and exponential populations
5. **Parameter Estimation**: Accuracy analysis across different sample sizes

### Sample Output
```
======================================================================
COMPREHENSIVE NORMAL DISTRIBUTION ANALYSIS
======================================================================

1. Univariate Normal Distribution Analysis
--------------------------------------------------

Normality test results for Standard Normal:
  Shapiro-Wilk: p-value = 0.234567, Normal = True
  D'Agostino-Pearson: p-value = 0.456789, Normal = True
  Kolmogorov-Smirnov: p-value = 0.123456, Normal = True

2. Bivariate Normal Distribution Analysis
--------------------------------------------------

3. High-Dimensional Normal Distribution Analysis
--------------------------------------------------

4. Central Limit Theorem Demonstration
--------------------------------------------------
======================================================================
CENTRAL LIMIT THEOREM DEMONSTRATION
Population Distribution: UNIFORM
======================================================================

Sample Size n=1:
  Theoretical: μ=5.0000, σ=2.8868
  Empirical:   μ=5.0123, σ=2.8901
  Difference:  μ=0.0123, σ=0.0033

Sample Size n=5:
  Theoretical: μ=5.0000, σ=1.2910
  Empirical:   μ=4.9987, σ=1.2934
  Difference:  μ=0.0013, σ=0.0024

...

5. Parameter Estimation Analysis
--------------------------------------------------
n=  10: Mean=10.1234 (error=0.1234), Std=2.8901 (error=0.1099)
n=  50: Mean=9.9876 (error=0.0124), Std=3.0234 (error=0.0234)
n= 100: Mean=10.0045 (error=0.0045), Std=2.9876 (error=0.0124)
...

======================================================================
NORMAL DISTRIBUTION ANALYSIS COMPLETE
======================================================================
```

## Advanced Features

### Covariance Matrix Validation
- Automatic positive definite checking
- Regularization for numerical stability
- Eigenvalue decomposition for matrix correction

### High-Dimensional Handling
- PCA for visualization of high-dimensional data
- Efficient computation for large covariance matrices
- Memory-optimized sample generation

### Statistical Robustness
- Multiple normality tests for comprehensive assessment
- Confidence intervals for parameter estimates
- Bootstrap methods for uncertainty quantification

### Visualization Optimization
- Adaptive binning for histograms
- Automatic axis scaling
- Color-blind friendly palettes
- Interactive plot elements

## Potential Extensions

### Advanced Distributions
1. **Truncated Normal**: Normal distributions with bounds
2. **Mixture of Normals**: Multiple normal components
3. **Skew Normal**: Asymmetric normal distributions
4. **Student's t**: Heavy-tailed alternatives

### Advanced Testing
1. **Multivariate Normality**: Mardia's test, Henze-Zirkler test
2. **Goodness of Fit**: Anderson-Darling, Cramér-von Mises
3. **Outlier Detection**: Mahalanobis distance, isolation forests
4. **Change Point Detection**: Structural breaks in normality

### Machine Learning Integration
1. **Generative Models**: VAEs, GANs for complex distributions
2. **Density Estimation**: Kernel density estimation, normalizing flows
3. **Anomaly Detection**: One-class SVM, isolation forests
4. **Dimensionality Reduction**: t-SNE, UMAP for visualization

### Performance Optimization
1. **Parallel Generation**: Multiprocessing for large samples
2. **Memory Efficiency**: Streaming generation for huge datasets
3. **GPU Acceleration**: CUDA-based generation
4. **Approximate Methods**: Fast approximate normality tests

## Troubleshooting

### Common Issues
1. **Singular Covariance Matrix**: Use regularization or check for linear dependencies
2. **Memory Errors**: Reduce sample size or use streaming generation
3. **Slow Performance**: Use smaller sample sizes for interactive exploration
4. **Visualization Errors**: Check matplotlib backend configuration

### Performance Tips
1. **Sample Size Selection**: Balance accuracy with computational cost
2. **Covariance Design**: Ensure positive definiteness for stability
3. **Test Selection**: Use appropriate tests for sample size
4. **Visualization Limits**: Limit high-dimensional plots to essential views

### Numerical Stability
1. **Matrix Conditioning**: Check condition numbers for covariance matrices
2. **Regularization**: Add small values to diagonal for stability
3. **Precision**: Use double precision for critical calculations
4. **Validation**: Always validate generated samples

## References

1. **Probability and Statistics** - DeGroot & Schervish
2. **The Elements of Statistical Learning** - Hastie, Tibshirani & Friedman
3. **Multivariate Statistical Analysis** - Johnson & Wichern
4. **Statistical Inference** - Casella & Berger
5. **SciPy Documentation**: Statistical functions and distributions
6. **NumPy Documentation**: Random number generation