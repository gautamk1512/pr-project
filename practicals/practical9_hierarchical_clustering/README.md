# Practical 9: Hierarchical Clustering

A comprehensive implementation of hierarchical clustering algorithms with advanced visualization and analysis capabilities.

## Overview

Hierarchical clustering is a method of cluster analysis that builds a hierarchy of clusters. This implementation provides both agglomerative (bottom-up) and divisive (top-down) approaches with extensive visualization tools.

## Features

### Core Algorithm
- **Agglomerative Clustering**: Bottom-up approach starting with individual points
- **Multiple Linkage Methods**: Ward, Complete, Average, and Single linkage
- **Distance Metrics**: Euclidean, Manhattan, and Cosine distance
- **Automatic Cluster Assignment**: Based on specified number of clusters

### Advanced Analysis Tools
- **Dendrogram Visualization**: Interactive tree-like cluster hierarchy display
- **Linkage Method Comparison**: Side-by-side comparison of different linkage criteria
- **Distance Matrix Heatmap**: Visual representation of point-to-point distances
- **Optimal Cluster Selection**: Silhouette analysis and elbow method

### Enhanced Visualizations
- **Professional Style Plots**: Publication-ready cluster visualizations
- **Multiple Style Options**: Default, professional, and seaborn styling
- **Comprehensive Comparison**: Multi-panel analysis layouts
- **High-Quality Output**: 300 DPI resolution for all saved plots

### Quality Evaluation
- **Silhouette Score**: Measure of cluster cohesion and separation
- **Inertia Calculation**: Within-cluster sum of squares
- **Adjusted Rand Index**: External validation when true labels available
- **Comprehensive Metrics**: Multiple evaluation criteria

## Installation

### Required Dependencies
```bash
pip install numpy matplotlib seaborn scipy scikit-learn
```

### Optional Dependencies
```bash
pip install pandas jupyter  # For data manipulation and notebooks
```

## Usage

### Basic Usage
```python
from hierarchical_clustering import HierarchicalClustering
import numpy as np
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Initialize and fit hierarchical clustering
hc = HierarchicalClustering(linkage_method='ward')
hc.fit(X, n_clusters=4)

# Plot dendrogram
hc.plot_dendrogram(title='Sample Data Dendrogram')

# Plot clusters
hc.plot_clusters(style='professional')

# Evaluate clustering quality
metrics = hc.evaluate_clustering(true_labels=y)
print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
```

### Advanced Analysis
```python
# Compare different linkage methods
hc.compare_linkage_methods(X, n_clusters=4)

# Plot distance matrix
hc.plot_distance_matrix()

# Comprehensive evaluation
metrics = hc.evaluate_clustering(true_labels=y)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

### Complete Demonstration
```python
# Run the complete demonstration
from hierarchical_clustering import demonstrate_hierarchical_clustering
demonstrate_hierarchical_clustering()
```

## Output Files

The demonstration generates several visualization files:

### Core Visualizations
- **`hierarchical_dendrogram_*.png`** - Tree-like hierarchy visualization
- **`hierarchical_clusters_*.png`** - Final cluster assignments
- **`hierarchical_linkage_comparison_*.png`** - Comparison of linkage methods
- **`hierarchical_distance_matrix_*.png`** - Distance matrix heatmaps
- **`hierarchical_optimal_clusters.png`** - Optimal cluster selection analysis

### Analysis Features
- High-resolution (300 DPI) publication-quality images
- Professional styling with clear legends and labels
- Comprehensive metric displays
- Educational step-by-step visualizations

## Mathematical Foundation

### Linkage Methods

1. **Ward Linkage**: Minimizes within-cluster variance
   ```
   d(u,v) = √(|v|+|s|)/(T) * d(v,s)² + (|v|+|t|)/(T) * d(v,t)² - |v|/T * d(s,t)²
   ```

2. **Complete Linkage**: Maximum distance between clusters
   ```
   d(u,v) = max(dist(u[i], v[j])) for all i,j
   ```

3. **Average Linkage**: Average distance between all pairs
   ```
   d(u,v) = Σᵢⱼ dist(u[i], v[j]) / (|u| × |v|)
   ```

4. **Single Linkage**: Minimum distance between clusters
   ```
   d(u,v) = min(dist(u[i], v[j])) for all i,j
   ```

### Quality Metrics

**Silhouette Score**: Measures cluster cohesion and separation
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Where:
- a(i) = average distance to points in same cluster
- b(i) = average distance to points in nearest cluster

**Adjusted Rand Index**: External validation metric
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

## Algorithm Comparison

| Method | Time Complexity | Space Complexity | Best For |
|--------|----------------|------------------|----------|
| Ward | O(n³) | O(n²) | Spherical clusters |
| Complete | O(n³) | O(n²) | Compact clusters |
| Average | O(n³) | O(n²) | Balanced approach |
| Single | O(n³) | O(n²) | Chain-like clusters |

## Enhanced Functions Reference

### Core Methods
- `HierarchicalClustering()`: Main clustering class
- `fit(X, n_clusters)`: Fit clustering model to data
- `plot_dendrogram()`: Visualize cluster hierarchy
- `plot_clusters()`: Display final cluster assignments

### Advanced Analysis
- `compare_linkage_methods()`: Compare different linkage criteria
- `plot_distance_matrix()`: Visualize distance relationships
- `evaluate_clustering()`: Comprehensive quality assessment

### Utility Functions
- `demonstrate_hierarchical_clustering()`: Complete demonstration
- Multiple styling options for all visualizations
- Automatic optimal cluster detection

## Educational Applications

1. **Understanding Cluster Hierarchies**: Dendrogram interpretation
2. **Linkage Method Selection**: Comparing different approaches
3. **Distance Metric Impact**: Visualizing metric effects
4. **Cluster Validation**: Quality assessment techniques
5. **Algorithm Comparison**: Hierarchical vs. partitional methods

## Research Applications

- **Phylogenetic Analysis**: Evolutionary relationships
- **Market Segmentation**: Customer grouping
- **Gene Expression**: Biological data clustering
- **Social Network Analysis**: Community detection
- **Image Segmentation**: Computer vision applications

## Troubleshooting

### Common Issues
1. **Memory Error**: Reduce dataset size for large data
2. **Slow Performance**: Use approximate methods for n > 1000
3. **Poor Clustering**: Try different linkage methods
4. **Visualization Issues**: Adjust figure size parameters

### Performance Tips
- Standardize features before clustering
- Use Ward linkage for spherical clusters
- Consider dimensionality reduction for high-dimensional data
- Validate results with multiple metrics

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Kaufman, L., & Rousseeuw, P. J. (2009). Finding Groups in Data
- Scikit-learn Documentation: Hierarchical Clustering
- SciPy Documentation: Cluster Analysis

---

**Note**: This implementation is designed for educational and research purposes. For production use with large datasets, consider using optimized libraries like scikit-learn's AgglomerativeClustering with appropriate parameters.