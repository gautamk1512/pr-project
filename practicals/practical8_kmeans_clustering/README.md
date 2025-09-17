# Practical 8: K-Means Clustering Implementation

This practical implements K-Means clustering algorithm from scratch with comprehensive functionality for data generation, clustering, visualization, and analysis.

## Overview

K-Means clustering is an unsupervised machine learning algorithm that partitions data into k clusters based on feature similarity. This implementation provides:

- Complete K-Means algorithm from scratch
- Data generation utilities
- Visualization functions
- Elbow method for optimal k selection
- Performance metrics calculation

## Features

### Core Algorithm
- **Complete K-Means Implementation**: From scratch implementation without using sklearn's KMeans
- **Flexible Initialization**: Random centroid initialization with configurable random state
- **Convergence Detection**: Automatic stopping when centroids stabilize
- **Robust Prediction**: Classify new data points using trained model

### Advanced Analysis Tools
- **Elbow Method with Silhouette Analysis**: Comprehensive optimal k determination
- **Convergence History Tracking**: Visualize algorithm convergence behavior
- **Quality Metrics Evaluation**: Multiple clustering quality assessments
- **Comparative Analysis**: Side-by-side comparison of different k values

### Enhanced Visualizations
- **Professional Style Plots**: Multiple visualization styles (default, professional, seaborn)
- **Comprehensive Comparison Views**: Original data vs K-means vs true clusters
- **Convergence Analysis Plots**: Inertia convergence and centroid movement tracking
- **Multi-K Comparison**: Visual comparison of clustering with different k values
- **High-Quality Output**: 300 DPI publication-ready plots

### Utility Functions
- **Sample Data Generation**: Built-in function to create test datasets with known clusters
- **Automatic Elbow Point Detection**: Mathematical elbow point identification
- **Flexible Plotting Options**: Customizable titles, styles, and save paths

## Usage

### Basic Usage

```python
from kmeans_clustering import KMeansClustering
import numpy as np

# Create K-Means instance
kmeans = KMeansClustering(k=3, random_state=42)

# Generate sample data
X, y_true = kmeans.generate_sample_data(n_samples=300, centers=3)

# Fit the model
kmeans.fit(X)

# Enhanced visualizations with different styles
kmeans.plot_clusters(X, title="Professional Style", style='professional')
kmeans.plot_clusters(X, title="Seaborn Style", style='seaborn')

# Comprehensive comparison plot
kmeans.plot_cluster_comparison(X, y_true, save_path='comparison.png')

# Analyze convergence behavior
kmeans.plot_convergence_history(X, save_path='convergence.png')

# Enhanced elbow method with silhouette analysis
elbow_results = kmeans.elbow_method(X, k_range=range(1, 10), 
                                   show_silhouette=True)
print(f"Optimal k (Elbow): {elbow_results['optimal_k_elbow']}")
print(f"Optimal k (Silhouette): {elbow_results['optimal_k_silhouette']}")

# Evaluate clustering quality
quality_metrics = kmeans.evaluate_clustering_quality(X, y_true)
print(f"Silhouette Score: {quality_metrics['silhouette_score']:.3f}")
print(f"Adjusted Rand Score: {quality_metrics['adjusted_rand_score']:.3f}")

# Predict new points
new_points = np.array([[0, 0], [3, 3], [-2, 2]])
predictions = kmeans.predict(new_points)
print(f"Predictions: {predictions}")
```

### Visualization

```python
# Plot clustering results
kmeans.plot_clusters(X, title="My Clustering Results", save_path="results.png")

# Perform elbow method analysis
inertias = kmeans.elbow_method(X, k_range=range(1, 10), save_path="elbow.png")
```

### Complete Demonstration

```python
# Run the complete demonstration
python kmeans_clustering.py
```

## Algorithm Steps

1. **Initialization**: 
   - Choose number of clusters (k)
   - Initialize k centroids randomly within data range

2. **Assignment Step**:
   - Calculate distance from each point to all centroids
   - Assign each point to nearest centroid

3. **Update Step**:
   - Recalculate centroids as mean of assigned points
   - Handle empty clusters by keeping previous centroid

4. **Convergence Check**:
   - Compare new centroids with previous ones
   - Stop if centroids don't change significantly
   - Continue until convergence or max iterations

## Key Parameters

- **k**: Number of clusters to form
- **max_iterations**: Maximum number of algorithm iterations
- **random_state**: Seed for reproducible results
- **n_samples**: Number of data points to generate
- **centers**: Number of true clusters in synthetic data
- **cluster_std**: Standard deviation of generated clusters

## Output Files

When you run the enhanced demonstration, the following files will be generated:

### Core Visualizations
- `enhanced_kmeans_comparison.png`: Comprehensive three-panel comparison (original data, K-means results, true clusters)
- `convergence_history.png`: Dual-panel convergence analysis (inertia vs iterations, centroid movement paths)

### Style Variations
- `professional_clusters.png`: Professional style clustering visualization
- `seaborn_clusters.png`: Seaborn style clustering visualization

### Analysis Plots
- `enhanced_elbow_analysis.png`: Comprehensive elbow method with silhouette analysis (dual-panel layout)
- `different_k_comparison.png`: Grid comparison of clustering results with k=2 through k=7

### Legacy Files (for compatibility)
- `kmeans_comparison.png`: Original three-panel comparison
- `elbow_method.png`: Basic elbow method plot

All plots are saved in high resolution (300 DPI) for publication quality with professional styling and clear legends.

## Applications

This K-Means implementation can be used for:

- **Customer Segmentation**: Group customers by behavior patterns
- **Image Compression**: Reduce color palette by clustering pixels
- **Market Research**: Identify distinct consumer groups
- **Data Preprocessing**: Reduce dataset complexity
- **Anomaly Detection**: Identify outliers not belonging to any cluster

## Enhanced Functions Reference

### Core Methods
- `fit(X)`: Train the K-means model on data X
- `predict(X)`: Predict cluster assignments for new data
- `calculate_inertia(X)`: Calculate within-cluster sum of squares

### Advanced Analysis
- `elbow_method(X, k_range, show_silhouette=True)`: Enhanced elbow analysis with silhouette scores
- `evaluate_clustering_quality(X, y_true=None)`: Comprehensive quality metrics evaluation
- `_find_elbow_point(k_values, inertias)`: Mathematical elbow point detection

### Enhanced Visualizations
- `plot_clusters(X, style='default', title=None, save_path=None)`: Multi-style cluster plotting
- `plot_cluster_comparison(X, y_true=None, save_path=None)`: Three-panel comparison view
- `plot_convergence_history(X, save_path=None)`: Convergence analysis with centroid tracking

### Utility Functions
- `generate_sample_data(n_samples, centers, cluster_std=1.0)`: Generate test datasets
- `initialize_centroids(X)`: Random centroid initialization
- `assign_clusters(X, centroids)`: Assign points to nearest centroids
- `update_centroids(X, assignments)`: Update centroids based on assignments

## Mathematical Foundation

The K-Means algorithm minimizes the within-cluster sum of squares (WCSS):

```
WCSS = Σ(i=1 to k) Σ(x in Ci) ||x - μi||²
```

Where:
- `k` is the number of clusters
- `Ci` is the i-th cluster
- `μi` is the centroid of cluster i
- `||x - μi||²` is the squared Euclidean distance

### Quality Metrics

**Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Adjusted Rand Index**: Measures similarity between true and predicted clusters, adjusted for chance
```
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
```

### Euclidean Distance
```
distance(p, c) = √(Σ(pi - ci)²)
```

### Centroid Update
```
centroid = (1/n) * Σ(points_in_cluster)
```

### Inertia (Within-cluster sum of squares)
```
WCSS = Σ(Σ(||xi - ck||²))
```

## Requirements

- numpy
- matplotlib
- scikit-learn (for data generation)

## Installation

```bash
pip install numpy matplotlib scikit-learn
```

## Example Output

```
=== K-Means Clustering Demonstration ===

1. Generating sample data...
Generated 300 data points with 2 features

2. Fitting K-Means clustering...
Converged after 8 iterations

3. Clustering Results:
   - Final inertia: 245.67
   - Number of iterations: Converged

4. Performing elbow method analysis...
Elbow plot saved to elbow_method.png

Inertia values for different k:
   k=1: 1456.23
   k=2: 567.89
   k=3: 245.67
   k=4: 198.45
   ...

5. Testing prediction on new data...
New points and their predicted clusters:
   Point [0 0]: Cluster 2
   Point [3 3]: Cluster 1
   Point [-2 2]: Cluster 3

=== Demonstration Complete ===
```

## Notes

- The algorithm uses Euclidean distance for similarity measurement
- Centroids are initialized randomly within the data range
- Empty clusters retain their previous centroid position
- The elbow method helps determine optimal number of clusters
- All plots are saved to files for easy viewing and sharing