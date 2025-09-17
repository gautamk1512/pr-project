# -*- coding: utf-8 -*-
"""
Practical 9: Hierarchical Clustering Implementation
Author:  gautam   singh
Date: 2024

This module implements hierarchical clustering algorithms including:
- Agglomerative Clustering (Bottom-up approach)
- Divisive Clustering (Top-down approach)
- Dendrogram visualization
- Cluster comparison and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class HierarchicalClustering:
    """
    A comprehensive implementation of Hierarchical Clustering algorithms.
    
    This class provides both agglomerative and divisive clustering methods
    with extensive visualization and analysis capabilities.
    """
    
    def __init__(self, linkage_method='ward', distance_metric='euclidean'):
        """
        Initialize the Hierarchical Clustering object.
        
        Parameters:
        -----------
        linkage_method : str, default='ward'
            The linkage criterion to use ('ward', 'complete', 'average', 'single')
        distance_metric : str, default='euclidean'
            The distance metric to use ('euclidean', 'manhattan', 'cosine')
        """
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.linkage_matrix = None
        self.labels_ = None
        self.n_clusters = None
        self.X = None
        
    def fit(self, X, n_clusters=3):
        """
        Fit the hierarchical clustering model to the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data
        n_clusters : int, default=3
            Number of clusters to form
        """
        self.X = X
        self.n_clusters = n_clusters
        
        # Compute linkage matrix
        self.linkage_matrix = linkage(X, method=self.linkage_method, metric=self.distance_metric)
        
        # Get cluster labels
        self.labels_ = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        return self
    
    def plot_dendrogram(self, figsize=(12, 8), title=None, save_path=None):
        """
        Plot the dendrogram for hierarchical clustering.
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        if self.linkage_matrix is None:
            raise ValueError("Model must be fitted before plotting dendrogram")
        
        plt.figure(figsize=figsize)
        
        # Create dendrogram
        dendrogram(
            self.linkage_matrix,
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=0.7*max(self.linkage_matrix[:,2])
        )
        
        plt.title(title or f'Hierarchical Clustering Dendrogram ({self.linkage_method.title()} Linkage)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clusters(self, style='default', figsize=(10, 8), title=None, save_path=None):
        """
        Plot the clustered data points.
        
        Parameters:
        -----------
        style : str, default='default'
            Plot style ('default', 'professional', 'seaborn')
        figsize : tuple, default=(10, 8)
            Figure size
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        if self.X is None or self.labels_ is None:
            raise ValueError("Model must be fitted before plotting clusters")
        
        # Set style
        if style == 'seaborn':
            sns.set_style("whitegrid")
            colors = sns.color_palette("husl", self.n_clusters)
        elif style == 'professional':
            plt.style.use('seaborn-v0_8-whitegrid')
            colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters))
        
        plt.figure(figsize=figsize)
        
        # Plot clusters
        for i in range(self.n_clusters):
            mask = self.labels_ == i
            plt.scatter(self.X[mask, 0], self.X[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i+1}', 
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        plt.title(title or f'Hierarchical Clustering Results (k={self.n_clusters})', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_linkage_methods(self, X, n_clusters=3, figsize=(15, 10), save_path=None):
        """
        Compare different linkage methods on the same dataset.
        
        Parameters:
        -----------
        X : array-like
            Input data
        n_clusters : int, default=3
            Number of clusters
        figsize : tuple, default=(15, 10)
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        linkage_methods = ['ward', 'complete', 'average', 'single']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for idx, method in enumerate(linkage_methods):
            # Fit clustering
            hc = HierarchicalClustering(linkage_method=method)
            hc.fit(X, n_clusters)
            
            # Plot
            ax = axes[idx]
            colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
            
            for i in range(n_clusters):
                mask = hc.labels_ == i
                ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                          label=f'Cluster {i+1}', alpha=0.7, s=50)
            
            # Calculate silhouette score
            sil_score = silhouette_score(X, hc.labels_)
            
            ax.set_title(f'{method.title()} Linkage\nSilhouette Score: {sil_score:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distance_matrix(self, figsize=(10, 8), save_path=None):
        """
        Plot the distance matrix heatmap.
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 8)
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if self.X is None:
            raise ValueError("Model must be fitted before plotting distance matrix")
        
        # Compute distance matrix
        distances = pdist(self.X, metric=self.distance_metric)
        distance_matrix = squareform(distances)
        
        plt.figure(figsize=figsize)
        sns.heatmap(distance_matrix, cmap='viridis', square=True, 
                   cbar_kws={'label': 'Distance'})
        plt.title(f'Distance Matrix Heatmap ({self.distance_metric.title()} Distance)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Sample Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_clustering(self, true_labels=None):
        """
        Evaluate clustering quality using various metrics.
        
        Parameters:
        -----------
        true_labels : array-like, optional
            True cluster labels for external validation
        
        Returns:
        --------
        dict : Dictionary containing evaluation metrics
        """
        if self.X is None or self.labels_ is None:
            raise ValueError("Model must be fitted before evaluation")
        
        metrics = {}
        
        # Silhouette Score
        metrics['silhouette_score'] = silhouette_score(self.X, self.labels_)
        
        # Inertia (within-cluster sum of squares)
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = self.X[self.labels_ == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)
        metrics['inertia'] = inertia
        
        # External validation (if true labels provided)
        if true_labels is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, self.labels_)
        
        return metrics

def demonstrate_hierarchical_clustering():
    """
    Demonstrate hierarchical clustering with comprehensive analysis.
    """
    print("=" * 60)
    print("HIERARCHICAL CLUSTERING DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample datasets...")
    
    # Dataset 1: Blobs
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=1.0, 
                                  random_state=42)
    
    # Dataset 2: Circles
    X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.3, 
                                       random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_blobs_scaled = scaler.fit_transform(X_blobs)
    X_circles_scaled = scaler.fit_transform(X_circles)
    
    datasets = [
        (X_blobs_scaled, y_blobs, "Blob Dataset"),
        (X_circles_scaled, y_circles, "Circles Dataset")
    ]
    
    for X, y_true, dataset_name in datasets:
        print(f"\n2. Analyzing {dataset_name}...")
        
        # Initialize hierarchical clustering
        hc = HierarchicalClustering(linkage_method='ward')
        
        # Fit the model
        n_clusters = len(np.unique(y_true))
        hc.fit(X, n_clusters=n_clusters)
        
        print(f"   - Fitted hierarchical clustering with {n_clusters} clusters")
        
        # Plot dendrogram
        hc.plot_dendrogram(
            title=f'{dataset_name} - Dendrogram',
            save_path=f'hierarchical_dendrogram_{dataset_name.lower().replace(" ", "_")}.png'
        )
        
        # Plot clusters
        hc.plot_clusters(
            style='professional',
            title=f'{dataset_name} - Hierarchical Clustering',
            save_path=f'hierarchical_clusters_{dataset_name.lower().replace(" ", "_")}.png'
        )
        
        # Compare linkage methods
        hc.compare_linkage_methods(
            X, n_clusters=n_clusters,
            save_path=f'hierarchical_linkage_comparison_{dataset_name.lower().replace(" ", "_")}.png'
        )
        
        # Plot distance matrix
        hc.plot_distance_matrix(
            save_path=f'hierarchical_distance_matrix_{dataset_name.lower().replace(" ", "_")}.png'
        )
        
        # Evaluate clustering
        metrics = hc.evaluate_clustering(true_labels=y_true)
        print(f"   - Clustering Evaluation:")
        print(f"     * Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"     * Inertia: {metrics['inertia']:.4f}")
        print(f"     * Adjusted Rand Score: {metrics['adjusted_rand_score']:.4f}")
    
    # Demonstrate optimal cluster selection
    print("\n3. Optimal Cluster Selection Analysis...")
    
    X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=1.5, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    k_range = range(2, 11)
    silhouette_scores = []
    inertias = []
    
    for k in k_range:
        hc = HierarchicalClustering(linkage_method='ward')
        hc.fit(X, n_clusters=k)
        
        metrics = hc.evaluate_clustering()
        silhouette_scores.append(metrics['silhouette_score'])
        inertias.append(metrics['inertia'])
    
    # Plot optimal cluster analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette scores
    ax1.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Analysis for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Inertia (elbow method)
    ax2.plot(k_range, inertias, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Inertia')
    ax2.set_title('Elbow Method for Optimal k')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hierarchical_optimal_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal k
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    print(f"   - Optimal k (Silhouette): {optimal_k_silhouette}")
    print(f"   - Best Silhouette Score: {max(silhouette_scores):.4f}")
    
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING DEMONSTRATION COMPLETE")
    print("Generated files:")
    print("- hierarchical_dendrogram_*.png")
    print("- hierarchical_clusters_*.png")
    print("- hierarchical_linkage_comparison_*.png")
    print("- hierarchical_distance_matrix_*.png")
    print("- hierarchical_optimal_clusters.png")
    print("="*60)

if __name__ == "__main__":
    demonstrate_hierarchical_clustering()