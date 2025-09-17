import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random
from typing import List, Tuple, Dict
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

class KMeansClustering:
    """
    K-Means Clustering implementation from scratch
    
    This class implements the K-Means clustering algorithm with various utility functions
    for data generation, visualization, and analysis.
    """
    
    def __init__(self, k: int = 3, max_iterations: int = 100, random_state: int = 42):
        """
        Initialize K-Means clustering
        
        Args:
            k: Number of clusters
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.clusters = None
        
    def generate_sample_data(self, n_samples: int = 500, n_features: int = 2, 
                           centers: int = 3, cluster_std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sample data using make_blobs
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            centers: Number of centers (true clusters)
            cluster_std: Standard deviation of clusters
            
        Returns:
            Tuple of (X, y) where X is features and y is true labels
        """
        X, y = make_blobs(n_samples=n_samples, 
                         n_features=n_features, 
                         centers=centers,
                         cluster_std=cluster_std,
                         random_state=self.random_state)
        return X, y
    
    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids randomly
        
        Args:
            X: Input data
            
        Returns:
            Initial centroids
        """
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        
        # Initialize centroids within the range of the data
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        
        centroids = np.random.uniform(min_vals, max_vals, (self.k, n_features))
        return centroids
    
    def calculate_distance(self, point: np.ndarray, centroid: np.ndarray) -> float:
        """
        Calculate Euclidean distance between a point and centroid
        
        Args:
            point: Data point
            centroid: Cluster centroid
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(np.sum((point - centroid) ** 2))
    
    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid
        
        Args:
            X: Input data
            centroids: Current centroids
            
        Returns:
            Cluster assignments for each point
        """
        assignments = np.zeros(X.shape[0])
        
        for i, point in enumerate(X):
            distances = [self.calculate_distance(point, centroid) for centroid in centroids]
            assignments[i] = np.argmin(distances)
            
        return assignments
    
    def update_centroids(self, X: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """
        Update centroids based on current cluster assignments
        
        Args:
            X: Input data
            assignments: Current cluster assignments
            
        Returns:
            Updated centroids
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            cluster_points = X[assignments == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If no points assigned to cluster, keep the old centroid
                new_centroids[i] = self.centroids[i] if self.centroids is not None else np.random.rand(X.shape[1])
                
        return new_centroids
    
    def fit(self, X: np.ndarray) -> 'KMeansClustering':
        """
        Fit K-Means clustering to the data
        
        Args:
            X: Input data
            
        Returns:
            Self for method chaining
        """
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            assignments = self.assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self.update_centroids(X, assignments)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids = new_centroids
        
        # Store final assignments
        self.clusters = assignments
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data
        
        Args:
            X: New data points
            
        Returns:
            Cluster assignments
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.assign_clusters(X, self.centroids)
    
    def calculate_inertia(self, X: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squares (inertia)
        
        Args:
            X: Input data
            
        Returns:
            Inertia value
        """
        if self.centroids is None or self.clusters is None:
            raise ValueError("Model must be fitted before calculating inertia")
            
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.clusters == i]
            if len(cluster_points) > 0:
                centroid = self.centroids[i]
                inertia += np.sum((cluster_points - centroid) ** 2)
                
        return inertia
    
    def plot_clusters(self, X: np.ndarray, title: str = "K-Means Clustering Results", 
                     save_path: str = None, style: str = 'default') -> None:
        """
        Plot the clustering results with enhanced visualization options
        
        Args:
            X: Input data
            title: Plot title
            save_path: Path to save the plot (optional)
            style: Plot style ('default', 'seaborn', 'professional')
        """
        if self.clusters is None:
            raise ValueError("Model must be fitted before plotting")
            
        # Set style
        if style == 'seaborn':
            sns.set_style("whitegrid")
            plt.figure(figsize=(12, 8))
        elif style == 'professional':
            plt.style.use('seaborn-v0_8-darkgrid')
            plt.figure(figsize=(14, 10))
        else:
            plt.figure(figsize=(10, 8))
        
        # Enhanced color palette
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        # Plot data points with enhanced styling
        for i in range(self.k):
            cluster_points = X[self.clusters == i]
            if len(cluster_points) > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=colors[i % len(colors)], label=f'Cluster {i+1}', 
                          alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
        
        # Plot centroids with enhanced styling
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='black', marker='X', s=300, linewidths=2, 
                   label='Centroids', edgecolors='white')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Feature 1', fontsize=12, fontweight='bold')
        plt.ylabel('Feature 2', fontsize=12, fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def elbow_method(self, X: np.ndarray, k_range: range = range(1, 11), 
                    save_path: str = None, show_silhouette: bool = True) -> Dict[str, List[float]]:
        """
        Perform comprehensive elbow method analysis with silhouette scores
        
        Args:
            X: Input data
            k_range: Range of k values to test
            save_path: Path to save the plot (optional)
            show_silhouette: Whether to include silhouette analysis
            
        Returns:
            Dictionary containing inertia values and silhouette scores
        """
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeansClustering(k=k, random_state=self.random_state)
            kmeans.fit(X)
            inertias.append(kmeans.calculate_inertia(X))
            
            if show_silhouette and k > 1:
                score = silhouette_score(X, kmeans.clusters)
                silhouette_scores.append(score)
            elif k == 1:
                silhouette_scores.append(0)  # Silhouette score undefined for k=1
        
        # Create comprehensive plot
        if show_silhouette:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Elbow plot
            ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Silhouette plot
            ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
            ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax2.set_ylabel('Silhouette Score', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Clusters (k)', fontsize=12)
            plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced elbow analysis saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        return {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k_elbow': self._find_elbow_point(list(k_range), inertias),
            'optimal_k_silhouette': k_range[np.argmax(silhouette_scores)] if silhouette_scores else None
        }
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """
        Find the elbow point using the rate of change method
        
        Args:
            k_values: List of k values
            inertias: List of corresponding inertia values
            
        Returns:
            Optimal k value based on elbow method
        """
        if len(inertias) < 3:
            return k_values[0]
            
        # Calculate rate of change
        rates = []
        for i in range(1, len(inertias)):
            rate = abs(inertias[i-1] - inertias[i])
            rates.append(rate)
        
        # Find the point where rate of change decreases significantly
        max_decrease = 0
        elbow_idx = 0
        
        for i in range(1, len(rates)):
            decrease = rates[i-1] - rates[i]
            if decrease > max_decrease:
                max_decrease = decrease
                elbow_idx = i
        
        return k_values[elbow_idx + 1] if elbow_idx + 1 < len(k_values) else k_values[-1]
    
    def plot_convergence_history(self, X: np.ndarray, save_path: str = None) -> None:
        """
        Plot the convergence history of the algorithm
        
        Args:
            X: Input data
            save_path: Path to save the plot (optional)
        """
        # Re-run algorithm to track convergence
        centroids_history = []
        inertia_history = []
        
        # Initialize centroids
        centroids = self.initialize_centroids(X)
        centroids_history.append(centroids.copy())
        
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            assignments = self.assign_clusters(X, centroids)
            
            # Calculate inertia
            inertia = 0
            for i in range(self.k):
                cluster_points = X[assignments == i]
                if len(cluster_points) > 0:
                    inertia += np.sum((cluster_points - centroids[i]) ** 2)
            inertia_history.append(inertia)
            
            # Update centroids
            new_centroids = self.update_centroids(X, assignments)
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
            centroids_history.append(centroids.copy())
        
        # Plot convergence
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot inertia convergence
        ax1.plot(range(len(inertia_history)), inertia_history, 'b-o', linewidth=2)
        ax1.set_title('Convergence: Inertia vs Iterations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot centroid movement
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for k in range(self.k):
            centroid_path = np.array([hist[k] for hist in centroids_history])
            ax2.plot(centroid_path[:, 0], centroid_path[:, 1], 
                    'o-', color=colors[k % len(colors)], linewidth=2, 
                    markersize=6, label=f'Centroid {k+1}')
            
            # Mark start and end points
            ax2.scatter(centroid_path[0, 0], centroid_path[0, 1], 
                       color=colors[k % len(colors)], s=100, marker='s', 
                       edgecolors='black', linewidth=2)
            ax2.scatter(centroid_path[-1, 0], centroid_path[-1, 1], 
                       color=colors[k % len(colors)], s=100, marker='*', 
                       edgecolors='black', linewidth=2)
        
        ax2.set_title('Centroid Movement During Convergence', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Feature 1', fontsize=12)
        ax2.set_ylabel('Feature 2', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence history saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def plot_cluster_comparison(self, X: np.ndarray, y_true: np.ndarray = None, 
                              save_path: str = None) -> None:
        """
        Create a comprehensive comparison plot showing multiple views
        
        Args:
            X: Input data
            y_true: True cluster labels (optional)
            save_path: Path to save the plot (optional)
        """
        if self.clusters is None:
            raise ValueError("Model must be fitted before plotting")
        
        # Determine subplot layout
        n_plots = 3 if y_true is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        # Plot 1: Original data
        axes[0].scatter(X[:, 0], X[:, 1], alpha=0.7, c='gray', s=50)
        axes[0].set_title('Original Data', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Feature 1', fontsize=12)
        axes[0].set_ylabel('Feature 2', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: K-means results
        for i in range(self.k):
            cluster_points = X[self.clusters == i]
            if len(cluster_points) > 0:
                axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              c=colors[i % len(colors)], label=f'Cluster {i+1}', 
                              alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        axes[1].scatter(self.centroids[:, 0], self.centroids[:, 1], 
                       c='black', marker='X', s=200, linewidths=2, 
                       label='Centroids', edgecolors='white')
        axes[1].set_title('K-Means Results', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Feature 1', fontsize=12)
        axes[1].set_ylabel('Feature 2', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: True clusters (if available)
        if y_true is not None:
            unique_labels = np.unique(y_true)
            for i, label in enumerate(unique_labels):
                cluster_points = X[y_true == label]
                axes[2].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              c=colors[i % len(colors)], label=f'True Cluster {label+1}', 
                              alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
            
            axes[2].set_title('True Clusters', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Feature 1', fontsize=12)
            axes[2].set_ylabel('Feature 2', fontsize=12)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster comparison saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def evaluate_clustering_quality(self, X: np.ndarray, y_true: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics
        
        Args:
            X: Input data
            y_true: True cluster labels (optional)
            
        Returns:
            Dictionary containing quality metrics
        """
        if self.clusters is None:
            raise ValueError("Model must be fitted before evaluation")
        
        metrics = {}
        
        # Inertia (WCSS)
        metrics['inertia'] = self.calculate_inertia(X)
        
        # Silhouette score
        if len(np.unique(self.clusters)) > 1:
            metrics['silhouette_score'] = silhouette_score(X, self.clusters)
        else:
            metrics['silhouette_score'] = 0
        
        # If true labels are available
        if y_true is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(y_true, self.clusters)
        
        return metrics

def demonstrate_kmeans_clustering():
    """
    Comprehensive K-Means clustering demonstration with enhanced visualizations
    """
    print("=== Enhanced K-Means Clustering Demonstration ===")
    
    # Create K-Means instance
    kmeans = KMeansClustering(k=3, random_state=42)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y_true = kmeans.generate_sample_data(n_samples=300, centers=3)
    print(f"Generated {X.shape[0]} data points with {X.shape[1]} features")
    
    # Fit K-Means
    print("\n2. Fitting K-Means clustering...")
    kmeans.fit(X)
    
    # Create comprehensive comparison plot
    print("\n3. Creating comprehensive visualization...")
    kmeans.plot_cluster_comparison(X, y_true, save_path='enhanced_kmeans_comparison.png')
    
    # Plot convergence history
    print("\n4. Analyzing convergence behavior...")
    kmeans.plot_convergence_history(X, save_path='convergence_history.png')
    
    # Enhanced clustering visualization with different styles
    print("\n5. Creating enhanced cluster visualizations...")
    kmeans.plot_clusters(X, title="Professional K-Means Results", 
                        save_path='professional_clusters.png', style='professional')
    
    kmeans.plot_clusters(X, title="Seaborn Style K-Means Results", 
                        save_path='seaborn_clusters.png', style='seaborn')
    
    # Comprehensive elbow method analysis
    print("\n6. Performing comprehensive elbow method analysis...")
    elbow_results = kmeans.elbow_method(X, k_range=range(1, 10), 
                                       save_path='enhanced_elbow_analysis.png', 
                                       show_silhouette=True)
    
    print("\nElbow Method Results:")
    print(f"   - Optimal k (Elbow): {elbow_results['optimal_k_elbow']}")
    print(f"   - Optimal k (Silhouette): {elbow_results['optimal_k_silhouette']}")
    
    print("\nDetailed Analysis:")
    for i, (k, inertia, silhouette) in enumerate(zip(range(1, 10), 
                                                     elbow_results['inertias'], 
                                                     elbow_results['silhouette_scores'])):
        print(f"   k={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.3f}")
    
    # Evaluate clustering quality
    print("\n7. Evaluating clustering quality...")
    quality_metrics = kmeans.evaluate_clustering_quality(X, y_true)
    
    print("\nClustering Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"   - {metric.replace('_', ' ').title()}: {value:.3f}")
    
    # Test prediction on new data
    print("\n8. Testing prediction on new data...")
    new_points = np.array([[0, 0], [3, 3], [-2, 2], [1, -1], [-3, -3]])
    predictions = kmeans.predict(new_points)
    
    print("New points and their predicted clusters:")
    for i, (point, cluster) in enumerate(zip(new_points, predictions)):
        print(f"   Point {point}: Cluster {int(cluster) + 1}")
    
    # Demonstrate different k values
    print("\n9. Comparing different k values...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    k_values = [2, 3, 4, 5, 6, 7]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
             '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    for idx, k in enumerate(k_values):
        kmeans_temp = KMeansClustering(k=k, random_state=42)
        kmeans_temp.fit(X)
        
        for i in range(k):
            cluster_points = X[kmeans_temp.clusters == i]
            if len(cluster_points) > 0:
                axes[idx].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                c=colors[i % len(colors)], alpha=0.7, s=30)
        
        axes[idx].scatter(kmeans_temp.centroids[:, 0], kmeans_temp.centroids[:, 1], 
                         c='black', marker='X', s=100, linewidths=1)
        axes[idx].set_title(f'k = {k}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('K-Means Clustering with Different k Values', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('different_k_comparison.png', dpi=300, bbox_inches='tight')
    print("Different k values comparison saved to different_k_comparison.png")
    plt.close()
    
    print("\n=== Enhanced Demonstration Complete ===")
    print("\nGenerated Files:")
    print("- enhanced_kmeans_comparison.png: Comprehensive comparison view")
    print("- convergence_history.png: Algorithm convergence analysis")
    print("- professional_clusters.png: Professional style visualization")
    print("- seaborn_clusters.png: Seaborn style visualization")
    print("- enhanced_elbow_analysis.png: Comprehensive elbow method with silhouette")
    print("- different_k_comparison.png: Comparison of different k values")
    
    return quality_metrics, elbow_results

if __name__ == "__main__":
    # Set matplotlib backend for non-interactive plotting
    plt.switch_backend('Agg')
    
    # Run demonstration
    demonstrate_kmeans_clustering()
    
    print("\nAll plots have been saved to files:")
    print("- kmeans_comparison.png: Comparison of original data, K-means results, and true clusters")
    print("- elbow_method.png: Elbow method analysis for optimal k")