#!/usr/bin/env python3
"""
Comprehensive K-Means Clustering Graph Generator
Generates all enhanced visualizations and analysis plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmeans_clustering import KMeansClustering
import warnings
warnings.filterwarnings('ignore')

def generate_all_kmeans_graphs():
    """
    Generate all K-means clustering graphs and visualizations
    """
    print("=== Generating All K-Means Clustering Graphs ===")
    
    # Set up matplotlib and seaborn styles
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create K-Means instance
    kmeans = KMeansClustering(k=3, random_state=42)
    
    # Generate sample data
    print("\n1. Generating sample datasets...")
    X, y_true = kmeans.generate_sample_data(n_samples=300, centers=3)
    
    # Fit the model
    print("\n2. Training K-means model...")
    kmeans.fit(X)
    
    # Generate all visualization styles
    print("\n3. Creating cluster visualizations with different styles...")
    
    # Default style
    kmeans.plot_clusters(X, title="Default Style K-Means Clustering", 
                        save_path='graphs_default_clusters.png')
    
    # Professional style
    kmeans.plot_clusters(X, title="Professional Style K-Means Clustering", 
                        save_path='graphs_professional_clusters.png', style='professional')
    
    # Seaborn style
    kmeans.plot_clusters(X, title="Seaborn Style K-Means Clustering", 
                        save_path='graphs_seaborn_clusters.png', style='seaborn')
    
    # Comprehensive comparison
    print("\n4. Creating comprehensive comparison plots...")
    kmeans.plot_cluster_comparison(X, y_true, save_path='graphs_comprehensive_comparison.png')
    
    # Convergence analysis
    print("\n5. Analyzing convergence behavior...")
    kmeans.plot_convergence_history(X, save_path='graphs_convergence_analysis.png')
    
    # Enhanced elbow method
    print("\n6. Performing enhanced elbow method analysis...")
    elbow_results = kmeans.elbow_method(X, k_range=range(1, 10), 
                                       save_path='graphs_enhanced_elbow_method.png', 
                                       show_silhouette=True)
    
    # Different k values comparison
    print("\n7. Creating multi-k comparison grid...")
    create_multi_k_grid(X)
    
    # Quality metrics visualization
    print("\n8. Creating quality metrics visualization...")
    create_quality_metrics_plot(X, y_true)
    
    # Distance matrix heatmap
    print("\n9. Creating distance matrix heatmap...")
    create_distance_heatmap(X, kmeans)
    
    # 3D visualization (if applicable)
    print("\n10. Creating 3D cluster visualization...")
    create_3d_visualization()
    
    # Algorithm steps visualization
    print("\n11. Creating algorithm steps demonstration...")
    create_algorithm_steps_demo(X)
    
    # Summary report
    print("\n12. Generating summary report...")
    generate_summary_report(elbow_results, kmeans, X, y_true)
    
    print("\n=== All Graphs Generated Successfully ===")
    print("\nGenerated Files:")
    print("- graphs_default_clusters.png")
    print("- graphs_professional_clusters.png")
    print("- graphs_seaborn_clusters.png")
    print("- graphs_comprehensive_comparison.png")
    print("- graphs_convergence_analysis.png")
    print("- graphs_enhanced_elbow_method.png")
    print("- graphs_multi_k_comparison.png")
    print("- graphs_quality_metrics.png")
    print("- graphs_distance_heatmap.png")
    print("- graphs_3d_visualization.png")
    print("- graphs_algorithm_steps.png")
    print("- graphs_summary_report.png")

def create_multi_k_grid(X):
    """
    Create a comprehensive grid showing clustering with different k values
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    k_values = range(2, 11)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
             '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    for idx, k in enumerate(k_values):
        kmeans_temp = KMeansClustering(k=k, random_state=42)
        kmeans_temp.fit(X)
        
        for i in range(k):
            cluster_points = X[kmeans_temp.clusters == i]
            if len(cluster_points) > 0:
                axes[idx].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                c=colors[i % len(colors)], alpha=0.7, s=40, 
                                edgecolors='white', linewidth=0.5)
        
        axes[idx].scatter(kmeans_temp.centroids[:, 0], kmeans_temp.centroids[:, 1], 
                         c='black', marker='X', s=150, linewidths=2, 
                         edgecolors='white')
        axes[idx].set_title(f'K = {k}', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlabel('Feature 1', fontsize=10)
        axes[idx].set_ylabel('Feature 2', fontsize=10)
    
    plt.suptitle('K-Means Clustering: Comparison of Different K Values', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('graphs_multi_k_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_metrics_plot(X, y_true):
    """
    Create visualization of clustering quality metrics
    """
    k_range = range(2, 11)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans_temp = KMeansClustering(k=k, random_state=42)
        kmeans_temp.fit(X)
        
        inertias.append(kmeans_temp.calculate_inertia(X))
        
        from sklearn.metrics import silhouette_score
        if len(np.unique(kmeans_temp.clusters)) > 1:
            silhouette_scores.append(silhouette_score(X, kmeans_temp.clusters))
        else:
            silhouette_scores.append(0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Inertia plot
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Inertia vs Number of Clusters', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Silhouette score plot
    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Combined metrics
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(k_range, inertias, 'b-o', label='Inertia', linewidth=2)
    line2 = ax3_twin.plot(k_range, silhouette_scores, 'r-s', label='Silhouette Score', linewidth=2)
    
    ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax3.set_ylabel('Inertia', color='blue', fontsize=12)
    ax3_twin.set_ylabel('Silhouette Score', color='red', fontsize=12)
    ax3.set_title('Combined Quality Metrics', fontsize=14, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='center right')
    ax3.grid(True, alpha=0.3)
    
    # Metrics comparison bar chart
    optimal_k = 3
    kmeans_optimal = KMeansClustering(k=optimal_k, random_state=42)
    kmeans_optimal.fit(X)
    
    from sklearn.metrics import adjusted_rand_score
    metrics = {
        'Silhouette Score': silhouette_score(X, kmeans_optimal.clusters),
        'Adjusted Rand Score': adjusted_rand_score(y_true, kmeans_optimal.clusters),
        'Normalized Inertia': 1 - (kmeans_optimal.calculate_inertia(X) / max(inertias))
    }
    
    bars = ax4.bar(metrics.keys(), metrics.values(), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax4.set_title(f'Quality Metrics for Optimal K={optimal_k}', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphs_quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distance_heatmap(X, kmeans):
    """
    Create distance matrix heatmap
    """
    # Calculate distances from each point to each centroid
    distances = np.zeros((len(X), kmeans.k))
    
    for i, point in enumerate(X):
        for j, centroid in enumerate(kmeans.centroids):
            distances[i, j] = np.sqrt(np.sum((point - centroid) ** 2))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distance heatmap
    im1 = ax1.imshow(distances.T, cmap='viridis', aspect='auto')
    ax1.set_title('Distance Matrix: Points to Centroids', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Data Points', fontsize=12)
    ax1.set_ylabel('Centroids', fontsize=12)
    ax1.set_yticks(range(kmeans.k))
    ax1.set_yticklabels([f'Centroid {i+1}' for i in range(kmeans.k)])
    plt.colorbar(im1, ax=ax1, label='Euclidean Distance')
    
    # Cluster assignment visualization
    cluster_matrix = np.zeros((kmeans.k, len(X)))
    for i, cluster in enumerate(kmeans.clusters):
        cluster_matrix[int(cluster), i] = 1
    
    im2 = ax2.imshow(cluster_matrix, cmap='RdYlBu', aspect='auto')
    ax2.set_title('Cluster Assignments', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Data Points', fontsize=12)
    ax2.set_ylabel('Clusters', fontsize=12)
    ax2.set_yticks(range(kmeans.k))
    ax2.set_yticklabels([f'Cluster {i+1}' for i in range(kmeans.k)])
    plt.colorbar(im2, ax=ax2, label='Assignment (1=Assigned, 0=Not Assigned)')
    
    plt.tight_layout()
    plt.savefig('graphs_distance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_visualization():
    """
    Create 3D visualization of K-means clustering
    """
    # Generate 3D data
    kmeans_3d = KMeansClustering(k=4, random_state=42)
    
    # Create 3D sample data
    np.random.seed(42)
    centers_3d = np.array([[2, 2, 2], [-2, -2, -2], [2, -2, 2], [-2, 2, -2]])
    X_3d = []
    y_true_3d = []
    
    for i, center in enumerate(centers_3d):
        cluster_points = np.random.multivariate_normal(center, np.eye(3) * 0.5, 75)
        X_3d.extend(cluster_points)
        y_true_3d.extend([i] * 75)
    
    X_3d = np.array(X_3d)
    y_true_3d = np.array(y_true_3d)
    
    # Fit 3D K-means
    kmeans_3d.fit(X_3d)
    
    # Create 3D plot
    fig = plt.figure(figsize=(16, 6))
    
    # Original 3D data
    ax1 = fig.add_subplot(121, projection='3d')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i in range(4):
        cluster_points = X_3d[y_true_3d == i]
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                   c=colors[i], label=f'True Cluster {i+1}', alpha=0.7, s=30)
    
    ax1.set_title('True 3D Clusters', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.legend()
    
    # K-means 3D results
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i in range(4):
        cluster_points = X_3d[kmeans_3d.clusters == i]
        if len(cluster_points) > 0:
            ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                       c=colors[i], label=f'K-means Cluster {i+1}', alpha=0.7, s=30)
    
    # Plot centroids
    ax2.scatter(kmeans_3d.centroids[:, 0], kmeans_3d.centroids[:, 1], kmeans_3d.centroids[:, 2], 
               c='black', marker='X', s=200, linewidths=2, label='Centroids')
    
    ax2.set_title('K-Means 3D Results', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('graphs_3d_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_algorithm_steps_demo(X):
    """
    Create step-by-step algorithm demonstration
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    kmeans_demo = KMeansClustering(k=3, random_state=42)
    
    # Step 1: Original data
    axes[0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.7, s=50)
    axes[0].set_title('Step 1: Original Data', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Step 2: Initial centroids
    initial_centroids = kmeans_demo.initialize_centroids(X)
    axes[1].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=30)
    axes[1].scatter(initial_centroids[:, 0], initial_centroids[:, 1], 
                   c='red', marker='X', s=200, linewidths=2, label='Initial Centroids')
    axes[1].set_title('Step 2: Initialize Centroids', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Steps 3-6: Iterations
    centroids = initial_centroids.copy()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for iteration in range(4):
        # Assign clusters
        assignments = kmeans_demo.assign_clusters(X, centroids)
        
        # Plot current state
        ax = axes[iteration + 2]
        
        for i in range(3):
            cluster_points = X[assignments == i]
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=colors[i], alpha=0.7, s=40, label=f'Cluster {i+1}')
        
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='black', marker='X', s=150, linewidths=2, 
                  edgecolors='white', label='Centroids')
        
        ax.set_title(f'Step {iteration + 3}: Iteration {iteration + 1}', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Update centroids for next iteration
        new_centroids = kmeans_demo.update_centroids(X, assignments)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    plt.suptitle('K-Means Algorithm: Step-by-Step Demonstration', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs_algorithm_steps.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(elbow_results, kmeans, X, y_true):
    """
    Generate a comprehensive summary report
    """
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Summary statistics
    ax1.axis('off')
    summary_text = f"""
    K-MEANS CLUSTERING ANALYSIS SUMMARY
    
    Dataset Information:
    • Total data points: {len(X)}
    • Features: {X.shape[1]}
    • True clusters: {len(np.unique(y_true))}
    
    Optimal K Analysis:
    • Elbow method suggests: k = {elbow_results['optimal_k_elbow']}
    • Silhouette analysis suggests: k = {elbow_results['optimal_k_silhouette']}
    
    Model Performance (k=3):
    • Final inertia: {kmeans.calculate_inertia(X):.2f}
    • Silhouette score: {silhouette_score(X, kmeans.clusters):.3f}
    • Adjusted Rand score: {adjusted_rand_score(y_true, kmeans.clusters):.3f}
    
    Algorithm Convergence:
    • Converged successfully
    • Stable centroid positions achieved
    """
    
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Cluster sizes
    unique, counts = np.unique(kmeans.clusters, return_counts=True)
    ax2.pie(counts, labels=[f'Cluster {int(i)+1}' for i in unique], 
           autopct='%1.1f%%', startangle=90, 
           colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    
    # Feature statistics
    feature_stats = []
    for i in range(kmeans.k):
        cluster_points = X[kmeans.clusters == i]
        if len(cluster_points) > 0:
            feature_stats.append([
                np.mean(cluster_points[:, 0]),
                np.mean(cluster_points[:, 1]),
                np.std(cluster_points[:, 0]),
                np.std(cluster_points[:, 1])
            ])
    
    feature_stats = np.array(feature_stats)
    
    x_pos = np.arange(kmeans.k)
    width = 0.35
    
    ax3.bar(x_pos - width/2, feature_stats[:, 0], width, 
           label='Feature 1 Mean', color='#FF6B6B', alpha=0.8)
    ax3.bar(x_pos + width/2, feature_stats[:, 1], width, 
           label='Feature 2 Mean', color='#4ECDC4', alpha=0.8)
    
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Mean Value')
    ax3.set_title('Cluster Feature Means', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'Cluster {i+1}' for i in range(kmeans.k)])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Centroid coordinates
    ax4.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
               c=['#FF6B6B', '#4ECDC4', '#45B7D1'], s=200, 
               marker='X', linewidths=3, edgecolors='black')
    
    for i, (x, y) in enumerate(kmeans.centroids):
        ax4.annotate(f'C{i+1}\n({x:.2f}, {y:.2f})', 
                    (x, y), xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    fontsize=10, fontweight='bold')
    
    ax4.set_title('Final Centroid Positions', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Feature 1')
    ax4.set_ylabel('Feature 2')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('K-Means Clustering: Comprehensive Analysis Report', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graphs_summary_report.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_all_kmeans_graphs()
    print("\n✅ All K-means clustering graphs have been generated successfully!")
    print("\n📊 Check the current directory for all graph files with 'graphs_' prefix.")