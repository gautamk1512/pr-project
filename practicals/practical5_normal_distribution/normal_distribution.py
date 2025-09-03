import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import multivariate_normal, normaltest, shapiro, kstest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

class NormalDistributionGenerator:
    """
    Comprehensive Normal Distribution Sample Generator
    
    This class provides tools for:
    - Generating univariate and multivariate normal distributions
    - Statistical analysis and hypothesis testing
    - Visualization of distributions
    - Parameter estimation and validation
    - Comparison with real-world data
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.generated_samples = {}
        
    def generate_univariate_normal(self, mean=0, std=1, n_samples=1000, name="sample"):
        """
        Generate samples from univariate normal distribution
        
        Args:
            mean (float): Mean of the distribution
            std (float): Standard deviation
            n_samples (int): Number of samples to generate
            name (str): Name for the sample set
        
        Returns:
            tuple: (samples, parameters)
        """
        samples = np.random.normal(mean, std, n_samples)
        
        parameters = {
            'mean': mean,
            'std': std,
            'variance': std**2,
            'n_samples': n_samples,
            'distribution_type': 'univariate_normal'
        }
        
        self.generated_samples[name] = {
            'samples': samples,
            'parameters': parameters
        }
        
        return samples, parameters
    
    def generate_multivariate_normal(self, mean, cov, n_samples=1000, name="multivariate_sample"):
        """
        Generate samples from multivariate normal distribution
        
        Args:
            mean (array-like): Mean vector
            cov (array-like): Covariance matrix
            n_samples (int): Number of samples to generate
            name (str): Name for the sample set
        
        Returns:
            tuple: (samples, parameters)
        """
        mean = np.array(mean)
        cov = np.array(cov)
        
        # Validate covariance matrix
        if not self._is_positive_definite(cov):
            print("Warning: Covariance matrix is not positive definite. Making it positive definite.")
            cov = self._make_positive_definite(cov)
        
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        
        parameters = {
            'mean': mean,
            'covariance': cov,
            'correlation': self._cov_to_corr(cov),
            'n_samples': n_samples,
            'n_dimensions': len(mean),
            'distribution_type': 'multivariate_normal'
        }
        
        self.generated_samples[name] = {
            'samples': samples,
            'parameters': parameters
        }
        
        return samples, parameters
    
    def _is_positive_definite(self, matrix):
        """
        Check if matrix is positive definite
        
        Args:
            matrix (array): Matrix to check
        
        Returns:
            bool: True if positive definite
        """
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def _make_positive_definite(self, matrix, regularization=1e-6):
        """
        Make matrix positive definite by adding regularization
        
        Args:
            matrix (array): Input matrix
            regularization (float): Regularization parameter
        
        Returns:
            array: Positive definite matrix
        """
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, regularization)
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    def _cov_to_corr(self, cov):
        """
        Convert covariance matrix to correlation matrix
        
        Args:
            cov (array): Covariance matrix
        
        Returns:
            array: Correlation matrix
        """
        std_devs = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_devs, std_devs)
        return corr
    
    def estimate_parameters(self, samples):
        """
        Estimate parameters from samples
        
        Args:
            samples (array): Sample data
        
        Returns:
            dict: Estimated parameters
        """
        if samples.ndim == 1:
            # Univariate case
            estimated_mean = np.mean(samples)
            estimated_std = np.std(samples, ddof=1)
            estimated_var = np.var(samples, ddof=1)
            
            return {
                'estimated_mean': estimated_mean,
                'estimated_std': estimated_std,
                'estimated_variance': estimated_var,
                'sample_size': len(samples),
                'distribution_type': 'univariate'
            }
        else:
            # Multivariate case
            estimated_mean = np.mean(samples, axis=0)
            estimated_cov = np.cov(samples.T)
            estimated_corr = self._cov_to_corr(estimated_cov)
            
            return {
                'estimated_mean': estimated_mean,
                'estimated_covariance': estimated_cov,
                'estimated_correlation': estimated_corr,
                'sample_size': samples.shape[0],
                'n_dimensions': samples.shape[1],
                'distribution_type': 'multivariate'
            }
    
    def test_normality(self, samples, alpha=0.05):
        """
        Test normality of samples using multiple tests
        
        Args:
            samples (array): Sample data
            alpha (float): Significance level
        
        Returns:
            dict: Test results
        """
        if samples.ndim > 1:
            # For multivariate, test each dimension separately
            results = {}
            for i in range(samples.shape[1]):
                results[f'dimension_{i}'] = self.test_normality(samples[:, i], alpha)
            return results
        
        # Univariate normality tests
        results = {}
        
        # Shapiro-Wilk test (best for small samples)
        if len(samples) <= 5000:
            shapiro_stat, shapiro_p = shapiro(samples)
            results['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > alpha,
                'test_name': 'Shapiro-Wilk'
            }
        
        # D'Agostino-Pearson test
        try:
            dagostino_stat, dagostino_p = normaltest(samples)
            results['dagostino_pearson'] = {
                'statistic': dagostino_stat,
                'p_value': dagostino_p,
                'is_normal': dagostino_p > alpha,
                'test_name': "D'Agostino-Pearson"
            }
        except:
            pass
        
        # Kolmogorov-Smirnov test
        # Compare with normal distribution with estimated parameters
        estimated_mean = np.mean(samples)
        estimated_std = np.std(samples, ddof=1)
        ks_stat, ks_p = kstest(samples, lambda x: stats.norm.cdf(x, estimated_mean, estimated_std))
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': ks_p > alpha,
            'test_name': 'Kolmogorov-Smirnov'
        }
        
        # Summary
        normal_count = sum(1 for test in results.values() if test['is_normal'])
        total_tests = len(results)
        
        results['summary'] = {
            'tests_passed': normal_count,
            'total_tests': total_tests,
            'proportion_passed': normal_count / total_tests,
            'overall_normal': normal_count >= total_tests // 2
        }
        
        return results
    
    def plot_univariate_distribution(self, samples, true_params=None, title="Univariate Normal Distribution"):
        """
        Plot univariate distribution with analysis
        
        Args:
            samples (array): Sample data
            true_params (dict): True parameters if known
            title (str): Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Histogram with fitted normal
        axes[0, 0].hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Fit normal distribution
        estimated_mean = np.mean(samples)
        estimated_std = np.std(samples, ddof=1)
        
        x = np.linspace(samples.min(), samples.max(), 100)
        fitted_normal = stats.norm.pdf(x, estimated_mean, estimated_std)
        axes[0, 0].plot(x, fitted_normal, 'r-', linewidth=2, label=f'Fitted Normal\n(μ={estimated_mean:.3f}, σ={estimated_std:.3f})')
        
        if true_params:
            true_normal = stats.norm.pdf(x, true_params['mean'], true_params['std'])
            axes[0, 0].plot(x, true_normal, 'g--', linewidth=2, label=f'True Normal\n(μ={true_params["mean"]:.3f}, σ={true_params["std"]:.3f})')
        
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Histogram with Fitted Normal')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(samples, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot(samples, vert=True)
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Box Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics summary
        stats_text = f"""
        Sample Statistics:
        Mean: {estimated_mean:.4f}
        Std Dev: {estimated_std:.4f}
        Variance: {estimated_std**2:.4f}
        Skewness: {stats.skew(samples):.4f}
        Kurtosis: {stats.kurtosis(samples):.4f}
        Min: {samples.min():.4f}
        Max: {samples.max():.4f}
        Sample Size: {len(samples)}
        """
        
        if true_params:
            stats_text += f"""
        
        True Parameters:
        Mean: {true_params['mean']:.4f}
        Std Dev: {true_params['std']:.4f}
        Variance: {true_params['std']**2:.4f}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistics Summary')
        
        plt.tight_layout()
        plt.show()
    
    def plot_multivariate_distribution(self, samples, true_params=None, title="Multivariate Normal Distribution"):
        """
        Plot multivariate distribution analysis
        
        Args:
            samples (array): Sample data (n_samples x n_features)
            true_params (dict): True parameters if known
            title (str): Plot title
        """
        n_dims = samples.shape[1]
        
        if n_dims == 2:
            self._plot_bivariate_distribution(samples, true_params, title)
        elif n_dims > 2:
            self._plot_high_dimensional_distribution(samples, true_params, title)
        else:
            print("Use plot_univariate_distribution for 1D data")
    
    def _plot_bivariate_distribution(self, samples, true_params=None, title="Bivariate Normal Distribution"):
        """
        Plot bivariate normal distribution
        
        Args:
            samples (array): 2D sample data
            true_params (dict): True parameters if known
            title (str): Plot title
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        # Scatter plot
        axes[0, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Dimension 1')
        axes[0, 0].set_ylabel('Dimension 2')
        axes[0, 0].set_title('Scatter Plot')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Contour plot
        estimated_mean = np.mean(samples, axis=0)
        estimated_cov = np.cov(samples.T)
        
        x = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 100)
        y = np.linspace(samples[:, 1].min(), samples[:, 1].max(), 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        
        rv = multivariate_normal(estimated_mean, estimated_cov)
        axes[0, 1].contour(X, Y, rv.pdf(pos), levels=10)
        axes[0, 1].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)
        axes[0, 1].set_xlabel('Dimension 1')
        axes[0, 1].set_ylabel('Dimension 2')
        axes[0, 1].set_title('Contour Plot with Data')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Marginal distributions
        axes[0, 2].hist(samples[:, 0], bins=30, density=True, alpha=0.7, orientation='vertical')
        x_range = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 100)
        axes[0, 2].plot(stats.norm.pdf(x_range, estimated_mean[0], np.sqrt(estimated_cov[0, 0])), 
                       x_range, 'r-', linewidth=2)
        axes[0, 2].set_ylabel('Dimension 1')
        axes[0, 2].set_xlabel('Density')
        axes[0, 2].set_title('Marginal Distribution (Dim 1)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Marginal distribution for dimension 2
        axes[1, 0].hist(samples[:, 1], bins=30, density=True, alpha=0.7)
        y_range = np.linspace(samples[:, 1].min(), samples[:, 1].max(), 100)
        axes[1, 0].plot(y_range, stats.norm.pdf(y_range, estimated_mean[1], np.sqrt(estimated_cov[1, 1])), 
                       'r-', linewidth=2)
        axes[1, 0].set_xlabel('Dimension 2')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Marginal Distribution (Dim 2)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation matrix heatmap
        estimated_corr = self._cov_to_corr(estimated_cov)
        im = axes[1, 1].imshow(estimated_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Correlation Matrix')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_xticklabels(['Dim 1', 'Dim 2'])
        axes[1, 1].set_yticklabels(['Dim 1', 'Dim 2'])
        
        # Add correlation values
        for i in range(2):
            for j in range(2):
                axes[1, 1].text(j, i, f'{estimated_corr[i, j]:.3f}', 
                               ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        # Statistics summary
        stats_text = f"""
        Estimated Parameters:
        Mean: [{estimated_mean[0]:.4f}, {estimated_mean[1]:.4f}]
        
        Covariance Matrix:
        [{estimated_cov[0,0]:.4f}, {estimated_cov[0,1]:.4f}]
        [{estimated_cov[1,0]:.4f}, {estimated_cov[1,1]:.4f}]
        
        Correlation: {estimated_corr[0,1]:.4f}
        
        Sample Size: {samples.shape[0]}
        """
        
        if true_params:
            true_corr = self._cov_to_corr(true_params['covariance'])
            stats_text += f"""
        
        True Parameters:
        Mean: [{true_params['mean'][0]:.4f}, {true_params['mean'][1]:.4f}]
        
        True Covariance:
        [{true_params['covariance'][0,0]:.4f}, {true_params['covariance'][0,1]:.4f}]
        [{true_params['covariance'][1,0]:.4f}, {true_params['covariance'][1,1]:.4f}]
        
        True Correlation: {true_corr[0,1]:.4f}
        """
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=9)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Statistics Summary')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_high_dimensional_distribution(self, samples, true_params=None, title="High-Dimensional Normal Distribution"):
        """
        Plot high-dimensional distribution using PCA and pairwise plots
        
        Args:
            samples (array): High-dimensional sample data
            true_params (dict): True parameters if known
            title (str): Plot title
        """
        n_dims = samples.shape[1]
        
        # PCA for visualization
        pca = PCA(n_components=min(3, n_dims))
        samples_pca = pca.fit_transform(samples)
        
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f"{title} (PCA Projection)", fontsize=16)
        
        # 3D PCA plot if possible
        if samples_pca.shape[1] >= 3:
            ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            ax1.scatter(samples_pca[:, 0], samples_pca[:, 1], samples_pca[:, 2], alpha=0.6, s=20)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
            ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} var)')
            ax1.set_title('3D PCA Projection')
        
        # 2D PCA plot
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.6, s=20)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax2.set_title('2D PCA Projection')
        ax2.grid(True, alpha=0.3)
        
        # Explained variance plot
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.set_title('PCA Explained Variance')
        ax3.grid(True, alpha=0.3)
        
        # Correlation matrix heatmap
        ax4 = fig.add_subplot(2, 3, 4)
        estimated_corr = self._cov_to_corr(np.cov(samples.T))
        im = ax4.imshow(estimated_corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title('Correlation Matrix')
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        # Pairwise plot for first few dimensions
        n_pairs = min(4, n_dims)
        if n_pairs >= 2:
            ax5 = fig.add_subplot(2, 3, 5)
            for i in range(n_pairs):
                for j in range(i+1, n_pairs):
                    ax5.scatter(samples[:, i], samples[:, j], alpha=0.3, s=10, 
                              label=f'Dim {i+1} vs {j+1}')
            ax5.set_title('Pairwise Scatter (First Few Dims)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Statistics summary
        estimated_mean = np.mean(samples, axis=0)
        estimated_cov = np.cov(samples.T)
        
        stats_text = f"""
        Dataset Information:
        Dimensions: {n_dims}
        Sample Size: {samples.shape[0]}
        
        Mean Vector (first 5):
        {estimated_mean[:5]}
        
        Diagonal of Covariance (first 5):
        {np.diag(estimated_cov)[:5]}
        
        PCA Information:
        Total Variance Explained (3 PCs): {pca.explained_variance_ratio_[:3].sum():.2%}
        """
        
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Statistics Summary')
        
        plt.tight_layout()
        plt.show()
    
    def compare_distributions(self, samples_list, names_list, title="Distribution Comparison"):
        """
        Compare multiple distributions
        
        Args:
            samples_list (list): List of sample arrays
            names_list (list): List of names for each sample set
            title (str): Plot title
        """
        n_distributions = len(samples_list)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_distributions))
        
        # Overlaid histograms
        for i, (samples, name) in enumerate(zip(samples_list, names_list)):
            if samples.ndim == 1:
                axes[0, 0].hist(samples, bins=30, alpha=0.6, label=name, 
                              color=colors[i], density=True)
            else:
                # For multivariate, plot first dimension
                axes[0, 0].hist(samples[:, 0], bins=30, alpha=0.6, 
                              label=f"{name} (Dim 1)", color=colors[i], density=True)
        
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Overlaid Histograms')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plots
        box_data = []
        box_labels = []
        for samples, name in zip(samples_list, names_list):
            if samples.ndim == 1:
                box_data.append(samples)
                box_labels.append(name)
            else:
                # For multivariate, use first dimension
                box_data.append(samples[:, 0])
                box_labels.append(f"{name} (Dim 1)")
        
        axes[0, 1].boxplot(box_data, labels=box_labels)
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Box Plot Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Q-Q plots against normal
        for i, (samples, name) in enumerate(zip(samples_list, names_list)):
            if samples.ndim == 1:
                sample_data = samples
            else:
                sample_data = samples[:, 0]  # First dimension
            
            stats.probplot(sample_data, dist="norm", plot=axes[1, 0])
        
        axes[1, 0].set_title('Q-Q Plots (Normal)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics comparison table
        stats_data = []
        for samples, name in zip(samples_list, names_list):
            if samples.ndim == 1:
                sample_data = samples
            else:
                sample_data = samples[:, 0]  # First dimension
            
            stats_data.append({
                'Distribution': name,
                'Mean': f"{np.mean(sample_data):.4f}",
                'Std': f"{np.std(sample_data, ddof=1):.4f}",
                'Skewness': f"{stats.skew(sample_data):.4f}",
                'Kurtosis': f"{stats.kurtosis(sample_data):.4f}",
                'Size': len(sample_data)
            })
        
        df = pd.DataFrame(stats_data)
        
        # Create table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=df.values, colLabels=df.columns, 
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Statistics Comparison')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_central_limit_theorem(self, population_dist='uniform', sample_sizes=[1, 5, 10, 30], n_samples=1000):
        """
        Demonstrate Central Limit Theorem
        
        Args:
            population_dist (str): 'uniform', 'exponential', 'binomial'
            sample_sizes (list): List of sample sizes to demonstrate
            n_samples (int): Number of sample means to generate
        """
        print(f"\n{'='*60}")
        print(f"CENTRAL LIMIT THEOREM DEMONSTRATION")
        print(f"Population Distribution: {population_dist.upper()}")
        print(f"{'='*60}")
        
        # Generate population
        if population_dist == 'uniform':
            population = np.random.uniform(0, 10, 10000)
            pop_mean = 5.0
            pop_var = (10**2) / 12  # Variance of uniform distribution
        elif population_dist == 'exponential':
            population = np.random.exponential(2, 10000)
            pop_mean = 2.0
            pop_var = 4.0  # Variance of exponential distribution
        elif population_dist == 'binomial':
            population = np.random.binomial(20, 0.3, 10000)
            pop_mean = 6.0  # n * p
            pop_var = 4.2   # n * p * (1-p)
        
        fig, axes = plt.subplots(2, len(sample_sizes), figsize=(4*len(sample_sizes), 10))
        fig.suptitle(f'Central Limit Theorem - {population_dist.title()} Population', fontsize=16)
        
        sample_means_list = []
        
        for i, n in enumerate(sample_sizes):
            # Generate sample means
            sample_means = []
            for _ in range(n_samples):
                sample = np.random.choice(population, n, replace=True)
                sample_means.append(np.mean(sample))
            
            sample_means = np.array(sample_means)
            sample_means_list.append(sample_means)
            
            # Plot histogram of sample means
            axes[0, i].hist(sample_means, bins=30, density=True, alpha=0.7, 
                           color='skyblue', edgecolor='black')
            
            # Overlay theoretical normal distribution
            theoretical_mean = pop_mean
            theoretical_std = np.sqrt(pop_var / n)
            
            x = np.linspace(sample_means.min(), sample_means.max(), 100)
            theoretical_normal = stats.norm.pdf(x, theoretical_mean, theoretical_std)
            axes[0, i].plot(x, theoretical_normal, 'r-', linewidth=2, 
                           label=f'Theoretical N({theoretical_mean:.2f}, {theoretical_std:.3f}²)')
            
            axes[0, i].set_title(f'Sample Size n={n}')
            axes[0, i].set_xlabel('Sample Mean')
            axes[0, i].set_ylabel('Density')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(sample_means, dist="norm", plot=axes[1, i])
            axes[1, i].set_title(f'Q-Q Plot (n={n})')
            axes[1, i].grid(True, alpha=0.3)
            
            # Print statistics
            empirical_mean = np.mean(sample_means)
            empirical_std = np.std(sample_means, ddof=1)
            
            print(f"\nSample Size n={n}:")
            print(f"  Theoretical: μ={theoretical_mean:.4f}, σ={theoretical_std:.4f}")
            print(f"  Empirical:   μ={empirical_mean:.4f}, σ={empirical_std:.4f}")
            print(f"  Difference:  μ={abs(theoretical_mean-empirical_mean):.4f}, σ={abs(theoretical_std-empirical_std):.4f}")
        
        plt.tight_layout()
        plt.show()
        
        # Test normality for largest sample size
        if len(sample_sizes) > 0:
            largest_n = max(sample_sizes)
            largest_idx = sample_sizes.index(largest_n)
            normality_results = self.test_normality(sample_means_list[largest_idx])
            
            print(f"\nNormality Tests for Sample Means (n={largest_n}):")
            for test_name, result in normality_results.items():
                if test_name != 'summary':
                    print(f"  {result['test_name']}: p-value = {result['p_value']:.6f}, Normal = {result['is_normal']}")
        
        return sample_means_list
    
    def comprehensive_normal_analysis(self):
        """
        Comprehensive demonstration of normal distribution concepts
        
        Returns:
            dict: Analysis results
        """
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE NORMAL DISTRIBUTION ANALYSIS")
        print(f"{'='*70}")
        
        results = {}
        
        # 1. Univariate Normal Distribution
        print("\n1. Univariate Normal Distribution Analysis")
        print("-" * 50)
        
        # Generate samples with different parameters
        samples1, params1 = self.generate_univariate_normal(mean=0, std=1, n_samples=1000, name="standard_normal")
        samples2, params2 = self.generate_univariate_normal(mean=5, std=2, n_samples=1000, name="shifted_normal")
        samples3, params3 = self.generate_univariate_normal(mean=0, std=0.5, n_samples=1000, name="narrow_normal")
        
        # Plot and analyze
        self.plot_univariate_distribution(samples1, params1, "Standard Normal Distribution (μ=0, σ=1)")
        self.plot_univariate_distribution(samples2, params2, "Shifted Normal Distribution (μ=5, σ=2)")
        
        # Compare distributions
        self.compare_distributions([samples1, samples2, samples3], 
                                 ["Standard (0,1)", "Shifted (5,2)", "Narrow (0,0.5)"],
                                 "Univariate Normal Comparison")
        
        # Test normality
        normality1 = self.test_normality(samples1)
        print(f"\nNormality test results for Standard Normal:")
        for test_name, result in normality1.items():
            if test_name != 'summary':
                print(f"  {result['test_name']}: p-value = {result['p_value']:.6f}, Normal = {result['is_normal']}")
        
        results['univariate'] = {
            'samples': [samples1, samples2, samples3],
            'parameters': [params1, params2, params3],
            'normality_tests': normality1
        }
        
        # 2. Bivariate Normal Distribution
        print("\n\n2. Bivariate Normal Distribution Analysis")
        print("-" * 50)
        
        # Independent case
        mean_indep = [0, 0]
        cov_indep = [[1, 0], [0, 1]]
        samples_indep, params_indep = self.generate_multivariate_normal(
            mean_indep, cov_indep, n_samples=1000, name="independent_bivariate"
        )
        
        # Correlated case
        mean_corr = [2, 3]
        cov_corr = [[2, 1.5], [1.5, 3]]
        samples_corr, params_corr = self.generate_multivariate_normal(
            mean_corr, cov_corr, n_samples=1000, name="correlated_bivariate"
        )
        
        # Negative correlation
        mean_neg = [0, 0]
        cov_neg = [[1, -0.8], [-0.8, 1]]
        samples_neg, params_neg = self.generate_multivariate_normal(
            mean_neg, cov_neg, n_samples=1000, name="negative_corr_bivariate"
        )
        
        # Plot and analyze
        self.plot_multivariate_distribution(samples_indep, params_indep, 
                                          "Independent Bivariate Normal")
        self.plot_multivariate_distribution(samples_corr, params_corr, 
                                          "Positively Correlated Bivariate Normal")
        self.plot_multivariate_distribution(samples_neg, params_neg, 
                                          "Negatively Correlated Bivariate Normal")
        
        results['bivariate'] = {
            'independent': {'samples': samples_indep, 'parameters': params_indep},
            'positive_corr': {'samples': samples_corr, 'parameters': params_corr},
            'negative_corr': {'samples': samples_neg, 'parameters': params_neg}
        }
        
        # 3. High-dimensional Normal Distribution
        print("\n\n3. High-Dimensional Normal Distribution Analysis")
        print("-" * 50)
        
        # Generate 5D normal distribution
        mean_5d = np.array([1, 2, 3, 4, 5])
        # Create a structured covariance matrix
        cov_5d = np.eye(5) + 0.3 * np.ones((5, 5))
        np.fill_diagonal(cov_5d, [1, 2, 1.5, 3, 2.5])
        
        samples_5d, params_5d = self.generate_multivariate_normal(
            mean_5d, cov_5d, n_samples=1000, name="high_dim_normal"
        )
        
        self.plot_multivariate_distribution(samples_5d, params_5d, 
                                          "5-Dimensional Normal Distribution")
        
        results['high_dimensional'] = {
            'samples': samples_5d,
            'parameters': params_5d
        }
        
        # 4. Central Limit Theorem Demonstration
        print("\n\n4. Central Limit Theorem Demonstration")
        print("-" * 50)
        
        clt_uniform = self.demonstrate_central_limit_theorem('uniform', [1, 5, 10, 30])
        clt_exponential = self.demonstrate_central_limit_theorem('exponential', [1, 5, 10, 30])
        
        results['central_limit_theorem'] = {
            'uniform': clt_uniform,
            'exponential': clt_exponential
        }
        
        # 5. Parameter Estimation Analysis
        print("\n\n5. Parameter Estimation Analysis")
        print("-" * 50)
        
        # Test parameter estimation accuracy with different sample sizes
        true_mean, true_std = 10, 3
        sample_sizes = [10, 50, 100, 500, 1000, 5000]
        
        estimation_results = []
        for n in sample_sizes:
            samples, _ = self.generate_univariate_normal(true_mean, true_std, n)
            estimated_params = self.estimate_parameters(samples)
            
            mean_error = abs(estimated_params['estimated_mean'] - true_mean)
            std_error = abs(estimated_params['estimated_std'] - true_std)
            
            estimation_results.append({
                'sample_size': n,
                'mean_error': mean_error,
                'std_error': std_error,
                'estimated_mean': estimated_params['estimated_mean'],
                'estimated_std': estimated_params['estimated_std']
            })
            
            print(f"n={n:4d}: Mean={estimated_params['estimated_mean']:.4f} (error={mean_error:.4f}), "
                  f"Std={estimated_params['estimated_std']:.4f} (error={std_error:.4f})")
        
        # Plot estimation accuracy
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sample_sizes_arr = [r['sample_size'] for r in estimation_results]
        mean_errors = [r['mean_error'] for r in estimation_results]
        std_errors = [r['std_error'] for r in estimation_results]
        
        axes[0].loglog(sample_sizes_arr, mean_errors, 'bo-', label='Mean Estimation Error')
        axes[0].loglog(sample_sizes_arr, std_errors, 'ro-', label='Std Estimation Error')
        axes[0].set_xlabel('Sample Size')
        axes[0].set_ylabel('Estimation Error')
        axes[0].set_title('Parameter Estimation Accuracy vs Sample Size')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        estimated_means = [r['estimated_mean'] for r in estimation_results]
        estimated_stds = [r['estimated_std'] for r in estimation_results]
        
        axes[1].semilogx(sample_sizes_arr, estimated_means, 'bo-', label='Estimated Mean')
        axes[1].axhline(y=true_mean, color='b', linestyle='--', label=f'True Mean ({true_mean})')
        axes[1].semilogx(sample_sizes_arr, estimated_stds, 'ro-', label='Estimated Std')
        axes[1].axhline(y=true_std, color='r', linestyle='--', label=f'True Std ({true_std})')
        axes[1].set_xlabel('Sample Size')
        axes[1].set_ylabel('Parameter Value')
        axes[1].set_title('Parameter Convergence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        results['parameter_estimation'] = estimation_results
        
        print(f"\n{'='*70}")
        print(f"NORMAL DISTRIBUTION ANALYSIS COMPLETE")
        print(f"{'='*70}")
        
        return results

def demonstrate_normal_distributions():
    """
    Main demonstration function
    """
    print("NORMAL DISTRIBUTION SAMPLE GENERATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize generator
    generator = NormalDistributionGenerator(random_state=42)
    
    # Run comprehensive analysis
    results = generator.comprehensive_normal_analysis()
    
    return results

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_normal_distributions()