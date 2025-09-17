import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    Comprehensive Feature Engineering and Representation toolkit
    
    This class provides various feature engineering techniques:
    - Feature creation and transformation
    - Feature selection methods
    - Dimensionality reduction
    - Feature scaling and normalization
    - Custom feature combinations
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_history = {}
        self.transformers = {}
        
    def load_dataset(self, dataset_name='iris'):
        """
        Load datasets for feature engineering experiments
        
        Args:
            dataset_name (str): 'iris', 'wine', 'breast_cancer', 'synthetic'
        
        Returns:
            tuple: (X, y, feature_names, target_names)
        """
        if dataset_name == 'iris':
            data = load_iris()
            return data.data, data.target, data.feature_names, data.target_names
        
        elif dataset_name == 'wine':
            data = load_wine()
            return data.data, data.target, data.feature_names, data.target_names
        
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            return data.data, data.target, data.feature_names, data.target_names
        
        elif dataset_name == 'synthetic':
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=10,
                n_redundant=5, n_clusters_per_class=1, random_state=self.random_state
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            target_names = [f'class_{i}' for i in range(len(np.unique(y)))]
            return X, y, feature_names, target_names
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def create_polynomial_features(self, X, degree=2, interaction_only=False, include_bias=False):
        """
        Create polynomial features
        
        Args:
            X (array): Input features
            degree (int): Polynomial degree
            interaction_only (bool): Only interaction terms
            include_bias (bool): Include bias column
        
        Returns:
            tuple: (transformed_features, feature_names, transformer)
        """
        poly = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out()
        
        self.transformers['polynomial'] = poly
        
        return X_poly, feature_names, poly
    
    def create_statistical_features(self, X, feature_names=None):
        """
        Create statistical features from existing features
        
        Args:
            X (array): Input features
            feature_names (list): Original feature names
        
        Returns:
            tuple: (new_features, new_feature_names)
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(X, columns=feature_names)
        
        # Statistical features
        stat_features = pd.DataFrame()
        
        # Row-wise statistics
        stat_features['row_mean'] = df.mean(axis=1)
        stat_features['row_std'] = df.std(axis=1)
        stat_features['row_min'] = df.min(axis=1)
        stat_features['row_max'] = df.max(axis=1)
        stat_features['row_range'] = stat_features['row_max'] - stat_features['row_min']
        stat_features['row_skew'] = df.skew(axis=1)
        stat_features['row_kurtosis'] = df.kurtosis(axis=1)
        
        # Pairwise ratios (for first few features to avoid explosion)
        n_ratio_features = min(5, len(feature_names))
        for i in range(n_ratio_features):
            for j in range(i+1, n_ratio_features):
                col1, col2 = feature_names[i], feature_names[j]
                # Avoid division by zero
                ratio_name = f'{col1}_div_{col2}'
                stat_features[ratio_name] = df[col1] / (df[col2] + 1e-8)
        
        # Pairwise differences
        for i in range(n_ratio_features):
            for j in range(i+1, n_ratio_features):
                col1, col2 = feature_names[i], feature_names[j]
                diff_name = f'{col1}_minus_{col2}'
                stat_features[diff_name] = df[col1] - df[col2]
        
        # Log transformations (for positive features)
        for col in feature_names[:n_ratio_features]:
            if (df[col] > 0).all():
                stat_features[f'log_{col}'] = np.log(df[col] + 1)
        
        # Square root transformations
        for col in feature_names[:n_ratio_features]:
            if (df[col] >= 0).all():
                stat_features[f'sqrt_{col}'] = np.sqrt(df[col])
        
        return stat_features.values, list(stat_features.columns)
    
    def create_binning_features(self, X, feature_names=None, n_bins=5, strategy='uniform'):
        """
        Create binning/discretization features
        
        Args:
            X (array): Input features
            feature_names (list): Feature names
            n_bins (int): Number of bins
            strategy (str): 'uniform', 'quantile', or 'kmeans'
        
        Returns:
            tuple: (binned_features, bin_feature_names)
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        from sklearn.preprocessing import KBinsDiscretizer
        
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, 
            encode='ordinal', 
            strategy=strategy
        )
        
        X_binned = discretizer.fit_transform(X)
        bin_feature_names = [f'{name}_bin' for name in feature_names]
        
        self.transformers['binning'] = discretizer
        
        return X_binned, bin_feature_names
    
    def feature_selection_univariate(self, X, y, k=10, score_func=f_classif):
        """
        Univariate feature selection
        
        Args:
            X (array): Features
            y (array): Target
            k (int): Number of features to select
            score_func: Scoring function
        
        Returns:
            tuple: (selected_features, selector, scores)
        """
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get feature scores
        scores = selector.scores_
        selected_indices = selector.get_support(indices=True)
        
        return X_selected, selector, scores, selected_indices
    
    def feature_selection_rfe(self, X, y, estimator=None, n_features=10):
        """
        Recursive Feature Elimination
        
        Args:
            X (array): Features
            y (array): Target
            estimator: Base estimator
            n_features (int): Number of features to select
        
        Returns:
            tuple: (selected_features, selector, rankings)
        """
        if estimator is None:
            estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        
        rankings = selector.ranking_
        selected_indices = selector.get_support(indices=True)
        
        return X_selected, selector, rankings, selected_indices
    
    def feature_selection_importance(self, X, y, estimator=None, threshold=0.01):
        """
        Feature selection based on feature importance
        
        Args:
            X (array): Features
            y (array): Target
            estimator: Tree-based estimator
            threshold (float): Importance threshold
        
        Returns:
            tuple: (selected_features, importances, selected_indices)
        """
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            )
        
        estimator.fit(X, y)
        importances = estimator.feature_importances_
        
        # Select features above threshold
        selected_indices = np.where(importances >= threshold)[0]
        X_selected = X[:, selected_indices]
        
        return X_selected, importances, selected_indices
    
    def dimensionality_reduction_pca(self, X, n_components=None, explained_variance_ratio=0.95):
        """
        Principal Component Analysis
        
        Args:
            X (array): Features
            n_components (int): Number of components
            explained_variance_ratio (float): Minimum variance to explain
        
        Returns:
            tuple: (transformed_features, pca_model, explained_variance)
        """
        if n_components is None:
            # Find number of components for desired variance
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= explained_variance_ratio) + 1
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        self.transformers['pca'] = pca
        
        return X_pca, pca, pca.explained_variance_ratio_
    
    def dimensionality_reduction_ica(self, X, n_components=None):
        """
        Independent Component Analysis
        
        Args:
            X (array): Features
            n_components (int): Number of components
        
        Returns:
            tuple: (transformed_features, ica_model)
        """
        if n_components is None:
            n_components = min(X.shape[1], X.shape[0] // 2)
        
        ica = FastICA(n_components=n_components, random_state=self.random_state)
        X_ica = ica.fit_transform(X)
        
        self.transformers['ica'] = ica
        
        return X_ica, ica
    
    def dimensionality_reduction_tsne(self, X, n_components=2, perplexity=30):
        """
        t-SNE dimensionality reduction (mainly for visualization)
        
        Args:
            X (array): Features
            n_components (int): Number of components (usually 2 or 3)
            perplexity (float): Perplexity parameter
        
        Returns:
            tuple: (transformed_features, tsne_model)
        """
        tsne = TSNE(
            n_components=n_components, 
            perplexity=perplexity, 
            random_state=self.random_state
        )
        X_tsne = tsne.fit_transform(X)
        
        return X_tsne, tsne
    
    def scale_features(self, X, method='standard'):
        """
        Scale features using different methods
        
        Args:
            X (array): Features
            method (str): 'standard', 'minmax', 'robust'
        
        Returns:
            tuple: (scaled_features, scaler)
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_scaled = scaler.fit_transform(X)
        self.transformers[f'scaler_{method}'] = scaler
        
        return X_scaled, scaler
    
    def evaluate_feature_set(self, X, y, feature_names=None, test_size=0.2, cv_folds=5):
        """
        Evaluate a feature set using multiple classifiers
        
        Args:
            X (array): Features
            y (array): Target
            feature_names (list): Feature names
            test_size (float): Test set proportion
            cv_folds (int): Cross-validation folds
        
        Returns:
            dict: Evaluation results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        results = {}
        
        for name, clf in classifiers.items():
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv_folds)
            
            # Train and test
            clf.fit(X_train_scaled, y_train)
            train_pred = clf.predict(X_train_scaled)
            test_pred = clf.predict(X_test_scaled)
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'model': clf
            }
        
        # Feature set info
        results['feature_info'] = {
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'feature_names': list(feature_names) if feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
        }
        
        return results
    
    def plot_feature_importance(self, importances, feature_names, title="Feature Importance", top_k=20):
        """
        Plot feature importance
        
        Args:
            importances (array): Feature importance scores
            feature_names (list): Feature names
            title (str): Plot title
            top_k (int): Number of top features to show
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_k]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(indices)), importances[indices])
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_correlation(self, X, feature_names, title="Feature Correlation Matrix"):
        """
        Plot feature correlation matrix
        
        Args:
            X (array): Features
            feature_names (list): Feature names
            title (str): Plot title
        """
        # Calculate correlation matrix
        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def plot_dimensionality_reduction(self, X_original, X_reduced, y, method_name, feature_names=None):
        """
        Plot dimensionality reduction results
        
        Args:
            X_original (array): Original features
            X_reduced (array): Reduced features
            y (array): Target labels
            method_name (str): Name of reduction method
            feature_names (list): Original feature names
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original features (first 2 dimensions)
        if X_original.shape[1] >= 2:
            scatter = axes[0].scatter(X_original[:, 0], X_original[:, 1], c=y, cmap='viridis', alpha=0.7)
            axes[0].set_title('Original Features (First 2 Dimensions)')
            if feature_names and len(feature_names) >= 2:
                axes[0].set_xlabel(feature_names[0])
                axes[0].set_ylabel(feature_names[1])
            else:
                axes[0].set_xlabel('Feature 1')
                axes[0].set_ylabel('Feature 2')
        
        # Reduced features
        if X_reduced.shape[1] >= 2:
            scatter = axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.7)
            axes[1].set_title(f'{method_name} (First 2 Components)')
            axes[1].set_xlabel('Component 1')
            axes[1].set_ylabel('Component 2')
        
        # Add colorbar
        plt.colorbar(scatter, ax=axes, shrink=0.8)
        plt.tight_layout()
        plt.show()
    
    def comprehensive_feature_engineering(self, dataset_name='iris'):
        """
        Comprehensive feature engineering demonstration
        
        Args:
            dataset_name (str): Dataset to use
        
        Returns:
            dict: Complete feature engineering results
        """
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE FEATURE ENGINEERING - {dataset_name.upper()} DATASET")
        print(f"{'='*70}")
        
        # Load dataset
        X_original, y, feature_names, target_names = self.load_dataset(dataset_name)
        
        print(f"\nOriginal Dataset:")
        print(f"- Samples: {X_original.shape[0]}")
        print(f"- Features: {X_original.shape[1]}")
        print(f"- Classes: {len(target_names)}")
        
        # Evaluate original features
        print("\n1. Evaluating Original Features...")
        original_results = self.evaluate_feature_set(X_original, y, feature_names)
        
        print(f"Original feature performance:")
        for clf_name, metrics in original_results.items():
            if clf_name != 'feature_info':
                print(f"  {clf_name}: CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}, "
                      f"Test={metrics['test_accuracy']:.4f}")
        
        # 2. Create polynomial features
        print("\n2. Creating Polynomial Features...")
        X_poly, poly_names, poly_transformer = self.create_polynomial_features(X_original, degree=2)
        poly_results = self.evaluate_feature_set(X_poly, y, poly_names)
        
        print(f"Polynomial features: {X_poly.shape[1]} features")
        for clf_name, metrics in poly_results.items():
            if clf_name != 'feature_info':
                print(f"  {clf_name}: CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}, "
                      f"Test={metrics['test_accuracy']:.4f}")
        
        # 3. Create statistical features
        print("\n3. Creating Statistical Features...")
        X_stat, stat_names = self.create_statistical_features(X_original, feature_names)
        
        # Combine original and statistical features
        X_combined = np.hstack([X_original, X_stat])
        combined_names = list(feature_names) + stat_names
        
        combined_results = self.evaluate_feature_set(X_combined, y, combined_names)
        
        print(f"Combined features: {X_combined.shape[1]} features")
        for clf_name, metrics in combined_results.items():
            if clf_name != 'feature_info':
                print(f"  {clf_name}: CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}, "
                      f"Test={metrics['test_accuracy']:.4f}")
        
        # 4. Feature selection
        print("\n4. Feature Selection...")
        
        # Univariate selection
        X_univariate, univariate_selector, scores, selected_indices = self.feature_selection_univariate(
            X_combined, y, k=min(10, X_combined.shape[1]//2)
        )
        
        univariate_results = self.evaluate_feature_set(X_univariate, y)
        
        print(f"Univariate selection: {X_univariate.shape[1]} features")
        for clf_name, metrics in univariate_results.items():
            if clf_name != 'feature_info':
                print(f"  {clf_name}: CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}, "
                      f"Test={metrics['test_accuracy']:.4f}")
        
        # Feature importance selection
        X_importance, importances, importance_indices = self.feature_selection_importance(
            X_combined, y, threshold=0.01
        )
        
        if X_importance.shape[1] > 0:
            importance_results = self.evaluate_feature_set(X_importance, y)
            
            print(f"Importance-based selection: {X_importance.shape[1]} features")
            for clf_name, metrics in importance_results.items():
                if clf_name != 'feature_info':
                    print(f"  {clf_name}: CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}, "
                          f"Test={metrics['test_accuracy']:.4f}")
        
        # 5. Dimensionality reduction
        print("\n5. Dimensionality Reduction...")
        
        # Scale features for dimensionality reduction
        X_scaled, scaler = self.scale_features(X_combined, method='standard')
        
        # PCA
        X_pca, pca_model, explained_variance = self.dimensionality_reduction_pca(
            X_scaled, explained_variance_ratio=0.95
        )
        
        pca_results = self.evaluate_feature_set(X_pca, y)
        
        print(f"PCA: {X_pca.shape[1]} components (95% variance)")
        print(f"Explained variance ratio: {explained_variance[:5]}...")  # Show first 5
        for clf_name, metrics in pca_results.items():
            if clf_name != 'feature_info':
                print(f"  {clf_name}: CV={metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}, "
                      f"Test={metrics['test_accuracy']:.4f}")
        
        # 6. Visualization
        print("\n6. Creating Visualizations...")
        
        # Feature importance plot
        if len(importances) > 0:
            selected_names = [combined_names[i] for i in importance_indices]
            self.plot_feature_importance(importances[importance_indices], selected_names, 
                                       "Feature Importance (Random Forest)")
        
        # Correlation matrix (for original features)
        corr_matrix = self.plot_feature_correlation(X_original, feature_names, 
                                                   "Original Features Correlation")
        
        # Dimensionality reduction visualization
        if X_pca.shape[1] >= 2:
            self.plot_dimensionality_reduction(X_original, X_pca, y, "PCA", feature_names)
        
        # t-SNE for visualization
        if X_scaled.shape[0] <= 1000:  # t-SNE is slow for large datasets
            X_tsne, tsne_model = self.dimensionality_reduction_tsne(X_scaled)
            self.plot_dimensionality_reduction(X_original, X_tsne, y, "t-SNE", feature_names)
        
        # 7. Summary comparison
        print("\n7. Performance Summary:")
        print("=" * 80)
        
        all_results = {
            'Original': original_results,
            'Polynomial': poly_results,
            'Combined': combined_results,
            'Univariate': univariate_results,
            'PCA': pca_results
        }
        
        if X_importance.shape[1] > 0:
            all_results['Importance'] = importance_results
        
        # Create summary table
        summary_data = []
        for method_name, results in all_results.items():
            if 'Logistic Regression' in results:
                lr_metrics = results['Logistic Regression']
                rf_metrics = results['Random Forest']
                
                summary_data.append({
                    'Method': method_name,
                    'Features': results['feature_info']['n_features'],
                    'LR_CV': f"{lr_metrics['cv_mean']:.4f}±{lr_metrics['cv_std']:.4f}",
                    'LR_Test': f"{lr_metrics['test_accuracy']:.4f}",
                    'RF_CV': f"{rf_metrics['cv_mean']:.4f}±{rf_metrics['cv_std']:.4f}",
                    'RF_Test': f"{rf_metrics['test_accuracy']:.4f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Find best performing method
        best_method = None
        best_score = 0
        
        for method_name, results in all_results.items():
            if 'Logistic Regression' in results:
                avg_score = (results['Logistic Regression']['test_accuracy'] + 
                           results['Random Forest']['test_accuracy']) / 2
                if avg_score > best_score:
                    best_score = avg_score
                    best_method = method_name
        
        print(f"\nBest performing method: {best_method} (Average test accuracy: {best_score:.4f})")
        
        print("\n" + "="*70)
        print("FEATURE ENGINEERING ANALYSIS COMPLETE")
        print("="*70)
        
        return {
            'original': original_results,
            'polynomial': poly_results,
            'combined': combined_results,
            'univariate': univariate_results,
            'pca': pca_results,
            'importance': importance_results if X_importance.shape[1] > 0 else None,
            'summary': summary_df,
            'best_method': best_method,
            'transformers': self.transformers
        }

def demonstrate_feature_engineering():
    """
    Main demonstration function
    """
    print("FEATURE ENGINEERING AND REPRESENTATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize feature engineering toolkit
    fe = FeatureEngineering(random_state=42)
    
    # Comprehensive analysis on Iris dataset
    iris_results = fe.comprehensive_feature_engineering('iris')
    
    # Quick analysis on Wine dataset
    print("\n" + "="*60)
    print("QUICK ANALYSIS - WINE DATASET")
    print("="*60)
    
    wine_results = fe.comprehensive_feature_engineering('wine')
    
    return {
        'iris_analysis': iris_results,
        'wine_analysis': wine_results
    }

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_feature_engineering()