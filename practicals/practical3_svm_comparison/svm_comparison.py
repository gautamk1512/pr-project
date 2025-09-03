import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.datasets import make_classification, load_iris, load_wine, load_breast_cancer
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class SVMComparison:
    """
    Comprehensive SVM Classification Accuracy Comparison
    
    This class provides tools to compare different SVM configurations:
    - Different kernels (linear, polynomial, RBF, sigmoid)
    - Various hyperparameters (C, gamma, degree)
    - Multiple datasets
    - Cross-validation and performance metrics
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.scalers = {}
        
    def load_dataset(self, dataset_name='iris'):
        """
        Load various datasets for comparison
        
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
                n_redundant=10, n_clusters_per_class=1, random_state=self.random_state
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            target_names = [f'class_{i}' for i in range(len(np.unique(y)))]
            return X, y, feature_names, target_names
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def compare_kernels(self, X, y, test_size=0.2, cv_folds=5):
        """
        Compare different SVM kernels
        
        Args:
            X (array): Feature matrix
            y (array): Target vector
            test_size (float): Test set proportion
            cv_folds (int): Cross-validation folds
        
        Returns:
            dict: Results for each kernel
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define kernels to compare
        kernels = {
            'Linear': {'kernel': 'linear', 'C': 1.0},
            'Polynomial (degree=2)': {'kernel': 'poly', 'degree': 2, 'C': 1.0},
            'Polynomial (degree=3)': {'kernel': 'poly', 'degree': 3, 'C': 1.0},
            'RBF (gamma=scale)': {'kernel': 'rbf', 'gamma': 'scale', 'C': 1.0},
            'RBF (gamma=auto)': {'kernel': 'rbf', 'gamma': 'auto', 'C': 1.0},
            'Sigmoid': {'kernel': 'sigmoid', 'C': 1.0}
        }
        
        results = {}
        
        for name, params in kernels.items():
            print(f"Training SVM with {name} kernel...")
            
            # Create and train model
            model = SVC(random_state=self.random_state, **params)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
            
            # Store results
            results[name] = {
                'model': model,
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'train_pred': train_pred,
                'test_pred': test_pred,
                'confusion_matrix': confusion_matrix(y_test, test_pred),
                'classification_report': classification_report(y_test, test_pred, output_dict=True)
            }
        
        self.results['kernel_comparison'] = results
        self.scalers['kernel_comparison'] = scaler
        
        return results
    
    def hyperparameter_tuning(self, X, y, kernel='rbf', test_size=0.2, cv_folds=5):
        """
        Perform hyperparameter tuning for SVM
        
        Args:
            X (array): Feature matrix
            y (array): Target vector
            kernel (str): Kernel type to tune
            test_size (float): Test set proportion
            cv_folds (int): Cross-validation folds
        
        Returns:
            dict: Tuning results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define parameter grids for different kernels
        param_grids = {
            'linear': {
                'C': [0.1, 1, 10, 100]
            },
            'poly': {
                'C': [0.1, 1, 10, 100],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            'rbf': {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
            },
            'sigmoid': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        }
        
        print(f"Performing hyperparameter tuning for {kernel} kernel...")
        
        # Grid search
        svm = SVC(kernel=kernel, random_state=self.random_state)
        grid_search = GridSearchCV(
            svm, param_grids[kernel], cv=cv_folds, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model predictions
        best_model = grid_search.best_estimator_
        train_pred = best_model.predict(X_train_scaled)
        test_pred = best_model.predict(X_test_scaled)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': best_model,
            'grid_search': grid_search,
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'classification_report': classification_report(y_test, test_pred, output_dict=True)
        }
        
        self.results[f'{kernel}_tuning'] = results
        self.scalers[f'{kernel}_tuning'] = scaler
        
        return results
    
    def validation_curves_analysis(self, X, y, kernel='rbf', param_name='C', param_range=None):
        """
        Generate validation curves for hyperparameter analysis
        
        Args:
            X (array): Feature matrix
            y (array): Target vector
            kernel (str): Kernel type
            param_name (str): Parameter to analyze
            param_range (list): Range of parameter values
        
        Returns:
            dict: Validation curve results
        """
        if param_range is None:
            if param_name == 'C':
                param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            elif param_name == 'gamma':
                param_range = [0.001, 0.01, 0.1, 1, 10, 100]
            elif param_name == 'degree':
                param_range = [1, 2, 3, 4, 5, 6]
            else:
                param_range = [0.1, 1, 10]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create SVM model
        svm = SVC(kernel=kernel, random_state=self.random_state)
        
        # Generate validation curves
        train_scores, val_scores = validation_curve(
            svm, X_scaled, y, param_name=param_name, param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        results = {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_mean': train_scores.mean(axis=1),
            'train_std': train_scores.std(axis=1),
            'val_mean': val_scores.mean(axis=1),
            'val_std': val_scores.std(axis=1)
        }
        
        return results
    
    def plot_kernel_comparison(self, results):
        """
        Plot comparison of different kernels
        
        Args:
            results (dict): Results from compare_kernels method
        """
        kernels = list(results.keys())
        train_acc = [results[k]['train_accuracy'] for k in kernels]
        test_acc = [results[k]['test_accuracy'] for k in kernels]
        cv_mean = [results[k]['cv_mean'] for k in kernels]
        cv_std = [results[k]['cv_std'] for k in kernels]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        x = np.arange(len(kernels))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_acc, width, label='Train Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_acc, width, label='Test Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Kernel Type')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Train vs Test Accuracy by Kernel')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(kernels, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cross-validation scores
        axes[0, 1].errorbar(x, cv_mean, yerr=cv_std, fmt='o-', capsize=5, capthick=2)
        axes[0, 1].set_xlabel('Kernel Type')
        axes[0, 1].set_ylabel('CV Accuracy')
        axes[0, 1].set_title('Cross-Validation Accuracy by Kernel')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(kernels, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion matrix for best performing kernel
        best_kernel = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        cm = results[best_kernel]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_kernel}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Performance metrics comparison
        metrics = ['precision', 'recall', 'f1-score']
        metric_data = {}
        
        for metric in metrics:
            metric_data[metric] = []
            for kernel in kernels:
                # Get weighted average of the metric
                report = results[kernel]['classification_report']
                metric_data[metric].append(report['weighted avg'][metric])
        
        x_metrics = np.arange(len(metrics))
        width = 0.1
        
        for i, kernel in enumerate(kernels):
            values = [metric_data[metric][i] for metric in metrics]
            axes[1, 1].bar(x_metrics + i*width, values, width, label=kernel, alpha=0.8)
        
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_xticks(x_metrics + width * (len(kernels)-1) / 2)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary table
        print("\nKernel Comparison Summary:")
        print("=" * 80)
        print(f"{'Kernel':<20} {'Train Acc':<10} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<10}")
        print("=" * 80)
        
        for kernel in kernels:
            print(f"{kernel:<20} {train_acc[kernels.index(kernel)]:<10.4f} "
                  f"{test_acc[kernels.index(kernel)]:<10.4f} "
                  f"{cv_mean[kernels.index(kernel)]:<10.4f} "
                  f"{cv_std[kernels.index(kernel)]:<10.4f}")
        
        print("\nBest performing kernel:", best_kernel)
        print(f"Best test accuracy: {results[best_kernel]['test_accuracy']:.4f}")
    
    def plot_validation_curves(self, validation_results, title_suffix=""):
        """
        Plot validation curves
        
        Args:
            validation_results (dict): Results from validation_curves_analysis
            title_suffix (str): Additional title text
        """
        param_range = validation_results['param_range']
        train_mean = validation_results['train_mean']
        train_std = validation_results['train_std']
        val_mean = validation_results['val_mean']
        val_std = validation_results['val_std']
        param_name = validation_results['param_name']
        
        plt.figure(figsize=(10, 6))
        
        # Plot training scores
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        # Plot validation scores
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='red')
        
        plt.xlabel(f'{param_name.upper()}')
        plt.ylabel('Accuracy Score')
        plt.title(f'Validation Curve - {param_name.upper()} {title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if param_name in ['C', 'gamma']:
            plt.xscale('log')
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal parameter
        optimal_idx = np.argmax(val_mean)
        optimal_param = param_range[optimal_idx]
        optimal_score = val_mean[optimal_idx]
        
        print(f"\nOptimal {param_name}: {optimal_param}")
        print(f"Optimal validation score: {optimal_score:.4f}")
    
    def comprehensive_analysis(self, dataset_name='iris'):
        """
        Perform comprehensive SVM analysis on a dataset
        
        Args:
            dataset_name (str): Dataset to analyze
        
        Returns:
            dict: Complete analysis results
        """
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE SVM ANALYSIS - {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Load dataset
        X, y, feature_names, target_names = self.load_dataset(dataset_name)
        
        print(f"\nDataset Information:")
        print(f"- Samples: {X.shape[0]}")
        print(f"- Features: {X.shape[1]}")
        print(f"- Classes: {len(target_names)}")
        print(f"- Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # 1. Kernel Comparison
        print("\n1. Comparing Different Kernels...")
        kernel_results = self.compare_kernels(X, y)
        self.plot_kernel_comparison(kernel_results)
        
        # 2. Hyperparameter Tuning for RBF kernel
        print("\n2. Hyperparameter Tuning for RBF Kernel...")
        rbf_tuning = self.hyperparameter_tuning(X, y, kernel='rbf')
        
        print(f"Best parameters: {rbf_tuning['best_params']}")
        print(f"Best CV score: {rbf_tuning['best_score']:.4f}")
        print(f"Test accuracy: {rbf_tuning['test_accuracy']:.4f}")
        
        # 3. Validation Curves
        print("\n3. Validation Curve Analysis...")
        
        # C parameter validation curve
        c_validation = self.validation_curves_analysis(X, y, kernel='rbf', param_name='C')
        self.plot_validation_curves(c_validation, "(RBF Kernel)")
        
        # Gamma parameter validation curve
        gamma_validation = self.validation_curves_analysis(X, y, kernel='rbf', param_name='gamma')
        self.plot_validation_curves(gamma_validation, "(RBF Kernel)")
        
        # 4. Final Model Evaluation
        print("\n4. Final Model Evaluation...")
        best_model = rbf_tuning['best_model']
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y, best_model.predict(self.scalers['rbf_tuning'].transform(X)), 
                                  target_names=target_names))
        
        # Store all results
        analysis_results = {
            'dataset_info': {
                'name': dataset_name,
                'samples': X.shape[0],
                'features': X.shape[1],
                'classes': len(target_names),
                'feature_names': feature_names,
                'target_names': target_names
            },
            'kernel_comparison': kernel_results,
            'hyperparameter_tuning': rbf_tuning,
            'validation_curves': {
                'C': c_validation,
                'gamma': gamma_validation
            }
        }
        
        return analysis_results

def compare_multiple_datasets():
    """
    Compare SVM performance across multiple datasets
    """
    print("\n" + "="*70)
    print("MULTI-DATASET SVM COMPARISON")
    print("="*70)
    
    datasets = ['iris', 'wine', 'breast_cancer', 'synthetic']
    svm_comp = SVMComparison()
    
    results_summary = []
    
    for dataset in datasets:
        print(f"\nAnalyzing {dataset} dataset...")
        
        try:
            X, y, _, _ = svm_comp.load_dataset(dataset)
            kernel_results = svm_comp.compare_kernels(X, y)
            
            # Find best kernel
            best_kernel = max(kernel_results.keys(), 
                            key=lambda k: kernel_results[k]['test_accuracy'])
            best_accuracy = kernel_results[best_kernel]['test_accuracy']
            
            results_summary.append({
                'Dataset': dataset,
                'Best Kernel': best_kernel,
                'Best Accuracy': best_accuracy,
                'Samples': X.shape[0],
                'Features': X.shape[1],
                'Classes': len(np.unique(y))
            })
            
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_summary)
    
    print("\nMulti-Dataset Comparison Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Plot summary
    plt.figure(figsize=(12, 8))
    
    # Accuracy by dataset
    plt.subplot(2, 2, 1)
    plt.bar(summary_df['Dataset'], summary_df['Best Accuracy'], alpha=0.7)
    plt.title('Best SVM Accuracy by Dataset')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Dataset characteristics
    plt.subplot(2, 2, 2)
    plt.scatter(summary_df['Samples'], summary_df['Best Accuracy'], 
               s=summary_df['Features']*10, alpha=0.7)
    plt.xlabel('Number of Samples')
    plt.ylabel('Best Accuracy')
    plt.title('Accuracy vs Dataset Size\n(Bubble size = Features)')
    plt.grid(True, alpha=0.3)
    
    # Kernel distribution
    plt.subplot(2, 2, 3)
    kernel_counts = summary_df['Best Kernel'].value_counts()
    plt.pie(kernel_counts.values, labels=kernel_counts.index, autopct='%1.1f%%')
    plt.title('Best Kernel Distribution')
    
    # Features vs Classes
    plt.subplot(2, 2, 4)
    plt.scatter(summary_df['Features'], summary_df['Classes'], 
               c=summary_df['Best Accuracy'], cmap='viridis', s=100)
    plt.xlabel('Number of Features')
    plt.ylabel('Number of Classes')
    plt.title('Dataset Complexity\n(Color = Accuracy)')
    plt.colorbar(label='Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return summary_df

def demonstrate_svm_comparison():
    """
    Main demonstration function
    """
    print("SVM CLASSIFICATION ACCURACY COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Initialize SVM comparison
    svm_comp = SVMComparison()
    
    # Comprehensive analysis on Iris dataset
    iris_results = svm_comp.comprehensive_analysis('iris')
    
    # Multi-dataset comparison
    multi_dataset_summary = compare_multiple_datasets()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    
    return {
        'iris_analysis': iris_results,
        'multi_dataset_summary': multi_dataset_summary
    }

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_svm_comparison()