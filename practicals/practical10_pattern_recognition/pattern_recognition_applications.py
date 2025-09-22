"""
Practical 10: Applications of Pattern Recognition
- Speech and Speaker Recognition
- Character Recognition  
- Scene Analysis

This implementation demonstrates core concepts using basic libraries.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

class SpeechRecognition:
    """
    Speech Recognition using basic signal processing techniques.
    Simulates audio feature extraction and classification.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = None
        self.labels = None
        
    def simulate_audio_features(self, n_samples=1000):
        """
        Simulate audio features (MFCC-like features)
        In real implementation, this would extract features from audio files
        """
        print("Simulating audio feature extraction...")
        
        # Simulate different speech patterns
        features = []
        labels = []
        
        # Simulate 5 different speakers/words
        for speaker_id in range(5):
            for sample in range(n_samples // 5):
                # Generate synthetic features representing audio characteristics
                base_freq = 100 + speaker_id * 50  # Different base frequencies
                feature_vector = []
                
                # Simulate 13 MFCC coefficients
                for i in range(13):
                    coeff = np.random.normal(base_freq + i * 10, 20)
                    feature_vector.append(coeff)
                
                # Add spectral features
                spectral_centroid = np.random.normal(1000 + speaker_id * 200, 100)
                spectral_rolloff = np.random.normal(2000 + speaker_id * 300, 150)
                zero_crossing_rate = np.random.normal(0.1 + speaker_id * 0.02, 0.01)
                
                feature_vector.extend([spectral_centroid, spectral_rolloff, zero_crossing_rate])
                
                features.append(feature_vector)
                labels.append(f"Speaker_{speaker_id}")
        
        return np.array(features), np.array(labels)
    
    def train(self):
        """Train the speech recognition model"""
        print("Training Speech Recognition Model...")
        
        # Generate synthetic data
        self.features, self.labels = self.simulate_audio_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Speech Recognition Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def visualize_features(self):
        """Visualize speech features using PCA"""
        if self.features is None:
            print("No features available. Train the model first.")
            return
        
        print("Visualizing speech features...")
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.features)
        
        plt.figure(figsize=(10, 6))
        
        # Plot features by speaker
        unique_labels = np.unique(self.labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7)
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Speech Features Visualization (PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class CharacterRecognition:
    """
    Character Recognition using image processing and machine learning.
    """
    
    def __init__(self):
        self.model = SVC(kernel='rbf', random_state=42)
        self.features = None
        self.labels = None
        
    def generate_synthetic_characters(self, n_samples=1000):
        """
        Generate synthetic character images for demonstration
        """
        print("Generating synthetic character data...")
        
        features = []
        labels = []
        
        # Generate synthetic data for digits 0-9
        for digit in range(10):
            for sample in range(n_samples // 10):
                # Create a 28x28 image (like MNIST)
                img = np.zeros((28, 28))
                
                # Add some pattern based on digit
                center_x, center_y = 14, 14
                
                if digit == 0:  # Circle
                    y, x = np.ogrid[:28, :28]
                    mask = (x - center_x)**2 + (y - center_y)**2 <= 100
                    img[mask] = 1
                    inner_mask = (x - center_x)**2 + (y - center_y)**2 <= 50
                    img[inner_mask] = 0
                    
                elif digit == 1:  # Vertical line
                    img[:, 12:16] = 1
                    
                elif digit == 2:  # Horizontal lines
                    img[8:12, :] = 1
                    img[16:20, :] = 1
                    
                # Add noise
                noise = np.random.normal(0, 0.1, (28, 28))
                img = np.clip(img + noise, 0, 1)
                
                # Extract features (flatten + some basic features)
                flat_features = img.flatten()
                
                # Add some geometric features
                moments = cv2.moments(img.astype(np.uint8))
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    cx, cy = 14, 14
                
                geometric_features = [cx, cy, np.sum(img), np.std(img)]
                
                # Combine features (use subset to reduce dimensionality)
                combined_features = np.concatenate([
                    flat_features[::4],  # Downsample pixel features
                    geometric_features
                ])
                
                features.append(combined_features)
                labels.append(digit)
        
        return np.array(features), np.array(labels)
    
    def train(self):
        """Train the character recognition model"""
        print("Training Character Recognition Model...")
        
        # Generate synthetic data
        self.features, self.labels = self.generate_synthetic_characters()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Character Recognition Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def visualize_characters(self):
        """Visualize character features"""
        if self.features is None:
            print("No features available. Train the model first.")
            return
        
        print("Visualizing character features...")
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.features)
        
        plt.figure(figsize=(10, 6))
        
        # Plot features by digit
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for digit in range(10):
            mask = self.labels == digit
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[digit]], label=f'Digit {digit}', alpha=0.7)
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Character Features Visualization (PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class SceneAnalysis:
    """
    Scene Analysis using computer vision techniques.
    """
    
    def __init__(self):
        self.model = KMeans(n_clusters=5, random_state=42)
        self.features = None
        self.scene_types = ['Indoor', 'Outdoor', 'Urban', 'Nature', 'Vehicle']
        
    def extract_scene_features(self, n_samples=500):
        """
        Extract features from synthetic scene images
        """
        print("Extracting scene features...")
        
        features = []
        labels = []
        
        for scene_idx, scene_type in enumerate(self.scene_types):
            for sample in range(n_samples // len(self.scene_types)):
                # Generate synthetic scene features
                img_features = []
                
                if scene_type == 'Indoor':
                    # Indoor scenes: more rectangular shapes, artificial lighting
                    edge_density = np.random.normal(0.3, 0.1)
                    color_variance = np.random.normal(0.2, 0.05)
                    texture_complexity = np.random.normal(0.4, 0.1)
                    
                elif scene_type == 'Outdoor':
                    # Outdoor scenes: more natural textures, varied lighting
                    edge_density = np.random.normal(0.5, 0.1)
                    color_variance = np.random.normal(0.4, 0.1)
                    texture_complexity = np.random.normal(0.6, 0.1)
                    
                elif scene_type == 'Urban':
                    # Urban scenes: high edge density, geometric patterns
                    edge_density = np.random.normal(0.7, 0.1)
                    color_variance = np.random.normal(0.3, 0.05)
                    texture_complexity = np.random.normal(0.5, 0.1)
                    
                elif scene_type == 'Nature':
                    # Nature scenes: organic textures, green dominance
                    edge_density = np.random.normal(0.4, 0.1)
                    color_variance = np.random.normal(0.6, 0.1)
                    texture_complexity = np.random.normal(0.8, 0.1)
                    
                else:  # Vehicle
                    # Vehicle scenes: metallic textures, specific shapes
                    edge_density = np.random.normal(0.6, 0.1)
                    color_variance = np.random.normal(0.25, 0.05)
                    texture_complexity = np.random.normal(0.3, 0.1)
                
                # Add color histogram features (simplified)
                color_hist = np.random.dirichlet(np.ones(8) * (scene_idx + 1))
                
                # Combine all features
                feature_vector = [
                    edge_density, color_variance, texture_complexity
                ] + color_hist.tolist()
                
                # Add some spatial features
                spatial_features = np.random.normal(scene_idx * 0.2, 0.1, 5)
                feature_vector.extend(spatial_features)
                
                features.append(feature_vector)
                labels.append(scene_type)
        
        return np.array(features), np.array(labels)
    
    def analyze_scenes(self):
        """Perform scene analysis"""
        print("Performing Scene Analysis...")
        
        # Generate synthetic data
        self.features, self.labels = self.extract_scene_features()
        
        # Perform clustering
        clusters = self.model.fit_predict(self.features)
        
        # Analyze clustering results
        print("\nScene Clustering Results:")
        for i, scene_type in enumerate(self.scene_types):
            scene_mask = np.array(self.labels) == scene_type
            scene_clusters = clusters[scene_mask]
            dominant_cluster = np.bincount(scene_clusters).argmax()
            purity = np.sum(scene_clusters == dominant_cluster) / len(scene_clusters)
            print(f"{scene_type}: Dominant Cluster {dominant_cluster}, Purity: {purity:.3f}")
        
        return clusters
    
    def visualize_scenes(self):
        """Visualize scene analysis results"""
        if self.features is None:
            print("No features available. Analyze scenes first.")
            return
        
        print("Visualizing scene analysis...")
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.features)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Original scene types
        plt.subplot(1, 2, 1)
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.scene_types)))
        
        for i, scene_type in enumerate(self.scene_types):
            mask = np.array(self.labels) == scene_type
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=scene_type, alpha=0.7)
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Scene Types (Ground Truth)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Clustering results
        plt.subplot(1, 2, 2)
        clusters = self.model.labels_
        cluster_colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(clusters))))
        
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[cluster_colors[cluster_id]], label=f'Cluster {cluster_id}', alpha=0.7)
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Scene Clustering Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to run all pattern recognition applications
    """
    print("=" * 60)
    print("PRACTICAL 10: APPLICATIONS OF PATTERN RECOGNITION")
    print("=" * 60)
    
    # 1. Speech and Speaker Recognition
    print("\n1. SPEECH AND SPEAKER RECOGNITION")
    print("-" * 40)
    speech_recognizer = SpeechRecognition()
    speech_accuracy = speech_recognizer.train()
    speech_recognizer.visualize_features()
    
    # 2. Character Recognition
    print("\n2. CHARACTER RECOGNITION")
    print("-" * 40)
    char_recognizer = CharacterRecognition()
    char_accuracy = char_recognizer.train()
    char_recognizer.visualize_characters()
    
    # 3. Scene Analysis
    print("\n3. SCENE ANALYSIS")
    print("-" * 40)
    scene_analyzer = SceneAnalysis()
    scene_analyzer.analyze_scenes()
    scene_analyzer.visualize_scenes()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"Speech Recognition Accuracy: {speech_accuracy:.3f}")
    print(f"Character Recognition Accuracy: {char_accuracy:.3f}")
    print("Scene Analysis: Clustering completed successfully")
    print("\nAll pattern recognition applications demonstrated successfully!")
    print("Check the generated plots for visual analysis.")

if __name__ == "__main__":
    main()