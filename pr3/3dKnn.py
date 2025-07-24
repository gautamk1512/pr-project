import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification

# --- Step 1: Generate 3D Classification Data (3 classes) ---
X, y = make_classification(
    n_samples=100,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# Map numeric labels to colors
colors = ['red', 'green', 'yellow']
label_colors = [colors[label] for label in y]

# --- Step 2: Define Euclidean Distance ---
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

# --- Step 3: KNN Prediction ---
def knn_predict(X_train, y_train, test_point, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], test_point)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

# --- Step 4: Define a test point and get prediction ---
test_point = [6, 2.5, 4]  # you can change this
k = 3
predicted_label = knn_predict(X, y, test_point, k)

# --- Step 5: Plot 3D ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot training points
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label_colors, s=50, alpha=0.6)

# Plot the test point
ax.scatter(test_point[0], test_point[1], test_point[2], color='black', s=100, marker='X', label='Test Point')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title(f'3D KNN Classification (k={k}, Predicted: {colors[predicted_label]})')
ax.legend()
plt.show()

# To install the required package, uncomment the following line:
# !pip install scikit-learn
