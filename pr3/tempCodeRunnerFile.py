import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.colors import ListedColormap

# ---------- Euclidean Distance ----------
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# ---------- KNN Prediction ----------
def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i], training_data[i]))
    distances.sort(key=lambda x: x[0])
    
    k_nearest = distances[:k]
    k_nearest_labels = [label for _, label, _ in k_nearest]
    prediction = Counter(k_nearest_labels).most_common(1)[0][0]
    
    return prediction, [point for _, _, point in k_nearest]

# ---------- Data ----------
training_data = [
    [1, 2], [2, 1], [1.5, 1.8],   # Class A
    [5, 5], [6, 6], [5.5, 5.2],   # Class B
    [8, 1], [9, 2], [8.5, 1.5]    # Class C
]
training_labels = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
test_point = [4, 3]
k = 3

# ---------- Colors and Markers ----------
colors = {'A': 'red', 'B': 'green', 'C': 'blue'}
markers = {'A': 'o', 'B': 's', 'C': '^'}

# ---------- GRAPH 1: Basic Scatter Plot ----------
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
for point, label in zip(training_data, training_labels):
    plt.scatter(point[0], point[1], color=colors[label], marker=markers[label], label=label)
plt.scatter(test_point[0], test_point[1], color='black', marker='x', s=100, label='Test Point')
plt.title('Graph 1: Basic Class Scatter')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend(loc='upper left')

# ---------- GRAPH 2: Highlight Nearest Neighbors ----------
predicted_label, nearest_points = knn_predict(training_data, training_labels, test_point, k)

plt.subplot(1, 3, 2)
for point, label in zip(training_data, training_labels):
    plt.scatter(point[0], point[1], color=colors[label], marker=markers[label])
plt.scatter(test_point[0], test_point[1], color='black', marker='x', s=100)
for point in nearest_points:
    plt.scatter(point[0], point[1], edgecolors='black', facecolors='none', s=200, linewidths=2)
plt.title(f'Graph 2: Nearest Neighbors (Predicted: {predicted_label})')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

# ---------- GRAPH 3: Decision Boundaries ----------
h = 0.1
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict class for each grid point
Z = []
for i in range(xx.shape[0]):
    row = []
    for j in range(xx.shape[1]):
        p = [xx[i][j], yy[i][j]]
        label, _ = knn_predict(training_data, training_labels, p, k)
        row.append(label)
    Z.append(row)

Z = np.array(Z)
label_to_int = {'A': 0, 'B': 1, 'C': 2}
Z_int = np.vectorize(label_to_int.get)(Z)

plt.subplot(1, 3, 3)
cmap_background = ListedColormap(['#ffcccc', '#ccffcc', '#ccccff'])
plt.contourf(xx, yy, Z_int, cmap=cmap_background, alpha=0.5)

# Overlay data points
for point, label in zip(training_data, training_labels):
    plt.scatter(point[0], point[1], color=colors[label], marker=markers[label])
plt.scatter(test_point[0], test_point[1], color='black', marker='x', s=100)
plt.title('Graph 3: Decision Regions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

plt.tight_layout()
plt.show()
