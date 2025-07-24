import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Generate sample 3-class dataset
X, y = make_classification(n_samples=100, n_features=3, n_informative=3,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=42)

# Fit KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Create test point
test_point = np.array([[0, 0, 0]])
probs = knn.predict_proba(test_point)[0]

# Get nearest neighbors
distances, indices = knn.kneighbors(test_point)
nearest_points = X[indices[0]]
nearest_labels = y[indices[0]]

# Colors and labels
colors = ['red', 'green', 'yellow']
class_names = ['Class 0', 'Class 1', 'Class 2']

# Create 3D scatter plot with subplot for bar chart
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'bar'}]],
    subplot_titles=("3D KNN Visualization", "Prediction Probabilities")
)

# Training points
for class_value in np.unique(y):
    mask = y == class_value
    fig.add_trace(go.Scatter3d(
        x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
        mode='markers',
        marker=dict(size=5, color=colors[class_value]),
        name=f'Train Class {class_value}'
    ), row=1, col=1)

# Test point
fig.add_trace(go.Scatter3d(
    x=test_point[:, 0], y=test_point[:, 1], z=test_point[:, 2],
    mode='markers+text',
    marker=dict(size=10, color='blue', symbol='diamond'),
    name='Test Point',
    text=['Test Point'],
    textposition="top center"
), row=1, col=1)

# Nearest Neighbors
fig.add_trace(go.Scatter3d(
    x=nearest_points[:, 0], y=nearest_points[:, 1], z=nearest_points[:, 2],
    mode='markers',
    marker=dict(size=7, color='black'),
    name='k-NN (k=3)'
), row=1, col=1)

# Probability Bar Chart
fig.add_trace(go.Bar(
    x=class_names,
    y=probs,
    marker=dict(color=colors),
    name='Prediction Probability'
), row=1, col=2)

# Update layout with user's name
fig.update_layout(
    height=600, width=1000,
    title=dict(text="KNN Classification - Gautam Singh", x=0.5, font=dict(size=24)),
    showlegend=True
)

fig.show()
