import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit

# Sample 2D data points
data = np.array([
    [1, 1],
    [2, 5],
    [3, 3],
    [6, 7],
    [7, 2],
    [8, 8]
])

def kernel_density_estimate(x, y, data, h):
    """
    Compute the Kernel Density Estimate on a grid defined by x, y.
    Parameters:
        x, y: 2D grid coordinates (from meshgrid)
        data: array of data points
        h: bandwidth (smoothing parameter)
    Returns:
        z: KDE values on the grid
    """
    z = np.zeros_like(x)
    for (x_i, y_i) in data:
        # Gaussian kernel applied for each data point
        z += np.exp(-((x - x_i)**2 + (y - y_i)**2) / (2 * h**2))
    # Normalize the result
    z /= (len(data) * 2 * np.pi * h**2)
    return z

# Create a grid over which to evaluate the KDE
x = np.linspace(0, 10, 50)
y = np.linspace(0, 8, 40)
X, Y = np.meshgrid(x, y)  # Create 2D grid arrays

# Different bandwidths to show effect on smoothing
bandwidths = [1, 0.5, 0.2]

# Create figure for plotting
fig = plt.figure(figsize=(12, 8))

# Loop over each bandwidth, create subplot, and plot KDE surface
for i, h in enumerate(bandwidths, 1):
    ax = fig.add_subplot(2, 2, i, projection='3d')  # 3D subplot
    Z = kernel_density_estimate(X, Y, data, h)     # Compute KDE
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')  # 3D surface plot
    ax.set_title(f'Bandwidth h = {h}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('p(x)')

plt.tight_layout()  # Adjust layout so subplots don't overlap
plt.show()          # Display the figure
