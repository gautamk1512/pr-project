import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Spacing
step = 0.1
points_1d = np.arange(0, 1, step)
points_2d = np.arange(0, 1, step)
points_3d = np.arange(0, 1, step)

# 1D points
fig, ax = plt.subplots(figsize=(8, 1))
ax.scatter(points_1d, np.zeros_like(points_1d), color='blue')
ax.set_title("1D: Points spaced every 0.1 in [0,1]")
ax.set_xlim(-0.1, 1.1)
ax.set_yticks([])
plt.show()

# 2D points grid
X, Y = np.meshgrid(points_2d, points_2d)
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X, Y, color='green')
ax.set_title("2D: Grid points spaced every 0.1 in [0,1] x [0,1]")
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
plt.show()

# 3D points grid
X3, Y3, Z3 = np.meshgrid(points_3d, points_3d, points_3d)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X3.flatten(), Y3.flatten(), Z3.flatten(), color='red', s=10)
ax.set_title("3D: Grid points spaced every 0.1 in [0,1]^3")
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_zlim(-0.1, 1.1)
plt.show()
