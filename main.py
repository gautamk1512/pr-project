import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For data visualization

# Sample training data (x, y)
# X represents input features, Y represents target values
# This is a simple linear relationship where y = 2x
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

# Initialize model parameters
m = 0   # slope (weight)
c = 0   # intercept (bias)

# Hyperparameters for gradient descent
L = 0.01  # Learning rate - controls how fast the model learns
epochs = 1000  # Number of iterations for training

n = float(len(X))  # Number of training examples

# Gradient Descent Loop
# This is the main training loop that updates the parameters
for i in range(epochs):
    # Forward pass: Make predictions using current parameters
    Y_pred = m * X + c  # Linear equation: y = mx + c
    
    # Calculate error (difference between predictions and actual values)
    error = Y_pred - Y
    
    # Compute gradients (partial derivatives)
    # Gradient of loss with respect to m (slope)
    D_m = (2/n) * sum(X * error)
    # Gradient of loss with respect to c (intercept)
    D_c = (2/n) * sum(error)
    
    # Update parameters using gradient descent
    # Move in the opposite direction of the gradient
    m = m - L * D_m  # Update slope
    c = c - L * D_c  # Update intercept

    # Print progress every 100 epochs
    if i % 100 == 0:
        print(f"Epoch {i}: m = {m:.4f}, c = {c:.4f}, Loss = {np.mean(error**2):.4f}")

# Print final model parameters
print("\nFinal model: y = {:.2f}x + {:.2f}".format(m, c))

# Visualization
plt.scatter(X, Y, color='blue', label='Original Data')  # Plot original data points
plt.plot(X, m*X + c, color='red', label='Fitted Line')  # Plot the fitted line
plt.legend()  # Show legend
plt.xlabel('X')  # X-axis label
plt.ylabel('Y')  # Y-axis label
plt.title('Linear Regression using Gradient Descent')  # Plot title
plt.show()  # Display the plot
