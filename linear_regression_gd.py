import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_data(n_samples=100):
    np.random.seed(0)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    return X, y

# Linear Regression using Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.theta = None

    def fit(self, X, y):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]  # add bias term
        self.theta = np.zeros((X_b.shape[1], 1))
        for iteration in range(self.n_iters):
            gradients = 2/m * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

if __name__ == "__main__":
    # Generate data
    X, y = generate_data(100)

    # Train model
    model = LinearRegressionGD(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Plot results
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", label="Prediction")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression using Gradient Descent")
    plt.legend()
    plt.show() 