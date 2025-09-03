import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GradientDescent:
    """
    Implementation of Gradient Descent Algorithm
    
    This class implements gradient descent for finding the minimum of a function.
    It supports both single-variable and multi-variable functions.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the Gradient Descent optimizer
        
        Args:
            learning_rate (float): Step size for each iteration
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []
        
    def optimize(self, function, gradient_function, initial_point):
        """
        Perform gradient descent optimization
        
        Args:
            function: The function to minimize
            gradient_function: Function that computes the gradient
            initial_point: Starting point for optimization
            
        Returns:
            tuple: (optimal_point, optimal_value, convergence_info)
        """
        current_point = np.array(initial_point, dtype=float)
        self.history = [current_point.copy()]
        
        for iteration in range(self.max_iterations):
            # Compute gradient at current point
            gradient = gradient_function(current_point)
            
            # Update point using gradient descent rule
            new_point = current_point - self.learning_rate * gradient
            
            # Check for convergence
            if np.linalg.norm(new_point - current_point) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            current_point = new_point
            self.history.append(current_point.copy())
        
        optimal_value = function(current_point)
        convergence_info = {
            'iterations': len(self.history),
            'converged': iteration < self.max_iterations - 1,
            'final_gradient_norm': np.linalg.norm(gradient)
        }
        
        return current_point, optimal_value, convergence_info
    
    def plot_convergence_1d(self, function, x_range=(-10, 10), num_points=1000):
        """
        Plot the convergence for 1D functions
        """
        if len(self.history) == 0:
            print("No optimization history available. Run optimize() first.")
            return
            
        # Create x values for plotting the function
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = [function(np.array([xi])) for xi in x]
        
        # Plot function and optimization path
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Function and optimization path
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'b-', label='Function', linewidth=2)
        
        # Plot optimization path
        history_x = [point[0] for point in self.history]
        history_y = [function(point) for point in self.history]
        
        plt.plot(history_x, history_y, 'ro-', label='Optimization Path', markersize=4)
        plt.plot(history_x[0], history_y[0], 'go', markersize=8, label='Start')
        plt.plot(history_x[-1], history_y[-1], 'rs', markersize=8, label='End')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gradient Descent Optimization Path')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Convergence curve
        plt.subplot(1, 2, 2)
        plt.plot(range(len(history_y)), history_y, 'b-o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title('Convergence Curve')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence_2d(self, function, x_range=(-5, 5), y_range=(-5, 5), num_points=100):
        """
        Plot the convergence for 2D functions
        """
        if len(self.history) == 0:
            print("No optimization history available. Run optimize() first.")
            return
            
        # Create meshgrid for contour plot
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = function(np.array([X[i, j], Y[i, j]]))
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Contour plot with optimization path
        plt.subplot(1, 3, 1)
        contour = plt.contour(X, Y, Z, levels=20, alpha=0.6)
        plt.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
        plt.colorbar(label='Function Value')
        
        # Plot optimization path
        history_x = [point[0] for point in self.history]
        history_y = [point[1] for point in self.history]
        
        plt.plot(history_x, history_y, 'ro-', linewidth=2, markersize=4, label='Path')
        plt.plot(history_x[0], history_y[0], 'go', markersize=8, label='Start')
        plt.plot(history_x[-1], history_y[-1], 'rs', markersize=8, label='End')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('2D Optimization Path')
        plt.legend()
        
        # Plot 2: 3D surface
        ax = plt.subplot(1, 3, 2, projection='3d')
        ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
        
        # Plot optimization path in 3D
        history_z = [function(point) for point in self.history]
        ax.plot(history_x, history_y, history_z, 'ro-', linewidth=2, markersize=4)
        ax.scatter(history_x[0], history_y[0], history_z[0], color='green', s=100, label='Start')
        ax.scatter(history_x[-1], history_y[-1], history_z[-1], color='red', s=100, label='End')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        ax.set_title('3D Optimization')
        
        # Plot 3: Convergence curve
        plt.subplot(1, 3, 3)
        plt.plot(range(len(history_z)), history_z, 'b-o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.title('Convergence Curve')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example functions and their gradients
def quadratic_1d(x):
    """Simple quadratic function: f(x) = (x-3)^2 + 2"""
    return (x[0] - 3)**2 + 2

def quadratic_1d_gradient(x):
    """Gradient of quadratic function"""
    return np.array([2 * (x[0] - 3)])

def rosenbrock_2d(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_2d_gradient(x):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

def sphere_2d(x):
    """Sphere function: f(x,y) = x^2 + y^2"""
    return x[0]**2 + x[1]**2

def sphere_2d_gradient(x):
    """Gradient of sphere function"""
    return 2 * x

def beale_function(x):
    """Beale function: complex 2D optimization problem"""
    term1 = (1.5 - x[0] + x[0]*x[1])**2
    term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3

def beale_gradient(x):
    """Gradient of Beale function"""
    term1 = 1.5 - x[0] + x[0]*x[1]
    term2 = 2.25 - x[0] + x[0]*x[1]**2
    term3 = 2.625 - x[0] + x[0]*x[1]**3
    
    dx = 2*term1*(-1 + x[1]) + 2*term2*(-1 + x[1]**2) + 2*term3*(-1 + x[1]**3)
    dy = 2*term1*x[0] + 2*term2*x[0]*2*x[1] + 2*term3*x[0]*3*x[1]**2
    
    return np.array([dx, dy])

def demonstrate_gradient_descent():
    """
    Demonstrate gradient descent on various functions
    """
    print("=" * 60)
    print("GRADIENT DESCENT ALGORITHM DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: 1D Quadratic Function
    print("\n1. 1D Quadratic Function: f(x) = (x-3)² + 2")
    print("-" * 50)
    
    gd1 = GradientDescent(learning_rate=0.1, max_iterations=100)
    result1 = gd1.optimize(quadratic_1d, quadratic_1d_gradient, [0.0])
    
    print(f"Starting point: [0.0]")
    print(f"Optimal point: {result1[0]}")
    print(f"Optimal value: {result1[1]:.6f}")
    print(f"Iterations: {result1[2]['iterations']}")
    print(f"Converged: {result1[2]['converged']}")
    
    gd1.plot_convergence_1d(quadratic_1d)
    
    # Example 2: 2D Sphere Function
    print("\n2. 2D Sphere Function: f(x,y) = x² + y²")
    print("-" * 50)
    
    gd2 = GradientDescent(learning_rate=0.1, max_iterations=100)
    result2 = gd2.optimize(sphere_2d, sphere_2d_gradient, [4.0, 3.0])
    
    print(f"Starting point: [4.0, 3.0]")
    print(f"Optimal point: {result2[0]}")
    print(f"Optimal value: {result2[1]:.6f}")
    print(f"Iterations: {result2[2]['iterations']}")
    print(f"Converged: {result2[2]['converged']}")
    
    gd2.plot_convergence_2d(sphere_2d)
    
    # Example 3: Rosenbrock Function (more challenging)
    print("\n3. Rosenbrock Function: f(x,y) = (1-x)² + 100(y-x²)²")
    print("-" * 50)
    
    gd3 = GradientDescent(learning_rate=0.001, max_iterations=10000)
    result3 = gd3.optimize(rosenbrock_2d, rosenbrock_2d_gradient, [-1.0, 1.0])
    
    print(f"Starting point: [-1.0, 1.0]")
    print(f"Optimal point: {result3[0]}")
    print(f"Optimal value: {result3[1]:.6f}")
    print(f"Iterations: {result3[2]['iterations']}")
    print(f"Converged: {result3[2]['converged']}")
    
    gd3.plot_convergence_2d(rosenbrock_2d, x_range=(-2, 2), y_range=(-1, 3))
    
    # Example 4: Learning Rate Comparison
    print("\n4. Learning Rate Comparison on Sphere Function")
    print("-" * 50)
    
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        gd = GradientDescent(learning_rate=lr, max_iterations=50)
        result = gd.optimize(sphere_2d, sphere_2d_gradient, [4.0, 3.0])
        
        history_values = [sphere_2d(point) for point in gd.history]
        
        plt.subplot(2, 2, i+1)
        plt.plot(range(len(history_values)), history_values, 'b-o', markersize=3)
        plt.title(f'Learning Rate = {lr}')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        print(f"LR={lr}: Converged in {result[2]['iterations']} iterations, Final value: {result[1]:.6f}")
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    demonstrate_gradient_descent()