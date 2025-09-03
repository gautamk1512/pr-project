# Practical 1: Implementation of Gradient Descent

## Overview
This practical implements the Gradient Descent algorithm from scratch, demonstrating its application on various optimization problems.

## Theory
Gradient Descent is an iterative optimization algorithm used to find the minimum of a function. It works by:
1. Starting at an initial point
2. Computing the gradient (slope) at the current point
3. Moving in the opposite direction of the gradient
4. Repeating until convergence

The update rule is: **x_{new} = x_{old} - α ∇f(x_{old})**

Where:
- α is the learning rate
- ∇f(x) is the gradient of function f at point x

## Implementation Features

### GradientDescent Class
- **Configurable parameters**: learning rate, max iterations, tolerance
- **Convergence detection**: Stops when change is below tolerance
- **History tracking**: Records optimization path for visualization
- **Multi-dimensional support**: Works with 1D and multi-dimensional functions

### Visualization Capabilities
- **1D Functions**: Shows function curve with optimization path
- **2D Functions**: Contour plots, 3D surface plots, and convergence curves
- **Learning rate comparison**: Demonstrates effect of different learning rates

## Example Functions Implemented

1. **Quadratic 1D**: f(x) = (x-3)² + 2
2. **Sphere 2D**: f(x,y) = x² + y²
3. **Rosenbrock 2D**: f(x,y) = (1-x)² + 100(y-x²)²
4. **Beale Function**: Complex 2D optimization problem

## Usage

### Basic Usage
```python
from gradient_descent import GradientDescent, sphere_2d, sphere_2d_gradient

# Create optimizer
gd = GradientDescent(learning_rate=0.1, max_iterations=100)

# Optimize
result = gd.optimize(sphere_2d, sphere_2d_gradient, [4.0, 3.0])
optimal_point, optimal_value, info = result

print(f"Optimal point: {optimal_point}")
print(f"Optimal value: {optimal_value}")

# Visualize
gd.plot_convergence_2d(sphere_2d)
```

### Running the Demo
```bash
python gradient_descent.py
```

This will run demonstrations on multiple functions with visualizations.

## Key Learning Outcomes

1. **Algorithm Implementation**: Understanding the core gradient descent algorithm
2. **Convergence Analysis**: How learning rate affects convergence
3. **Visualization**: Seeing optimization paths and convergence behavior
4. **Function Complexity**: Different functions have different optimization challenges

## Parameters and Their Effects

### Learning Rate (α)
- **Too small**: Slow convergence, many iterations needed
- **Too large**: May overshoot minimum, cause oscillations or divergence
- **Optimal**: Fast convergence without overshooting

### Tolerance
- Determines when to stop optimization
- Smaller values = more precise results but more iterations

### Max Iterations
- Safety mechanism to prevent infinite loops
- Should be set based on problem complexity

## Dependencies
- numpy: Numerical computations
- matplotlib: Plotting and visualization
- mpl_toolkits.mplot3d: 3D plotting

## Files
- `gradient_descent.py`: Main implementation with demonstrations
- `README.md`: This documentation file

## Expected Output
The demonstration will show:
1. Optimization results for different functions
2. Convergence plots showing how function value decreases
3. Path visualization showing the route to the minimum
4. Learning rate comparison showing different convergence behaviors

## Extensions
You can extend this implementation by:
1. Adding momentum to the gradient descent
2. Implementing adaptive learning rates
3. Adding more complex test functions
4. Implementing stochastic gradient descent
5. Adding regularization terms