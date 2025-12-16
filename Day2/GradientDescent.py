import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)
# Predict values
y_pred = model.predict(X)
# Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

# --- CORRECTED GRADIENT DESCENT FUNCTION ---
def gradient_descent_fixed(X, y, learning_rate=0.01, epochs=1000):
    m = len(y)
    
    # 1. Reshape y to ensure it's a column vector (m, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1) 
    
    # Initialize theta (parameters) as a column vector (n_features, 1)
    # X.shape[1] is the number of features (2 in X_b)
    theta = np.zeros((X.shape[1], 1))
    
    loss_history = []
    
    for _ in range(epochs):
        # Predictions: X_b @ theta will result in (m, 1)
        predictions = X @ theta
        errors = predictions - y
        
        # Calculate Loss (MSE)
        loss = np.mean(errors ** 2)
        loss_history.append(loss)
        
        # Gradients: X.T @ errors will result in (n_features, 1)
        # @ is the matrix multiplication operator in Python 3.5+
        gradients = (2/m) * X.T @ errors
        
        # 4. Parameter Update: Now both theta and gradients have shape (2, 1),
        # making the subtraction possible via broadcasting with the scalar alpha.
        theta -= learning_rate * gradients
        
    return theta.flatten(), loss_history # Return flattened theta for easier printing

# Prepare data for gradient descent
# X_b is (100, 2)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Run Gradient Descent
theta_gd, loss_history = gradient_descent_fixed(X_b, y, learning_rate=0.01, epochs=1000)

print("Parameters from Gradient Descent (Intercept, Slope):", theta_gd)
print("Mean Squared Error after Gradient Descent:", loss_history[-1])


# --- PLOT LOSS VS. ITERATIONS ---
plt.figure(figsize=(10, 6))
# Create the x-axis (iterations)
iterations = range(len(loss_history)) 
plt.plot(iterations, loss_history, color='purple')
plt.title('Loss (MSE) vs. Iterations (Training Progress)')
plt.xlabel('Iteration')
plt.ylabel('Loss (Mean Squared Error)')
plt.grid(True)
plt.show()

# # Gradient Descent Implementation
# def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
#     m = len(y)
#     theta = np.zeros(X.shape[1])
#     for _ in range(epochs):
#         predictions = X.dot(theta)
#         errors = predictions - y
#         gradients = (2/m) * X.T.dot(errors)
#         theta -= learning_rate * gradients
#     return theta

# # Prepare data for gradient descent (add intercept term)
# X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
# theta_gd = gradient_descent(X_b, y)
# print("Parameters from Gradient Descent:", theta_gd)

# # Predict values using gradient descent parameters
# y_pred_gd = X_b.dot(theta_gd)
# # Plot the results from gradient descent
# plt.scatter(X, y, color='blue', label='Actual Data')
# plt.plot(X, y_pred_gd, color='green', label='GD Regression Line')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Gradient Descent Linear Regression Example')
# plt.legend()
# plt.show()