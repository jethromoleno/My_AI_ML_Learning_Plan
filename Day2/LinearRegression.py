import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Create and fit the model
model = LinearRegression()
model.fit(x, y)
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)
# Predict values
y_pred = model.predict(x)
# Plot the results
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression Example')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Implement MSE function manually
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Write prediction function
def predict(X_new):
    return model.intercept_ + model.coef_ * X_new
X_new = np.array([[0], [2]])
predictions = predict(X_new)
print("Predictions for new data points:\n", predictions)

# Evaluate error on new data
y_new = 4 + 3 * X_new + np.random.randn(2, 1)
mse_new = mean_squared_error(y_new, predictions)
print("Mean Squared Error on new data:", mse_new)

