import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Compute mean and variance of a dataset manually
data = np.array([10, 12, 23, 23, 16, 23, 21, 16])
mean_manual = np.sum(data) / len(data)
variance_manual = np.sum((data - mean_manual) ** 2) / len(data)
print("Manual Mean:", mean_manual)
print("Manual Variance:", variance_manual)

# Compute mean and variance using NumPy functions
mean_numpy = np.mean(data)
variance_numpy = np.var(data)
print("NumPy Mean:", mean_numpy)
print("NumPy Variance:", variance_numpy)

# Visualize normal distribution
standard_devs = np.std(data)
x = np.linspace(mean_manual - 4*standard_devs, mean_manual + 4*standard_devs, 100)
pdf = norm.pdf(x, mean_manual, standard_devs)
plt.plot(x, pdf, label='Normal Distribution')
plt.title('Normal Distribution of Data')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

