import numpy as np

# Define and create two vectors
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# Vector addition
vector_sum = vector_a + vector_b
print("Vector Addition:", vector_sum)

# Vector multiplication (dot product)
vector_dot_product = np.dot(vector_a, vector_b)
print("Vector Dot Product:", vector_dot_product)

# Define and create two matrices
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# Matrix addition
matrix_sum = matrix_a + matrix_b
print("Matrix Addition:\n", matrix_sum)

# Matrix multiplication
matrix_product = np.dot(matrix_a, matrix_b)
print("Matrix Multiplication:\n", matrix_product)


# Verify shape compatibility for matrix multiplication
matrix_c = np.array([[1, 2, 3], [4, 5, 6]])
matrix_d = np.array([[7, 8], [9, 10], [11, 12]])

print("Matrix C shape:", matrix_c.shape)
print("Matrix D shape:", matrix_d.shape)

if matrix_c.shape[1] == matrix_d.shape[0]:
    print("Matrices are compatible for multiplication.")
else:
    print("Matrices are not compatible for multiplication.")

# Reshape matrix_c to make it incompatible with matrix_d
matrix_c_reshaped = matrix_c.reshape(3, 2)
print("Reshaped Matrix C:\n", matrix_c_reshaped)

# Now multiply the reshaped matrix_c with matrix_d
matrix_new_product = np.dot(matrix_c_reshaped, matrix_d)
print("New Matrix Multiplication Result:\n", matrix_new_product)

# Interpret matrix results in Machine Learning context (Samples x Features)

features = np.array([[1, 2, 3], [4, 5, 6]])  # 2 samples, 3 features
weights = np.array([[0.2, 0.8], [0.5, 0.5], [0.3, 0.7]])  # 3 features, 2 outputs
predictions = np.dot(features, weights)
print("Predictions (Feature x Weights):\n", predictions)

