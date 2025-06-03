import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input data: 4 samples, each with 2 features
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output labels (like XOR, but let's start with AND: 1 only if both are 1)
y = np.array([[0],
              [0],
              [0],
              [1]])

# Set a seed for reproducibility
np.random.seed(1)

# Initialize weights randomly with mean 0
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1

weights_input_hidden = np.random.uniform(-1, 1, (input_layer_size, hidden_layer_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_layer_size, output_layer_size))


# Learning rate
lr = 0.1

# Training loop
for epoch in range(50000):
    # ---- FORWARD PROPAGATION ----
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output)
    final_output = sigmoid(final_input)

    # ---- BACKPROPAGATION ----
    error = y - final_output
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # ---- WEIGHTS UPDATE ----
    weights_hidden_output += hidden_output.T.dot(d_output) * lr
    weights_input_hidden += X.T.dot(d_hidden) * lr

print("\nTrained Output:")
for i in range(len(X)):
    print(f"Input: {X[i]} → Predicted: {final_output[i][0]:.2f} | Actual: {y[i][0]}")

