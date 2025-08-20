import numpy as np

"""

This is the MultipleNeuralNetwork class. 

It implements a feedforward neural network with support for multiple layers and various activation functions.

Attributes:
- layer_sizes: The sizes of each layer in the network.
- activations: The activation functions used for each layer.
- error: The loss function used during training.
- weights: The weights of the connections between layers.
- biases: The biases for each layer.

Methods:
    - forward_pass: Computes the forward pass through the network.
    - backward_pass: Computes the backward pass and updates weights.
    - fit: Trains the model on the provided data.
    - predict: Makes predictions on new data.

Example:
```python
# Create a neural network with 2 hidden layers
nn = MultipleNeuralNetwork(layer_sizes=[784, 128, 64, 10], activations=['relu', 'relu', 'softmax'])

# Train the model
nn.fit(X_train, y_train, epochs=10)

# Make predictions
predictions = nn.predict(X_test)
```

"""

class MultipleNeuralNetwork:
    def __init__(self, layer_sizes, activations="sigmoid", error="mse"):
        """
        layer_sizes: list, e.g. [input_dim, hidden1, ..., output_dim]
        activations: string or list of strings; e.g. 'sigmoid' or ['relu', 'sigmoid']
        error: loss type
        """
        self.l = len(layer_sizes) - 1  # length of hidden layers
        self.error = error

        # Support single string or per-layer list for activations
        if isinstance(activations, str):
            self.activations = [activations] * self.l
        else:
            assert len(activations) == self.l, "Activations list must match layer count."
            self.activations = activations

        # Randomly assign weights and biases
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.l)]
        self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(self.l)]

    # Forward pass
    def forward_pass(self, x):

        # Initialize lists to hold activations and weighted sum values
        activations = [x]
        zs = []

        # Initialize input
        A = x

        # Looping through each layers to calculate weighted sum and activation
        for i in range(self.l):
            Z = np.dot(A, self.weights[i]) + self.biases[i]  # Formula: z = A * W + b
            zs.append(Z)
            A = self.activate(Z, self.activations[i])     # Formula: Activation = f(z)
            activations.append(A)

        # Return the final activations and the weighted sums
        return activations, zs

    def backward_pass(self, x, y, learning_rate=0.01, epochs=100, verbose=True):
        n_sample = x.shape[0]
        for epoch in range(epochs):
            # Forward pass
            activations, zs = self.forward_pass(x)

            # Compute loss and deltas
            A_out = activations[-1]  # Output of the last layer
            loss = self.loss(A_out, y)  # Calculate loss or error

            deltas = [None] * self.l  # delta term for each layer

            # Delta for last neuron. Formula: delta = dL/dA * dA/dz = dL/dz
            # For softmax + cross-entropy, use derivative directly.
            if self.activations[-1] == "softmax":
                deltas[-1] = A_out - y  # Simplified for softmax + cross-entropy
            else:
                deltas[-1] = self.derivative_loss(A_out, y) * self.activate_derivative(zs[-1], self.activations[-1])

            # Computing delta for hidden layers.
            # formula: delta[i] = delta[i+1] * W[i+1] * f'(z[i])
            for i in reversed(range(self.l - 1)):
                der_act = self.activate_derivative(zs[i], self.activations[i])
                deltas[i] = np.dot(deltas[i + 1], self.weights[i + 1].T) * der_act

            # Update weights and biases
            # Formula for w: W[i] = W[i] - learning_rate * dW[i], where dW[i] = A[i]* delta[i]
            # Formula for b: b[i] = b[i] - learning_rate * db[i], where db[i] = sum(delta[i])
            for i in range(self.l):
                A_prev = activations[i]
                dw = np.dot(A_prev.T, deltas[i]) / n_sample
                db = np.sum(deltas[i], axis=0, keepdims=True) / n_sample
                self.weights[i] -= learning_rate * dw
                self.biases[i] -= learning_rate * db

            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                print(f"Epoch: {epoch+1}/{epochs} \t Loss: {loss:.4f}")

        return activations[-1]

    # Activation functions and their derivatives
    @staticmethod
    def activate(x, func):
        if func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif func == "tanh":
            return np.tanh(x)
        elif func == "relu":
            return np.maximum(0, x)
        elif func == "linear":
            return x
        elif func == "softmax":
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation: {func}")

    @staticmethod
    def activate_derivative(x, func):
        if func == "sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif func == "tanh":
            return 1 - np.tanh(x) ** 2
        elif func == "relu":
            return (x > 0).astype(float)
        elif func == "linear":
            return np.ones_like(x)
        elif func == "softmax":
            # The derivative for softmax is more complex and usually handled with cross-entropy loss.
            # For backprop, use: delta = (A_out - y) if using softmax + cross-entropy
            # Here, return ones (not used directly)
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {func}")

    def loss(self, A, y):
        if self.error == "mse":
            return np.mean((A - y) ** 2)
        raise NotImplementedError("Only MSE loss is implemented.")

    def derivative_loss(self, A, y):
        if self.error == "mse":
            return 2 * (A - y) / A.shape[0]
        raise NotImplementedError("Only MSE loss derivative is implemented.")

    def fit(self, x, y, learning_rate=0.01, epochs=100, verbose=True):
        self.backward_pass(x, y, learning_rate, epochs, verbose)

    def predict(self, x):
        activations, _ = self.forward_pass(x)
        return activations[-1]