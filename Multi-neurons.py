import numpy as np

'''
This is a multiple layer neural network class.

For now it only supports forward pass and uses sigmoid activation function.

'''

class MultipleNeuralNetwork:
    def __init__(self, layer_sizes, activation_function="sigmoid"):
        """
        layer_sizes: list of integers, e.g. [input_dim, hidden1, ..., output_dim]
        """
        self.weights = []
        self.biases = []
        self.l = len(layer_sizes) - 1  # number of layers excluding input
        self.activation_function = activation_function

        # Initialization of weights and biases
        for i in range(self.l):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.random.randn(1,layer_sizes[i+1]))

    def forward_pass(self, x):
        """
        x: input matrix of shape (num_samples, input_dim)
        """
        A = x
        for l in range(self.l):
            Z = np.dot(A, self.weights[l]) + self.biases[l]
            if self.activation_function == "sigmoid":
                A = self.sigmoid(Z)
        return A

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
