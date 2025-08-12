import numpy as np

'''
This is a multiple layer neural network class.

For now it only supports forward pass and uses sigmoid activation function.

'''

class MultipleNeuralNetwork:

    def __init__(self, layer_size=5, Activation_function="sigmoid"):
        self.weights = []
        self.biases = []
        self.l = layer_size - 1  # layer size excluding input layer
        self.activation_function = Activation_function

    def forward_pass(self, x):
        n_sample, n_features = x.shape

        # Initializing weights and biases
        for i in range(self.l):
            self.weights.append(np.random.randn(n_features, n_features)) # Assumes n neurons in each layer
            self.biases.append(np.random.randn(n_features))

        # Looping through layers
        for l in range(self.l):
            A = x
            Z = np.dot(A, self.weights[l]) + self.biases[l]

            # Apply activation function
            if self.activation_function == "sigmoid":
                A = self.sigmoid(Z)
                
        return A


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))