import numpy as np
import pandas as pd


class LogisticRegression:
    '''
    This class implements a simple logistic regression model from scratch.
    It uses gradient descent to optimize the weights and bias for binary classification tasks.
    
    Parameters:
    - x: Input features (numpy array of shape (m, n))
    - y: Target labels (numpy array of shape (m, 1) or (m,))
    
    Methods:
    - train: Trains the model using gradient descent.
    - predict: Makes predictions on new data.
    - loss: Computes the binary cross-entropy loss.
    - sigmoid: Computes the sigmoid activation function.        

    Attributes:
    - param: Dictionary containing model parameters 'w' (weights) and 'b' (bias).
    - m: Number of samples.
    - n: Number of features.
    - x: Input features.
    - y: Target labels.
    - result: DataFrame to store true and predicted labels for analysis.    

    Note:
    - The model assumes binary classification with labels 0 and 1.
    - The input features x should be a numpy array of shape (m, n) where m is the number of samples and n is the number of features.
    - The target labels y should be a numpy array of shape (m, 1) or (m,) containing binary values (0 or 1).
    - The model uses a learning rate (alpha) and number of epochs for training.
    - The loss function used is binary cross-entropy.
    - The sigmoid function is used for activation.
    - The model can be used to make predictions on new data after training.
    - The model prints the loss at each epoch and the final weights and bias after training.    

    Usage:

    ```python
    import numpy as np
    from LogisticRegression import LogisticRegression

    # Example data
    x = np.random.rand(100, 2)  # 100 samples, 2 features
    y = np.random.randint(0, 2, size=(100,))  # Binary labels

    # Create and train the model
    model = LogisticRegression(x, y)
    model.train(alpha=0.01, epochs=10)

    # Make predictions on new data
    predictions = model.predict(x)
    print(predictions)
    ``` 

    '''


    def __init__(self, x, y):
        self.param = {}
        self.m, self.n = x.shape
        self.param['w'] = np.random.randn(self.n, 1) * 0.01
        self.param['b'] = np.zeros(1)

        self.x = x
        # Ensure y is numpy array of shape (m, 1)
        self.y = np.array(y).reshape(-1, 1)
        self.result = pd.DataFrame()

    def train(self, alpha=0.01, epochs=10):
        for epoch in range(epochs):
            print("Epoch:", epoch, end="")
            z = np.dot(self.x, self.param['w']) + self.param['b']
            self.y_pred = self.sigmoid(z)

            self.result["y_true"] = self.y.flatten()
            self.result["y_pred"] = self.y_pred.flatten()

            # Update the parameters
            dw = (1. / self.m) * np.dot(self.x.T, (self.y_pred - self.y))
            db = (1. / self.m) * np.sum(self.y_pred - self.y)
            self.param['w'] = self.param['w'] - alpha * dw
            self.param['b'] = self.param['b'] - alpha * db

            # Calculate new predictions for loss
            self.y_pred = self.sigmoid(np.dot(self.x, self.param['w']) + self.param['b'])
            loss = self.loss(self.y, self.y_pred)
            print(" Loss:", loss)

        print(", Final Loss=", loss)
        print("W:{}, b={}".format(self.param['w'].flatten(), self.param['b']))

    def loss(self, y, y_pred):
        """
        Compute the binary cross-entropy loss function
        """
        # Clip y_pred to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # Use correct binary cross-entropy loss formula
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def sigmoid(self, z):
        """
        Compute the sigmoid function
        """
        return 1.0 / (1 + np.exp(-z))

    def predict(self, x):
        y_pred = self.sigmoid(np.dot(x, self.param['w']) + self.param['b'])
        return (y_pred > 0.5).astype(int)