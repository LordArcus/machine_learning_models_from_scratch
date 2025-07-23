import numpy as np
import pandas as pd

'''
This function calculates linear regression coefficients using stochastic gradient descent (SGD).
Attributes:
    coef_ (np.ndarray): Coefficients of the regression model.
    intercept_ (float): Intercept of the regression model.
Methods:
    fit(X_train, y_train): Fits the model to the training data using SGD.
    predict(X_test): Predicts the target variable for the test data.
Usage:
    mlr_sgd = MultipleLinearRegressionGDS(learning_rate=0.01, epochs=100)
    mlr_sgd.fit(X_train, y_train)
    predictions = mlr_sgd.predict(X_test)
Example:
    >>> import numpy as np
    >>> X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y_train = np.array([2, 3, 5, 7])
    >>> mlr_sgd = MultipleLinearRegressionGDS(learning_rate=0.01, epochs=100)
    >>> mlr_sgd.fit(X_train, y_train)
    >>> print(mlr_sgd.coef_)
    [0.5 1. ]
    >>> print(mlr_sgd.intercept_)
    0.5
    >>> X_test = np.array([[5, 6], [6, 7]])
    >>> predictions = mlr_sgd.predict(X_test)
    >>> print(predictions)
    [ 8. 10.]
'''



# Multiple linear regression using stochastic gradient descent
class MultipleLinearRegressionSGD:
    def __init__(self, learning_rate= 0.01, epochs=100, random_state=42):
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_ = None

    def fit(self, X_train, y_train):
        X_train = pd.DataFrame(X_train)
        n_samples, n_features = X_train.shape
        X_train_aug = np.insert(X_train, 0, 1, axis=1)  # Add intercept column
        self.weights_ = np.random.rand(n_features + 1)  # Including intercept

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        for epoch in  range(self.epochs):
            for i in range(n_samples):
                # Predict for the current sample
                # Formula: y_pred = X_train_aug[i] * Weights including intercept
                # Calculate prediction for the current sample
                y_pred = np.dot(X_train_aug[i], self.weights_)

                # Compute the error
                error = y_pred - y_train[i]

                # Compute the gradient
                # Formula: gradients = error * X_train_aug[i]
                gradients = error * X_train_aug[i]

                # Update the weights
                self.weights_ -= self.learning_rate * gradients


    def predict(self, X_test):
        X_test = pd.DataFrame(X_test)
        X_test_aug = np.insert(X_test, 0, 1, axis=1)
        return np.dot(X_test_aug, self.weights_)

    @property
    def coef_(self):
        return self.weights_[1:]
    

    @property
    def intercept_(self):
        return self.weights_[0]