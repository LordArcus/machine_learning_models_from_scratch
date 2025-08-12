import numpy as np


'''
This is the hinge loss function used in Support Vector Machines (SVMs).

Total Loss = ||w||^2 + C * sum( max(0, 1 - y_i (w·x_i + b)) )

Also, Stochastic Gradient Descent is used to train the model.

The model is trained by iteratively updating the weights and bias using the gradients of the loss function with respect to these parameters.

Attributes:
- w: Weights of the model
- b: Bias term
- lr: Learning rate for weight updates
- C: Regularization parameter
- epochs: Number of iterations for training

Methods:
- fit: Trains the model on the provided data
- predict: Makes predictions on new data

Usage:
```python
# Initialize the model
my_model = SupportVectorMachine_linear(C=1.0, learning_rate=0.01, epochs=1000)
my_model.fit(X_train, y_train)
my_predictions = my_model.predict(X_test)

# Calculation of Accuracy, Precision and Recall
print("Accuracy:", accuracy_score(y_test, my_predictions))
print("Classification Report:\n", classification_report(y_test, my_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, my_predictions))

'''



class SupportVectorMachine_linear:

    def __init__(self, C=1.0, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.C = C
        self.epochs = epochs
        self.w = None  # Weights
        self.b = None


    def fit(self, X, y):
        # Fit the model to the training data
        n_samples, n_features = X.shape

        self.w = np.random.rand(n_features) * 0.1 #Randomly assigned the weights
        self.b = np.random.rand(1) * 0.1  # Randomly assigned bias

        for epoch in range(self.epochs):
            for i in range(n_samples):
                # Compute the hinge loss
                condition = y[i] * (np.dot(self.w, X[i]) + self.b)
                if condition >= 1:
                    # Update weight
                    self.w -= self.lr * ( 2 * self.w)
                else:
                    # Update weights and bias
                    self.w -= self.lr * (2 * self.w - self.C * y[i] * X[i])
                    self.b += self.lr * (self.C * y[i])
                    

    def predict(self, X):
        # Make predictions on new data
        z = np.dot(X, self.w) + self.b

        return np.sign(z)