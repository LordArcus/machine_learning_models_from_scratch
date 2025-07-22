import numpy as np

class MultipleLinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=100, random_state=42):
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_ = None    

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        X_train_aug = np.insert(X_train, 0, 1, axis=1)  # Add intercept column
        self.weights_ = np.random.rand(n_features + 1)  # Including intercept

        for epoch in range(self.epochs):
            y_pred = np.dot(X_train_aug, self.weights_)  # Formula: y_pred = X_train_aug * Weights including intercept
            error = y_pred - y_train
            gradient = np.dot(X_train_aug.T, error) / n_samples  # Formula: gradient = (X_train_aug^T * error) / n_samples
            
            # Weight update
            self.weights_ -= self.learning_rate * gradient

    def predict(self, X_test):
        X_test_aug = np.insert(X_test, 0, 1, axis=1)
        return np.dot(X_test_aug, self.weights_)
    
    # Return coefficient 
    @property
    def coef_(self):
        return self.weights_[1:]
    
    # Return intercept
    @property
    def intercept_(self):
        return self.weights_[0]