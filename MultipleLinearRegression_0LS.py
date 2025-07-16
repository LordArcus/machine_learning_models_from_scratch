import numpy as np

class MultipleLinearRegression:
    '''
    A multiple linear regression model using Ordinary Least Squares (OLS) method.
    Attributes:
        coef_ (np.ndarray): Coefficients of the regression model.
        intercept_ (float): Intercept of the regression model.
    Methods:
        fit(X_train, y_train): Fits the model to the training data.
        predict(X_test): Predicts the target variable for the test data.
    Usage:
        mlr = MultipleLinearRegression()
        mlr.fit(X_train, y_train)
        predictions = mlr.predict(X_test)       

    Example:
        >>> import numpy as np
        >>> X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y_train = np.array([2, 3, 5, 7])
        >>> mlr = MultipleLinearRegression()
        >>> mlr.fit(X_train, y_train)
        >>> print(mlr.coef_)
        [0.5 1. ]
        >>> print(mlr.intercept_)
        0.5
        >>> X_test = np.array([[5, 6], [6, 7]])
        >>> predictions = mlr.predict(X_test)
        >>> print(predictions)
        [ 8. 10.]   

    Note:
        This implementation assumes that the input data is two-dimensional for features and one-dimensional for the target variable.
        It uses the Ordinary Least Squares method to compute the coefficients and intercept of the regression model.
        The fit method computes the coefficients and intercept based on the training data, and the predict method
        uses these coefficients to make predictions on new data.

        Train data should be in the dataframe format for dataset with single feature to work correctly.

    '''
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X_train, y_train):
        # Add a bias term to the input features so that model can learn the intercept
        X_train = np.insert(X_train, 0, 1, axis=1) 

        # Calculate coefficients using matrix and vector operations
        # Formula: betas = (X^T * X)^-1 * X^T * y
        XT_X = np.dot(X_train.T, X_train)
        XT_X_inv = np.linalg.inv(XT_X)
        XT_y_dot = np.dot(X_train.T, y_train)
        betas = np.dot(XT_X_inv, XT_y_dot)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        # Formula: y_pred = intercept + X_test * coefficients
        return self.intercept_ + np.dot(X_test, self.coef_)