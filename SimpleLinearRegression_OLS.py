import numpy as np

# Simple linear regression from scratch
class SimpleLinearRegression:
    '''
    A simple linear regression model using Ordinary Least Squares (OLS) method.

    Attributes:
        slope (float): The slope of the regression line.
        intercept (float): The y-intercept of the regression line.

    Methods:
        fit(X_train, y_train): Fits the model to the training data.
        coefficients(): Returns the slope and intercept of the fitted model.
        predict(X_test): Predicts the target variable for the test data.    

    Usage:
        slr = SimpleLinearRegression()
        slr.fit(X_train, y_train)
        slope, intercept = slr.coefficients()
        predictions = slr.predict(X_test)

    Example:
        >>> import numpy as np
        >>> X_train = np.array([1, 2, 3, 4, 5])
        >>> y_train = np.array([2, 3, 5, 7, 11, 13])
        >>> slr = SimpleLinearRegression()
        >>> slr.fit(X_train, y_train)
        >>> slope, intercept = slr.coefficients()
        >>> print(f"Slope: {slope}, Intercept: {intercept}")
        >>> X_test = np.array([6, 7, 8])
        >>> predictions = slr.predict(X_test)
        >>> print(predictions)
        [15. 17. 19.]

    Note:
        This implementation assumes that the input data is one-dimensional for both features and target variable.
        It uses the Ordinary Least Squares method to compute the slope and intercept of the regression line.
        The fit method computes the slope and intercept based on the training data, and the predict method
        uses these coefficients to make predictions on new data.  
    
    '''
    
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X_train, y_train):
        n = len(X_train)
        sum_x = np.sum(X_train)
        sum_y = np.sum(y_train)
        sum_xy = np.sum(X_train * y_train)
        sum_x_squared = np.sum(X_train ** 2)

        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        self.intercept = (sum_y - self.slope * sum_x) / n


    def coefficients(self):
        return self.slope, self.intercept


    def predict(self, X_test):
        return self.intercept + self.slope * X_test