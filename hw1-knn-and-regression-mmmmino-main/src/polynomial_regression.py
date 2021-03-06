import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement polynomial regression from scratch.

        This class takes as input "degree", which is the degree of the polynomial
        used to fit the data. For example, degree = 2 would fit a polynomial of the
        form:

            ax^2 + bx + c

        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        http://interactiveaudiolab.github.io/teaching/eecs349stuff/eecs349_linear_regression.pdf


        Usage:
            import numpy as np

            x = np.random.random(100)
            y = np.random.random(100)
            learner = PolynomialRegression(degree = 1)
            learner.fit(x, y) # this should be pretty much a flat line
            predicted = learner.predict(x)

            new_data = np.random.random(100) + 10
            predicted = learner.predict(new_data)

            # confidence compares the given data with the training data
            confidence = learner.confidence(new_data)


        Args:
            degree (int): Degree of polynomial used to fit the data.
        """
        self.degree = degree
        self.coefs = np.zeros(degree + 1)
        self.transformed_features = None

    def fit(self, features, targets):
        """
        Fit the given data using a polynomial. The degree is given by self.degree,
        which is set in the __init__ function of this class. The goal of this
        function is fit features, a 1D numpy array, to targets, another 1D
        numpy array.


        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing real-valued targets.
        Returns:
            None (saves model and training data internally)
        """
        self.transformed_features = np.zeros((self.degree + 1, features.shape[0]))
        for i in range(self.degree + 1):
            self.transformed_features[i, :] = np.power(features, i)

        temp = np.linalg.inv(self.transformed_features.dot(self.transformed_features.T))
        temp = temp.dot(self.transformed_features)
        self.coefs = temp.dot(targets)

    def predict(self, features):
        """
        Given features, a 1D numpy array, use the trained model to predict target
        estimates. Call this after calling fit.

        Args:
            features (np.ndarray): 1D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """

        return self.transformed_features.T.dot(self.coefs)


