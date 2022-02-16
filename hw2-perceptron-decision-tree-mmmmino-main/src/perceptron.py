import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. The data
    in `data/transform_me.csv` is not linearly separable; you should implement
    this function such that the perceptron algorithm can achieve perfect performance.
    This function should only apply for this specific dataset -- it should not be
    used for the other datasets in this assignment.
    Refer to `tests/test_perceptron.py` for how this function will be used.

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    temp = np.sqrt(features[:, 0] ** 2 + features[:, 1] ** 2)
    transformed_features = features
    transformed_features[:, 0] = temp
    transformed_features[:, 1] = temp
    return transformed_features
    # raise NotImplementedError()


class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single layer
        perceptron is an algorithm for supervised learning of a binary classifier. The
        idea is to draw a linear decision boundary in the space that separates the
        points in the space into two partitions. Points on one side of the line are one
        class and points on the other side are the other class.

        To simplify comparisons and allow for reproducibility, the Perceptron's
        initial weights should be set to 1 rather than randomly initialized.

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.weights = None

        # raise NotImplementedError()

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets. Note
        that the csv datasets use class labels 0 and 1, but the Perceptron
        algorithm requires classes (-1 or 1). This function should terminate
        either after convergence (the decision boundary does not change between iterations)
        or after max_iterations (defaults to 200) iterations are done. Here is pseudocode
        for the perceptron learning algorithm:

        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that in the above pseudocode, label_for_example is either -1 or 1.

        Note that your weights vector should include a bias term, so if your data has
        two features your should initialize weights as (w_0=1, w_1=1, w_2=1).

        Use only numpy to implement this algorithm. 

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """

        # Relabel targets to (-1, 1) from (0, 1)
        targets = np.where(targets == 0, -1, targets)
        self.weights = np.ones(features.shape[1] + 1)
        for i in range(self.max_iterations):
            weight = self.weights
            for j in range(features.shape[0]):
                feature = features[j]
                feature = np.append(feature, 1)
                if np.dot(self.weights, feature) * targets[j] <= 0:
                    self.weights = self.weights + feature * targets[j]
            if (self.weights == weight).all():
                break
        return i
        # raise NotImplementedError()

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        NOTE: to comport with the other models in this homework,
        you should output these predictions as 0 or 1, not as -1 or 1.
        This can be done with `np.where(predictions == -1, 0, predictions)`
        as shown below.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        # raise NotImplementedError()
        predictions = np.zeros(features.shape[0])
        for i in range(features.shape[0]):
            feature = features[i]
            feature = np.append(feature, 1)
            if np.dot(self.weights, feature) >= 0:
                predictions[i] = 1.0
            else:
                predictions[i] = 0.0
        # Don't forget to relabel predictions back to (0, 1) from (-1, 1)

        return predictions
