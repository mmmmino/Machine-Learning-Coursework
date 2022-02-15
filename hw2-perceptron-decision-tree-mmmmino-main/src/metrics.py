import numpy as np


# Do not import sklearn!

def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    confusion_matrix = np.zeros((2, 2))
    for i in range(actual.shape[0]):
        if actual[i] and predictions[i]:
            confusion_matrix[1, 1] += 1
        elif actual[i] and not predictions[i]:
            confusion_matrix[1, 0] += 1
        elif not actual[i] and predictions[i]:
            confusion_matrix[0, 1] += 1
        else:
            confusion_matrix[0, 0] += 1
    return confusion_matrix
    # raise NotImplementedError()


def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    confusion_matrix = compute_confusion_matrix(actual, predictions)
    accuracy = (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / actual.shape[0]
    return accuracy
    # raise NotImplementedError()


def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall
    
    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    confusion_matrix = compute_confusion_matrix(actual, predictions)
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
    return precision, recall
    # raise NotImplementedError()


def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure
   
    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    precision, recall = compute_precision_and_recall(actual, predictions)
    f1_measure = 2 * precision * recall / (precision + recall)
    return f1_measure
    # raise NotImplementedError()
