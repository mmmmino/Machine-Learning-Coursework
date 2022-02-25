import numpy as np
from src import load_data
from src import compute_precision_and_recall, compute_confusion_matrix
from src import compute_f1_measure, compute_accuracy
from src.visualize import plot_decision_regions


class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value


def _visualize_helper(tree, level):
    """
    Helper function for visualize a decision tree at a given level of recursion.
    """
    tab_level = "  " * level
    val = tree.value if tree.value is not None else 0
    print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))


class DecisionTree():
    def __init__(self, attribute_names):
        """

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None
        self.median = []

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    @staticmethod
    def partition(features, targets, attribute_index):
        if features[0][0] == 0 or features[0][0] == 1:
            true_features = features[features[:, attribute_index] == 1]
            true_targets = targets[features[:, attribute_index] == 1]
            false_features = features[features[:, attribute_index] == 0]
            false_targets = targets[features[:, attribute_index] == 0]
        else:
            temp = np.median(features[:, attribute_index])
            true_features = features[features[:, attribute_index] >= temp]
            true_targets = targets[features[:, attribute_index] >= temp]
            false_features = features[features[:, attribute_index] < temp]
            false_targets = targets[features[:, attribute_index] < temp]

        return true_features, true_targets, false_features, false_targets

    def find_best_partition(self, features, targets):
        max_gain = -1
        best_index = None
        for index in range(np.shape(features)[1]):
            true_features, true_targets, false_features, false_targets = self.partition(features, targets, index)
            if true_targets.shape[0] == 0 or false_targets.shape[0] == 0:
                continue
            gain = information_gain(features, index, targets)
            if gain > max_gain:
                max_gain = gain
                best_index = index

        return max_gain, best_index

    @staticmethod
    def count_most(targets):
        num_1 = np.sum(targets)
        if num_1 > targets.shape[0] / 2:
            return 1
        else:
            return 0

    def build_tree(self, features, targets):
        max_gain, best_index = self.find_best_partition(features, targets)
        if best_index is None:
            return Node(value=self.count_most(targets))

        # features_de = np.delete(features, best_index, 1)
        median = np.median(features[:, best_index])
        true_features, true_targets, false_features, false_targets = self.partition(features, targets, best_index)
        true_branch = self.build_tree(true_features, true_targets)
        false_branch = self.build_tree(false_features, false_targets)
        return Node(value=self.count_most(targets), attribute_name=self.attribute_names[best_index],
                    attribute_index=best_index, branches=[false_branch, true_branch])

    def fit(self, features, targets):
        self._check_input(features)
        for index in range(np.shape(features)[1]):
            self.median.append(np.median(features[:, index]))
        self.tree = self.build_tree(features, targets)


    def dfs_predict(self, feature, node, features):
        if len(node.branches) == 0:
            return node.value
        if features[0][1] != 0 and features[0][1] != 1:
            # print(self.median[node.attribute_index])
            if feature[node.attribute_index] >= self.median[node.attribute_index]:
                return self.dfs_predict(feature, node.branches[1], features)
            else:
                return self.dfs_predict(feature, node.branches[0], features)
        else:
            if feature[node.attribute_index] == 1:
                return self.dfs_predict(feature, node.branches[1], features)
            else:
                return self.dfs_predict(feature, node.branches[0], features)

    def predict(self, features):
        self._check_input(features)
        predictions = np.ones(features.shape[0])
        median = np.median(features[:, 0])
        for i in range(predictions.shape[0]):
            predictions[i] = self.dfs_predict(features[i], self.tree, features)

        return predictions


    def visualize(self, branch=None, level=0):
        if not branch:
            branch = self.tree
        _visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level + 1)


def log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def information_gain(features, attribute_index, targets):
    p_1 = np.sum(targets) / targets.shape[0]
    p_0 = 1 - p_1
    H = -(p_1 * np.log2(p_1) + p_0 * np.log2(p_0))
    attribute = features[:, attribute_index]
    p_for_a1 = np.sum(attribute) / attribute.shape[0]
    p_for_a0 = 1 - p_for_a1
    if features[0][0] == 0 or features[0][0] == 1:
        p_1_in_a1 = np.sum(targets[attribute == 1]) / np.sum(attribute == 1)
        p_0_in_a1 = 1 - p_1_in_a1
        p_1_in_a0 = np.sum(targets[attribute == 0]) / np.sum(attribute == 0)
        p_0_in_a0 = 1 - p_1_in_a0
    else:
        temp = np.median(features[:, attribute_index])
        p_1_in_a1 = np.sum(targets[attribute >= temp]) / np.sum(attribute >= temp)
        p_0_in_a1 = 1 - p_1_in_a1
        p_1_in_a0 = np.sum(targets[attribute < temp]) / np.sum(attribute < temp)
        p_0_in_a0 = 1 - p_1_in_a0
    H_a1 = -(p_1_in_a1 * log2(p_1_in_a1) + p_0_in_a1 * log2(p_0_in_a1))
    H_a0 = -(p_1_in_a0 * log2(p_1_in_a0) + p_0_in_a0 * log2(p_0_in_a0))

    return H - (p_for_a1 * H_a1 + p_for_a0 * H_a0)



if __name__ == '__main__':
    # construct a fake tree
    features, targets, attribute_names = load_data(
        '/Users/user/Documents/CS 349/HW2/hw2-perceptron-decision-tree-mmmmino/data/blobs.csv')
    # attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    decision_tree.fit(features, targets)
    predictions = decision_tree.predict(features)
    accuracy = compute_accuracy(targets, predictions)
    decision_tree.visualize()
    plot_decision_regions(features, targets, decision_tree)
