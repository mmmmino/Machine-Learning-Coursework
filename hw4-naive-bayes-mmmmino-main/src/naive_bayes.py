from src.utils import softmax

import numpy as np


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.smoothing = smoothing
        self.vocab_size = None
        self.p_y = None
        self.count_x_y = None
        self.p_x_y = None

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        """
        probs = self.predict_proba(X)
        # print(np.argmax(probs, axis=1))
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.p_y and self.p_x_y, compute the probability p(y | x) for each row x of X.
        While you will have used log probabilities internally, the returned array should be
            probabilities, not log probabilities. You may use src.utils.softmax to transform log
            probabilities to probabilities.

        Args:
            X: a data matrix of shape `[n_documents, vocab_size]` on which to predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"
        # X = X.toarray()
        p_x_y = self.p_x_y
        p_y = self.p_y
        output = np.zeros([n_docs, n_labels])
        with np.errstate(divide='ignore'):
            for i in range(n_docs):
                for j in range(vocab_size):
                    if X[i, j] != 0:
                        output[i, 0] += X[i, j] * np.log(p_x_y[j, 0], dtype=np.float32)
                        output[i, 1] += X[i, j] * np.log(p_x_y[j, 1], dtype=np.float32)
                output[i, 0] = output[i, 0] + np.log(p_y[0])
                output[i, 1] = output[i, 1] + np.log(p_y[1])
        predict = softmax(output, axis=1)
        return predict

    def fit(self, X, y):
        """
        Compute self.p_y and self.p_x_y using the training data.
        You should store log probabilities to avoid underflow.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        self.p_y should contain the marginal probability of each class label.
            Because we are doing binary classification, you may choose
            to represent p_y as a single value representing p(y=1)

        self.p_x_y should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """

        n_docs, vocab_size = X.shape
        n_labels = 2
        X = X.toarray()
        M_0 = 0  # M_0 total words across all N documents with label y=0
        M_1 = 0  # M_1 total words across all N documents with label y=1
        count_x_y = np.zeros([vocab_size, n_labels])
        p_x_y = np.zeros([vocab_size, n_labels])
        self.vocab_size = vocab_size

        q_y_new = np.array([np.sum(y == 1) / (np.sum(y == 0) + np.sum(y == 1)), 1 - (np.sum(y == 1) / (np.sum(y == 0) + np.sum(y == 1)))])

        self.p_y = q_y_new

        for i in range(n_docs):
            if y[i] == 0:
                M_0 = M_0 + np.sum(X[i])
                for j in range(vocab_size):
                    count_x_y[j, 0] = count_x_y[j, 0] + X[i][j]
            elif y[i] == 1:
                M_1 = M_1 + np.sum(X[i])
                for j in range(vocab_size):
                    count_x_y[j, 1] = count_x_y[j, 1] + X[i][j]

        for i in range(vocab_size):
            p_x_y[i, 0] = (count_x_y[i, 0] + self.smoothing) / (M_0 + self.smoothing * vocab_size)
            p_x_y[i, 1] = (count_x_y[i, 1] + self.smoothing) / (M_1 + self.smoothing * vocab_size)

        self.count_x_y = count_x_y  # the number of each vocabulary in the whole train set with label =1 or label = 0
        self.p_x_y = p_x_y
        # print(np.sum(p_x_y[:, 0]))

    def likelihood(self, X, y):
        """
        Using fit self.p_y and self.p_x_y, compute the log likelihood of the data.
            You should use logs to avoid underflow.
            This function should not use unlabeled data. Wherever y is NaN,
            that label and the corresponding row of X should be ignored.

        Recall that the log likelihood of the data can be written:
          `sum_i (log p(y_i) + sum_j log p(x_j | y_i))`

        Note: If the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the (log) likelihood of the data.
        """
        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"

        n_docs, vocab_size = X.shape
        X = X.toarray()
        Likely_p_x_y, Likely_py = 0, 0
        for i in range(n_docs):
            if y[i] == 0:
                Likely_py += np.log(self.p_y[0], dtype=np.float32)
            elif y[i] == 1:
                Likely_py += np.log(self.p_y[1], dtype=np.float32)
        with np.errstate(divide='ignore'):
            for i in range(n_docs):
                for j in range(vocab_size):
                    if y[i] == 0 and X[i, j] != 0:
                        Likely_p_x_y += np.log(self.p_x_y[:, 0][j], dtype=np.float32) * X[i][j]  # calculate the label 0
                    if y[i] == 1 and X[i, j] != 0:
                        Likely_p_x_y += np.log(self.p_x_y[:, 1][j], dtype=np.float32) * X[i][j]  # calculate the label 1

        Likelihood = Likely_py + Likely_p_x_y

        return Likelihood


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data,
        that uses unlabeled data in the Expectation-Maximization algorithm
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm,
                where each iteration contains both an E step and M step.
                You should check for convergence after each iterations,
                e.g. with `np.isclose(prev_likelihood, likelihood)`, but
                should terminate after `max_iter` iterations regardless of
                convergence.
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        super().__init__(smoothing)
        self.max_iter = max_iter
        self.smoothing = smoothing
        self.p_y = None
        self.p_x_y = None

    def fit(self, X, y):
        """
        Compute self.p_y and self.p_x_y using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* replace the true labels with your predicted
            labels. You can use a `np.where` statement to only update the
            labels where `np.isnan(y)` is True.

        During the M-step, update self.p_y and self.p_x_y, similar to the
            `fit()` call from the NaiveBayes superclass. However, when counting
            words in an unlabeled example to compute p(x | y), instead of the
            binary label y you should use p(y | x).

        For help understanding the EM algorithm, refer to the lectures and
            http://www.cs.columbia.edu/~mcollins/em.pdf
            This PDF is also uploaded to the course website under readings.
            While Figure 1 of this PDF suggests randomly initializing
            p(y) and p(x | y) before your first E-step, please initialize
            all probabilities equally; e.g. if your vocab size is 4, p(x | y=1)
            would be 1/4 for all values of x. This will make it easier to
            debug your code without random variation, and will checked
            in the `test_em_initialization` test case.

        self.p_y should contain the marginal probability of each class label.
            Because we are doing binary classification, you may choose
            to represent p_y as a single value representing p(y=1)

        self.p_x_y should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total
            words across all documents with label y=1, have a vocabulary size
            of V words, and see the word "jackpot" `k` times, then:
            `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing * V)`
            Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size
        # E-step
        X = X.toarray()
        # unlabelled_y_index = np.where(np.isnan(y) == True)
        # labelled_y_index = np.where(np.isnan(y) == False)
        delta = np.ones([n_docs, 2])  # For the full data set
        q_y = 0.5  # Calculate the denominator based on the label
        # q_y_0 = 1 - q_y_1
        # p_y = np.sum(y == 1) / (np.sum(y==0)+np.sum(y==1))
        q_x_y = np.ones([vocab_size, n_labels]) / (2 * vocab_size)

        for j in range(self.max_iter):
            for i in range(n_docs):  # choose the unlabelled data
                numerator_0, numerator_1 = 1, 1
                for k in range(vocab_size):
                    if X[i, k] != 0:
                        numerator_0 = numerator_0 * q_x_y[k, 0] * X[i][k]  # calculate the numerator of delta
                        numerator_1 = numerator_1 * q_x_y[k, 1] * X[i][k]
                denominator = numerator_0 * (1 - q_y) + numerator_1 * q_y
                delta[i, 0] = numerator_0 / denominator
                delta[i, 1] = numerator_1 / denominator

            q_y = np.sum(delta[:, 1]) / n_docs

            for k in range(vocab_size):
                kilt_0, Melita_0, kilt_1, Melita_1 = 0, 0, 0, 0
                for i in range(n_docs):
                    if X[i, j] != 0:
                        kilt_0 += X[i][k] * delta[i, 0]  # numerator_0
                        Melita_0 += np.sum(X[i]) * delta[i, 0]  # denominator_0
                        kilt_1 += X[i][k] * delta[i, 1]  # numerator_1
                        Melita_1 += np.sum(X[i]) * delta[i, 1]  # denominator_1

                q_x_y[k, 0] = (kilt_0 + self.smoothing) / (Melita_0 + vocab_size * self.smoothing)
                q_x_y[k, 1] = (kilt_1 + self.smoothing) / (Melita_1 + vocab_size * self.smoothing)

        q_y_new = np.array([1-q_y, q_y])
        self.p_y = q_y_new
        self.p_x_y = q_x_y
        # print(self.p_x_y)

    def likelihood(self, X, y):
        """
        Using fit self.p_y and self.p_x_y, compute the likelihood of the data.
            You should use logs to avoid underflow.
            This function *should* use unlabeled data.

        For unlabeled data, we define `delta(y | i) = p(y | x_i)` using the
            previously-learned p(x|y) and p(y) necessary to compute
            that probability. For labeled data, we define `delta(y | i)`
            as 1 if `y_i = y` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.
            Following http://www.cs.columbia.edu/~mcollins/em.pdf,
            the log likelihood of the data can be written as:

            `sum_i sum_y (delta(y | i) * (log p(y) + sum_j log p(x_{i,j} | y)))`

        Note: If the word w appears `k` times in a document, the term
            `p(w | y)` should appear `k` times in the likelihood for that document!

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the (log) likelihood of the data.
        """

        assert hasattr(self, "p_y") and hasattr(self, "p_x_y"), "Model not fit!"
        n_docs, vocab_size = X.shape
        n_labels = 2
        count = np.zeros(n_docs)
        for i in range(n_docs):
            count[i] = np.sum(y == y[i])
        count = count / n_docs
        likelihood = 0
        with np.errstate(divide='ignore'):
            for i in range(n_docs):
                for y in range(n_labels):
                    for j in range(vocab_size):
                        likelihood += np.log(X[i, j])
                        if (y == 1) or (y == 0):
                            likelihood += np.log(self.p_y[y])
                        else:
                            likelihood += (self.p_x_y[i, y / count[i]]) * self.p_y[y] * np.log(self.p_y[y])
        return likelihood
