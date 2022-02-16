from your_code.load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

train_features, test_features, train_targets, test_targets = load_data('mnist-multiclass', fraction=0.7)
all_accuracy = []

for i in range(0, 10):
    np.random.seed()
    classifier = MLPClassifier(hidden_layer_sizes=(256,), activation = 'logistic', alpha = 0.001)
    classifier.fit(train_features, train_targets)
    predictions = classifier.predict(test_features)
    accuracy = classifier.score(test_features, test_targets)
    all_accuracy.append(accuracy)
mean = np.mean(all_accuracy)
std = np.std(all_accuracy)
fig, axes = plt.subplots(4, 4) # adjust it according to your hidden size
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = classifier.coefs_[0].min(), classifier.coefs_[0].max()
for coef, ax in zip(classifier.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
