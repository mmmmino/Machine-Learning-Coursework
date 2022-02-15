from your_code.load_data import load_data
import numpy as np
from sklearn.neural_network import MLPClassifier

train_features, test_features, train_targets, test_targets = load_data('mnist-multiclass', fraction=0.7)

# alpha: 1,0.1,0.01,0.001,0.0001
all_accuracy = []
for i in range(0, 10):
    np.random.seed()
    classifier = MLPClassifier(hidden_layer_sizes=(256,), activation = 'logistic', alpha = 0.0001)
    classifier.fit(train_features, train_targets)
    predictions = classifier.predict(test_features)
    accuracy = classifier.score(test_features, test_targets)
    all_accuracy.append(accuracy)
mean = np.mean(all_accuracy)
std = np.std(all_accuracy)
print(all_accuracy)
print(mean)
print(std)