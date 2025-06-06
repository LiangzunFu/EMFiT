import numpy as np


confusion_matrix = np.array([
    [1033, 201, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [313, 5591, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 108, 762, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 7, 7, 4055, 61, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 151, 141, 152, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 79, 65, 3940, 163, 35, 0, 51, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 514, 139, 189, 1, 51, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 182, 146, 195, 23, 624, 26, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 30, 64, 113, 59, 780, 57, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 111, 13, 23, 29, 3899, 693, 13, 56, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 13, 15, 986, 5533, 473, 38, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 314, 1564, 248, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53, 314, 1858, 129, 61],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 456, 772],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 193, 2130]
])
# total samples
total_samples = np.sum(confusion_matrix)

accuracy = np.sum(np.diag(confusion_matrix)) / total_samples

precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
precision = np.mean(precision)

recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.mean(recall)

f1 = 2 * (precision * recall) / (precision + recall)

acc_random = np.sum((np.sum(confusion_matrix, axis=1) * np.sum(confusion_matrix, axis=0)) / total_samples**2)

kappa = (accuracy - acc_random) / (1 - acc_random)

accuracy_per_class = []
for i in range(confusion_matrix.shape[0]):
    TP = confusion_matrix[i, i]
    class_total_samples = np.sum(confusion_matrix[i, :])
    class_accuracy = TP / class_total_samples
    accuracy_per_class.append(class_accuracy)
print(total_samples)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Kappa: {kappa:.4f}")
print("class accuracy:")
for i, acc in enumerate(accuracy_per_class):
    print(f"class {i}: {acc:.4f}")