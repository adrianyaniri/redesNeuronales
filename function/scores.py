import numpy as np


def calculate_accuracy(y_true, y_pred):
    class_counts = np.bincount(y_true)

    correct_predictions = np.bincount(y_true == y_pred)
    class_accuracies = correct_predictions / class_counts

    if len(class_accuracies) == 2:
        accuracy = np.mean(class_accuracies)
    else:
        accuracy = np.nan
    return accuracy
