import numpy as np


def sigmoid(x, derivative=False):

    if derivative:
        sig = sigmoid(x)
        return sig * (1 - sig)
    else:
        return 1 / (1 + np.exp(-x))
