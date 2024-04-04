import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivada(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x, derivative=False):
    if derivative:
        sig = sigmoid(x)
        return sig * (1 - sig)
    else:
        return 1 / (1 + np.exp(-x))
