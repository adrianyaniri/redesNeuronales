import numpy as np


def generar_datos_xor(num_muestras):
    np.random.seed(42)
    X = np.random.rand(num_muestras, 2) * 2 - 1
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    y = np.where(y, 1, -1).reshape(-1, 1)
    return X, y

