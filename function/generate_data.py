import numpy as np


def datos_linealmente_seprables(n_samples, n_features):
    """
    Genera un dataset de datos linealmente seprables
    :param n_samples: Numero de datos
    :param n_features: Numero de features
    :return:
    """
    X = np.random.rand(n_samples, n_features)

    # Etiquetas
    y = np.where(X[:, 0] + X[:, 1] > 1, 1, 0)

    return X, y


def datos_xor(n_samples, n_features):
    """
    Genera un dataset de conjuntos de datos XOR aleatorio
    :param n_samples: Numero de datos
    :param n_features: Numero de features
    :return:
    """

    X = np.random.rand(n_samples, n_features)
    y1 = np.where((X[:, 0] < 0.5) & (X[:, 1] < 0.5) | (X[:, 0] > 0.5) & (X[:, 1] > 0.5), 1, 0)
    y2 = np.where((X[:, 0] < 0.5) & (X[:, 1] < 0.5) | (X[:, 0] > 0.5) & (X[:, 1] > 0.5), 0, 1)
    y = np.column_stack((y1, y2))
    return X, y
