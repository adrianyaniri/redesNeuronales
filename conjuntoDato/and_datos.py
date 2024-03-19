import numpy as np


def And_datos(n_patrones):
    """
     Funcion para generar un conjunto de datos de entrada
      del tipo AND
     :param n_patrones: Cantidad de patrones (ejemplos del conjunto de entradas) para generar un conjunto de datos
     :return: Conjunto de datos
     """
    X = np.random.uniform(-1, 1, (n_patrones, 2))
    y = np.logical_and(X[:, 0] > 0, X[:, 1] > 0).astype(float)

    return X, y