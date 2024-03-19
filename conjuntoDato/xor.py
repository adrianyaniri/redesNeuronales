import numpy as np


def Xor(n_patrones):
    """
    Funcion para generar un conjunto de datos de entrada del tipo XOR
    :param n_patrones: Cantidad de patrones (ejemplos del conjunto de entradas) para generar un conjunto de datos
    :return: Conjunto de datos
    """
    # Genera las características de cada patrón
    X = np.random.uniform(-1, 1, (n_patrones, 2))

    # Generar las etiquetas
    y = np.zeros(n_patrones)

    for i in range(n_patrones):
        if (X[i, 0] > 0 and X[i, 1] > 0) or (X[i, 0] < 0 and X[i, 1] < 0):
            y[i] = 0
        else:
            y[i] = 1

    return X, y
