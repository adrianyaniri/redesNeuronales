import numpy as np

step_function = lambda x: 1 if x > 0 else -1


def sigmoid_function(z, derivative=False):
    """
    Funcion de activacion sigmoidea
    :param z: Entrada
    :param derivative: Indica si se calcula la derivada
    :return:
    """
    if derivative:
        return (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
    else:
        return 1 / (1 + np.exp(-z))


tanh_function = lambda x: np.tanh(x)
dTanh_function = lambda x: 1 - np.tanh(x) ** 2

relu = lambda x: np.maximum(0, x)
