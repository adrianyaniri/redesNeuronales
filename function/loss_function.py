import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calcula la pérdida de error medio cuadrático (MSE).

    Parámetros:
    - y_true: Etiquetas verdaderas.
    - y_pred: Predicciones del modelo.

    Devuelve:
    - MSE: Valor de la pérdida MSE.
    """
    z = np.square(y_true - y_pred)
    mse = np.mean(z)
    return mse
