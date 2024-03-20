import numpy as np

from conjuntoDato.xor import Xor
from function.funcion_activacion import sigmoid
from modelo.perceptronMulticapa import PerceptronMultilayer
from utils.plot import plot_datos

if __name__ == "__main__":
    # Definir capas y función de activación
    X, y = Xor(5000)
    layer_sizes = [2, 2, 1]  # Dos entradas, tres neuronas ocultas, una salida
    funcion_activacion = sigmoid

    # Crear red neuronal
    red_neuronal = PerceptronMultilayer(layer_sizes, funcion_activacion)

    plot_datos(X, y, ['red', 'green'])

    red_neuronal.train(entrada=X, etiquetas=y,epocas=100, learning_rate=0.1, t_lote=10)
