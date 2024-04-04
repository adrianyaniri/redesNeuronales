import numpy as np
from matplotlib import pyplot as plt

from function.funcion_activacion import  tanh, tanh_derivada
from modelo.perceptronMulticapa import PerceptronMulticapa, Capa
from utils.generar_datos import generar_datos_xor

if __name__ == "__main__":
    np.random.seed(42)
    X, y = generar_datos_xor(100)

    red = PerceptronMulticapa(funcion_activacion=tanh, derivada_activacion=tanh_derivada)
    red.agregar_capa(Capa(2, 4, funcion_activacion=tanh, derivada_activacion=tanh_derivada))
    red.agregar_capa(Capa(4, 2,funcion_activacion=tanh, derivada_activacion=tanh_derivada))
    red.agregar_capa(Capa(2, 1, funcion_activacion=tanh, derivada_activacion=tanh_derivada))

    lr = 0.001
    momentun = 0.9
    t_lote = 1
    n_epocas = 5000
    imprimir_cada = 1000

    red.entrenar(X=X, y=y, num_epocas=n_epocas, lr=lr, momentum=momentun, t_lote=t_lote, imprimir_cada=imprimir_cada)

    salida = red.forward(X)
    predicciones = np.where(salida > 0.5, 1, -1)

    # Graficar el conjunto real
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='coolwarm')
    plt.title('Conjunto Real')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()

    # Graficar el conjunto después de entrenar
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=predicciones.flatten(), cmap='coolwarm')
    plt.title('Conjunto Después de Entrenar')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
