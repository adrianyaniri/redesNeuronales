import numpy as np
import matplotlib.pyplot as plt

from utils.plot_error import plot_error


class PerceptronMultilayer:
    def __init__(self, layer_sizes, funcion_activacion):
        """
        Constructor de la clase PerceptronMultilayer
        :param layer_sizes: Numero de neuronas por capas
        :param funcion_activacion: Funcion de activacion

        """
        self.z = None  # Lista de las suma ponderada
        self.activaciones = None  # Lista de activaciones
        self.funcion_activacion = funcion_activacion
        self.layer_sizes = layer_sizes
        self.weights = []  # Vector de Pesos
        self.biases = []  # Vector de Bias

        """
        Inicializacion de los pesos y bias
        Se inicializa de forma aleatoria, de forma uniforme entre -1 - 1
        """
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.uniform(0, 1, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.random.uniform(layer_sizes[i + 1]))

    def forward(self, inputs):
        """
        Propagacion hacia adelante de la red neuronal
        :param inputs: Entrada de la red neuronal
        :return: Salida de la red neuronal
        """
        self.activaciones = [inputs]  # Activaciones
        self.z = []  # Suma ponderas

        for i in range(len(self.weights)):
            self.z.append(np.dot(self.activaciones[-1], self.weights[i]) + self.biases[i])
            self.activaciones.append(self.funcion_activacion(self.z[-1]))

        return self.activaciones[-1]

    def backpropagation(self, inputs, etiquetas, learning_rate):
        """
        Propagacion hacia atras de la red neuronal
        Para actualizar los pesos y bias
        :param inputs: Entrada de la red
        :param etiquetas:
        :param learning_rate: Taza de aprendizada de la red neuronal
        :return:
        """
        output = self.forward(inputs)  # Salida hacia adelante
        deltas = [output - etiquetas]  # Calculo de la diferencia entre la salida y las etiquetas

        # Propagacion hacia atras del error
        for i in range(len(self.activaciones) - 2, -1, -1):
            deltas.append(np.dot(deltas[-1], self.weights[i].T) * self.funcion_activacion(self.activaciones[i],
                                                                                          derivative=True))
        deltas.reverse()  # Se invierte la lista para la coincida con la capa anterior

        for i in range(len(self.weights)):
            self.weights[i] -= np.dot(self.activaciones[i].T, deltas[i + 1]) * learning_rate
            self.biases[i] -= np.sum(deltas[i + 1]) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        """
        Funcion de entrenamiento de la red neuronal
        :param inputs: Entradas de la red neuronal
        :param targets: Etiquetas
        :param epochs: Numero de epocas que se realiza el entrenamiento
        :param learning_rate: Taza de aprendizaje
        :return:
        """
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # .reshape(1,-1) Asegura que tenga la entrada correcta
                # .reshape(-1,1) Garantiza la matriz tenga la dimensiones para la propagacion hacia atras
                self.backpropagation(inputs[i].reshape(1, -1), targets[i].reshape(-1, 1), learning_rate)

            # Calcular error
            error = np.mean((targets - self.forward(inputs)) ** 2)
            if epoch % 20 == 0:
                print(f'Epoca {epoch}: Error = {error}')

    def predict(self, inputs, umbral=0.5):
        """
        Funcion de prediccion de la red neuronal
        :param inputs: Entradas de la red neuronal
        :param umbral: Umbral
        :return:
        """
        outputs = self.forward(inputs)  # Se obtiene promedio de las clases
        outputs = (outputs + 1) / 2
        predictions = np.where(outputs > umbral, 1, 0)  # Le aplica umbral

        return predictions

