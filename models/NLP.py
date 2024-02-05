import numpy as np

from function.act_functions import sigmoid_function
from function.loss_function import mean_squared_error


class NLP:
    """
    Clase que representa a un red neuronal multicapa
    """

    def __init__(self, n_layers, n_neurons_layer, input_features, activation_function, learning_rate=0.01):
        """
        Incializacion de la clase NLP

        :param n_layers: Numeros de capas
        :param n_neurons_layer: Numero de neuronas por capa
        :param input_features: Numero de caracteristicas de la capa de entrada
        :param activation_function: Funcion de activacion
        :param learning_rate: Taza de aprendizaje
        """

        self.weights = []
        self.bias = []
        self.activations = []

        self.n_layers = n_layers
        self.n_neurons_layer = n_neurons_layer
        self.input_features = input_features
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        # Inicializacion de los pesos y biases
        self.init_weights_and_bias()

    def init_weights_and_bias(self):
        """
        Inicializacion de los pesos y bias
        :return:
        """

        for i in range(self.n_layers - 1):
            if i == 0:
                self.weights.append(np.random.randn(self.input_features, self.n_neurons_layer[i + 1]))
            else:
                self.weights.append(np.random.randn(self.n_neurons_layer[i], self.n_neurons_layer[i + 1]))
            self.bias.append(np.random.randn(self.n_neurons_layer[i + 1]))

    def forward_propagation(self, input_data):

        self.activations = [input_data]

        for i in range(len(self.weights)):
            # Multiplicación de la entrada por los pesos y suma del bias
            z = np.dot(self.activations[-1], self.weights[i]) + self.bias[i]
            # Aplicación de la función de activación
            activation = self.activation_function(z)
            # Almacenamiento de la activación para la siguiente capa
            self.activations.append(activation)

        return self.activations[-1]

    def backward_propagation(self, X_train, y_train, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            # Propagacion hacia adelante
            forward = self.forward_propagation(X_train)
            # Calcular el error
            error = mean_squared_error(forward, y_train)

            # Calculo de las derivadas
            dx = self.calculate_derivada(forward, y_train)

            # Actualizacion de los pesos
            self.update_weights_bias(dx, learning_rate)
            print(f'Error en la epoca {epochs + 1}: {error}')

    def train(self, X_train, y_train, epochs=100):
        """
        Entrean la red neuronal
        :param X_train: Conjuntod de datos de entranamiento
        :param y_train: Etiquetas de entrenamiento
        :param epochs: Numero de iteraciones o epocas de training

        """

        for epoch in range(epochs):
            predictions = self.forward_propagation(X_train)
            error = mean_squared_error(predictions, y_train)
            # Calcular las derivadas
            dx = self.calculate_derivada(predictions, y_train)
            # Actualizar lo pesos y bias
            self.update_weights_bias(dx, self.learning_rate)
            print(f'Error en la epoca {epoch}:{error}')

    def predict(self, input_data):
        """
        Realiza la prediction para un conjunto de datos de entrada
        :param input_data:
        :return:
        """
        activations = self.forward_propagation(input_data)
        return activations[-1]

    def calculate_derivada(self, forward_output, y_true):
        """
        Calcula las derivadas del error con respecto a los pesos y sesgos utilizando retropropagación.

        :param forward_output: Salida de la propagación hacia adelante.
        :param y_true: Etiquetas verdaderas.
        :return: Diccionario con las derivadas de los pesos ('weights') y sesgos ('bias').
        """
        # Inicializar el diccionario de derivadas
        derivatives = {'weights': [], 'bias': []}

        # Calcular la derivada del error con respecto a la salida de la red
        d_error = 2 * (forward_output - y_true) / len(y_true)

        # Retropropagación a través de las capas
        for i in reversed(range(self.n_layers)):
            # Calcular la derivada de la función de activación
            if i + 1 < len(self.activations):
                d_activation = sigmoid_function(self.activations[i + 1], derivative=True)

                # Calcular la derivada del error con respecto a la entrada de la capa
                d_error_input = d_error * d_activation

                # Calcular la derivada del error con respecto a los pesos y sesgos
                d_weights = np.dot(self.activations[i].T, d_error_input)
                d_bias = np.sum(d_error_input, axis=0)

                # Almacenar las derivadas en el diccionario
                derivatives['weights'].insert(0, d_weights)
                derivatives['bias'].insert(0, d_bias)

                # Calcular la derivada del error con respecto a la salida de la capa anterior
                d_error = np.dot(d_error_input, self.weights[i].T)
            else:
                print(f"Error: No hay suficientes elementos en self.activations para el índice {i + 1}")

        return derivatives

    def update_weights_bias(self, dx, learning_rate):
        """
        Actualiza los pesos y el bias
        :param dx: Dicionario con las entrada de las derivada de error
        :param learning_rate: Taza de aprendizaje
        :return:
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dx['weights'][i]
            self.bias[i] -= learning_rate * dx['bias'][i]
