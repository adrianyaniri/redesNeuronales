import numpy as np
from function.act_functions import step_function


class Perceptron:

    def __init__(self, learning_rate=0.01, n_epochs=100, func_act=step_function):
        self.weights = None
        self.bias = None
        self.errors = None
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.func_act = func_act

    def train(self, X, y):
        """
        Function to fit the perceptron
        :param X: Conjunto de datos de entrada
        :param y:  Etiquetas
        """

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_epochs):
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])

                # Actualizar los pesos y bias
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias += self.learning_rate * (y[i] - y_pred)

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.where(z > 0, 1, 0)

    def calculate_error(self, y_true, y_predict):
        """
        Calcula el error de clasificacion para el conjuntos de datos
        :param y_true: Etiquetas reales
        :param y_predict: Etiquetas predichas
        :return:
        """
        return np.mean((y_true != y_predict))

    def get_weights(self):
        """
        Devuelve el vector de weights
        :return:
        """
        return self.weights

    def get_bias(self):
        """
        Devuelve el vector del bias
        :return:
        """
        return self.bias
