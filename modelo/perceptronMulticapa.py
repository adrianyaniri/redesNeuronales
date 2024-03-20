import numpy as np


class PerceptronMultilayer:
    def __init__(self, tamanio_capas, funcion_activacion):
        """
        Constructor de la clase PerceptronMultilayer
        :param tamanio_capas: Numero de neuronas por capas
        :param funcion_activacion: Funcion de activacion

        """
        self.z = [] # Lista de las suma ponderada
        self.activaciones = []  # Lista de activaciones
        self.funcion_activacion = funcion_activacion
        self.tamanio_capas = tamanio_capas
        self.pesos = []  # Vector de Pesos
        self.biases = []  # Vector de Bias

        """
        Inicializacion de los pesos y bias
        Se inicializa de forma aleatoria, de forma uniforme entre -1 - 1
        """
        for i in range(len(tamanio_capas) - 1):
            self.pesos.append(np.random.uniform(0, 1, (tamanio_capas[i], tamanio_capas[i + 1])))
            self.biases.append(np.random.uniform(tamanio_capas[i + 1]))

    def forward(self, entrada):
        """
        Propagacion hacia adelante de la red neuronal
        :param entrada: Entrada de la red neuronal
        :return: Salida de la red neuronal
        """
        self.activaciones = entrada  # Activaciones
        self.z = []  # Suma ponderas

        for i in range(len(self.pesos)):
            self.z.append(np.dot(self.activaciones[-1], self.pesos[i]) + self.biases[i])
            self.activaciones.append(self.funcion_activacion(self.z[-1]))

        return self.activaciones[-1]

    def backpropagation(self, entrada, etiquetas, learning_rate):

        output = self.forward(entrada)  # Salida hacia adelante
        deltas = [output - etiquetas]  # Calculo de la diferencia entre la salida y las etiquetas

        # Propagacion hacia atras del error
        for i in range(len(self.activaciones) - 2, -1, -1):
            deltas.append(np.dot(deltas[-1], self.pesos[i].T) * self.funcion_activacion(self.activaciones[i],
                                                                                        derivative=True))
        deltas.reverse()  # Se invierte la lista para la coincida con la capa anterior

        for i in range(len(self.pesos)):
            self.pesos[i] -= np.dot(self.activaciones[i].T, deltas[i + 1]) * learning_rate
            self.biases[i] -= np.sum(deltas[i + 1]) * learning_rate

    def train(self, entrada, etiquetas, epocas, learning_rate, t_lote):

        num_examples = len(entrada)

        # Verificar que el tamaño del lote sea válido
        if num_examples % t_lote != 0:
            raise ValueError(
                f"El tamaño del lote {t_lote} no es válido para el número total de ejemplos {num_examples}")

        for epoch in range(epocas):
            print(f'Epoch {epoch}:')
            for batch_start in range(0, num_examples, t_lote):
                batch_end = batch_start + t_lote
                batch_inputs = entrada[batch_start:batch_end]
                batch_targets = etiquetas[batch_start:batch_end]

                # Realizar el entrenamiento en el lote actual
                for i in range(len(batch_inputs)):
                    self.backpropagation(batch_inputs[i].reshape(1, -1), batch_targets[i].reshape(-1, 1), learning_rate)

            # Impresión del tamaño del lote fuera del bucle interno
            print(f'Tamaño del lote: {t_lote}')

            # Calcular error después de procesar todos los lotes en la época actual
            error = np.mean((etiquetas - self.forward(entrada)) ** 2)
            print(f'Error en la época {epoch}: {error}')

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
