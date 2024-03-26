import numpy as np


class PerceptronMulticapa:

    def __init__(self, tam_capas, funcion_activacion, learning_rate=0.01):
        self.tam_capas = tam_capas
        self.num_capas = len(tam_capas)
        self.funcion_activacion = funcion_activacion
        self.learning_rate = learning_rate
        self.activations = []

        # Inicializar pesos aleatorios para cada capa

        self.weights = [np.random.randn(tam_capas[i], tam_capas[i + 1]) for i in range(self.num_capas - 1)]
        self.biases = [np.random.randn(1, tam_capas[i + 1]) for i in range(self.num_capas - 1)]

    def forward(self, X_entrada):
        global capa_salida
        entrada = X_entrada
        for i in range(self.num_capas - 1):
            capa_entrada = np.dot(entrada, self.weights[i]) + self.biases[i]
            capa_salida = self.funcion_activacion(capa_entrada)
            entrada = capa_salida
            self.activations.append(capa_salida)
        return capa_salida

    def backward(self, y_true):

        delta_weights = [np.zeros_like(w) for w in self.weights]
        delta_bias = [np.zeros_like(b) for b in self.biases]

        error = self.activations[-1] - y_true

        # Calcula el gradiente de la funcion de perdida con repecto a la salida de la red
        delta = error * self.funcion_activacion(self.activations[-1])

        # Propagar el error
        for i in range(self.num_capas - 2, -1, -1):
            print("Shapes before backward loop:")
            print("delta shape:", delta.shape)
            print("Weights shape:", self.weights[i].shape)
            print("Activations shape:", self.activations[i].shape)
            delta_weights = np.dot(self.activations[i].T, delta)
            delta_bias = np.mean(delta, axis=0)

            # Gradiente salida de la capa salida de la forward
            delta = np.dot(delta, self.weights[i].T) * self.funcion_activacion(self.activations[i], derivative=True)
        return delta_weights, delta_bias

    def train(self, X_train, y_train, epocas, tam_lote):
        num_lotes = len(X_train) // tam_lote
        errores = 0.0
        for epoca in range(epocas):
            # Dividir los datos de entrada y las etiquetas en lotes
            for i in range(num_lotes):
                lote_start = i * tam_lote
                lote_end = lote_start + tam_lote

                X_lote = X_train[lote_start:lote_end]
                y_lote = y_train[lote_start:lote_end]

                prediccion = self.forward(X_lote)

                # Calcular error
                lote_error = np.mean(np.square(prediccion - y_lote))

                # Acumular los errores de cada lote
                errores += lote_error
            promedio_error = errores / num_lotes
            delta_weights, delta_biases = self.backward(y_train)
            print(delta_weights)

            if epoca % 10 == 0:
                print(f'Epoch {epoca}: Loss = {promedio_error}')



