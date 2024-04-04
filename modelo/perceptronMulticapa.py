import numpy as np


class Capa:
    def __init__(self, n_entradas, n_neuronas, funcion_activacion, derivada_activacion):
        self.salida = None
        self.pesos = np.random.rand(n_entradas, n_neuronas)
        self.bias = np.zeros((1, n_neuronas))
        self.funcion_activacion = funcion_activacion
        self.derivada_activacion = derivada_activacion
        self.delta_pesos_anterior = np.zeros_like(self.pesos)

    def forward(self, entradas):
        self.salida = self.funcion_activacion(np.dot(entradas, self.pesos) + self.bias)


class PerceptronMulticapa:
    def __init__(self, funcion_activacion, derivada_activacion):
        self.capas = []
        self.funcion_activacion = funcion_activacion
        self.derivada_activacion = derivada_activacion

    def agregar_capa(self, capa):
        self.capas.append(capa)

    def forward(self, entrada):
        for capa in self.capas:
            capa.forward(entrada)
            entrada = capa.salida
        return entrada

    def backpropation(self, X, y_true, salida, lr, momentum):
        m = X.shape[0]  # cantidad de ejemplos

        for i in reversed(range(len(self.capas))):
            capa = self.capas[i]
            if capa == self.capas[-1]:
                capa.error_neurona = y_true - capa.salida
                capa.delta_neurona = capa.error_neurona * self.derivada_activacion(salida)
            else:
                # Capa oculta
                siguiente_capa = self.capas[i + 1]
                capa.error_neurona = np.dot(siguiente_capa.delta_neurona, siguiente_capa.pesos.T)
                capa.delta_neurona = capa.error_neurona * self.derivada_activacion(capa.salida)

        # Actualizar pesos y bias usando los deltas
        for i in range(len(self.capas)):
            capa = self.capas[i]
            entrada_a_usar = X if i == 0 else self.capas[i - 1].salida

            delta_pesos = np.dot(entrada_a_usar.T, capa.delta_neurona) * lr / m
            deltas_bias = np.sum(capa.delta_neurona, axis=0) * lr / m

            # Actuliza los pesos
            capa.pesos += delta_pesos + momentum * capa.delta_pesos_anterior
            capa.bias += deltas_bias
            capa.delta_pesos_anterior = delta_pesos

    def entrenar(self, X, y, lr, momentum, num_epocas, t_lote, imprimir_cada=500):
        for epoch in range(num_epocas):
            for i in range(0, len(X), t_lote):
                lote_X = X[i:i + t_lote]
                y_batch = y[i:i + t_lote]
                salida = self.forward(lote_X)
                self.backpropation(lote_X, y_batch, salida, lr, momentum)

            if epoch % imprimir_cada == 0:
                mse = np.mean(np.square(self.forward(X) - y))
                print(f'Época: {epoch}, MSE: {mse:.4f}')

        mse_final = np.mean(np.square(self.forward(X) - y))
        print(f'Error final después de {num_epocas} épocas: MSE = {mse_final:.4f}')