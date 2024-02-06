import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score

from function.act_functions import relu, sigmoid_function
from function.generate_data import datos_linealmente_seprables, datos_xor
from function.loss_function import mean_squared_error
from models.NLP import NLP
from sklearn.model_selection import train_test_split


def main():
    # Generar datos
    X_xor, y_xor = datos_xor(10, 2)
    # y = np.random.randn(2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X_xor, y_xor, test_size=0.2, random_state=42)

    n_input = X_xor.shape[1]
    n_layers = 2
    n_neurons_layer = [15, 2]
    activation_function = sigmoid_function
    learning_rate = 0.1

    nlp = NLP(n_layers=n_layers, n_neurons_layer=n_neurons_layer, activation_function=activation_function,
              input_features=n_input, learning_rate=learning_rate)
    epochs = 1000
    nlp.train(X_train, y_train, epochs=epochs)


if __name__ == '__main__':
    main()
