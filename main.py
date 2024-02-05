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
    n_neurons_layer = [5, 2]
    activation_function = sigmoid_function
    learning_rate = 0.1

    nlp = NLP(n_layers=n_layers, n_neurons_layer=n_neurons_layer, activation_function=activation_function,
              input_features=n_input, learning_rate=learning_rate)

    predictions = nlp.forward_propagation(X_train)

    for i in range(len(X_train)):
        print(f"Predicción: {predictions[i]}")
        print(f"Etiqueta real: {y_train[i]}")
    errors = []
    epochs = 100
    for epoch in range(epochs):
        predictions = nlp.forward_propagation(X_train)
        error = mean_squared_error(predictions, y_train)
        errors.append(error)
    print(f'Errores: {errors}')

    nlp.train(X_train, y_train, epochs=100)
    predictions = nlp.forward_propagation(X_test)
    predictions_binary = np.where(predictions > 0.5, 1, 0)

    mse = mean_squared_error(predictions, y_test)
    accuracy = accuracy_score(y_test, predictions_binary)

    print(f'MSE: {mse}')
    print("Precisión:", accuracy)


if __name__ == '__main__':
    main()
