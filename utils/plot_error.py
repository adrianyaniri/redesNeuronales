def plot_error(epochs, errors):
    """
    Función auxiliar para dibujar el gráfico del error por época.
    :param epochs: Lista de las épocas.
    :param errors: Lista de los errores.
    :return: None.
    """

    import matplotlib.pyplot as plt
    if epochs is not None and errors is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, errors)
        plt.xlabel("Época")
        plt.ylabel("Error")
        plt.title("Error durante el Entrenamiento")
        plt.show()
