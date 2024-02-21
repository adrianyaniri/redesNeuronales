import matplotlib.pyplot as plt


def plot_error(epochs, errors):
    """
    Función para generar el gráfico de error.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, errors)
    plt.xlabel("Época")
    plt.ylabel("Error")
    plt.title("Error durante el entrenamiento")
    plt.show()
