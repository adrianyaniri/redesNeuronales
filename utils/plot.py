import matplotlib.pyplot as plt


def plot_datos(X, y, colores, titulo='Conjunto de datos'):
    """

    :param X:
    :param y:
    :param titulo:
    :param colores:

    :return:
    """
    if colores is None:
        colores = ['blue', 'green']
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color=colores[0], label='clase_0', s=30, marker='x')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color=colores[1], label='clase_1', s=30, marker='o')
    plt.legend(loc='best')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.tick_params(labelsize=8)

    bbox = plt.legend().get_bbox_to_anchor().transformed(
        plt.gcf().transFigure)
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), bbox_transform=plt.gcf().transFigure)
    plt.title(titulo)
    plt.grid(True, linewidth=0.3, alpha=0.5)
    plt.show()
