def get_perceptron_line(w, b):
    """
        Genera la línea del perceptrón a partir del vector de pesos y el bias.
        Args:
            w (array): Vector de pesos del perceptrón.
            b (float): Bias del perceptrón.
        Returns:
            x_min, x_max, y_min, y_max: Tupla con las coordenadas de la línea.
    """
    x_min, x_max = -1, 1
    y_min = -(w[0] * x_min + b) / w[1]
    y_max = -(w[0] * x_max + b) / w[1]
    return x_min, x_max, y_min, y_max
