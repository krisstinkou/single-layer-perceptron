import math
import matplotlib.pyplot as plt
from random import random


class MultilayerPerceptron:
    """
    Класс для обучения многослойного персептрона с 1 скрытым слоем с возможность прогнозирования временного ряда.
    """

    def __init__(self, alpha, Em, N_hidden: int, N_inputs: int, a: int, b: int):
        """
        Инциализация нейронной сети (задание желаемой ошибки, шага обучения, весов и порогов)
        :param alpha: шаг обучения
        :param Em: желаемая ошибка
        :param N_hidden: количество нейронов в скрытом слое
        :param N_inputs: количество входов сети
        :param a: левая граница обучения
        :param b: правая граница обучения
        """
        self.error_must_be = Em
        self.learning_step_alpha = alpha
        self.count_of_nuerons_in_hidden_layer = N_hidden
        self.count_of_nuerons_in_input_layer = N_inputs
        """
        Вид матрицы весов из входного слоя в скрытый: [
        input_neuron1: [ weight_to_hidden1, weight_to_hidden2... ]
        input_neuron2: [ weight_to_hidden1, weight_to_hidden2... ]
        и т.д.
        ]
        """
        self.weights_from_input_to_hidden_layer = [
            [random() for j in range(self.count_of_nuerons_in_hidden_layer)]
            for i in range(self.count_of_nuerons_in_input_layer)]
        self.weights_from_hidden_to_output_layer = [random() for i in range(self.count_of_nuerons_in_hidden_layer)]
        self.tresholds_in_hidden = [random() for i in range(self.count_of_nuerons_in_hidden_layer)]
        self.treshold_output = random()
        self.left_border = a
        self.right_border = b

    def function(self, x):
        """
        Функция согласно индивидуальному варианту
        :param x: координата точки
        :return: значение фукнции в этой точке
        """
        a = 0.3
        b = 0.1
        c = 0.06
        d = 0.1
        return a * math.cos(b * x) + c * math.sin(d * x)

    def sigmoid(self, x):
        """
        Функция активации: f(x) = 1 / (1 + e^(-x))
        :param x: взвешенная сумма, приходящая на функцию активации.
        :return: значение сигмоидной функции.
        """
        return 1 / (1 + math.exp(-x))

    def weighted_sum(self, index, inputs):
        """
        Взвешенная сумма для одного скрытого нейрона
        :param index: индекс скрытого нейрона
        :param inputs: массив входных данных
        :return: взвешенная сумма для скрытого нейрона под номером index
        """
        return sum([inputs[i] * self.weights_from_input_to_hidden_layer[i][index]
                    for i in range(self.count_of_nuerons_in_input_layer)])

    def derivate(self, x):
        """
        Производная сигмоидной функции активации.
        :return: значение производной от функции активации в точке.
        """
        return math.exp(-x) / ((1 + math.exp(-x)) ** 2)

    def graph_values_training(self):
        """
        Функция для рисования графиков значений на участке обучения.
        Красный - эталонные значения
        Голубой - по результатам обучения
        """
        x = [self.left_border + i * self.learning_step_alpha
             for i in range(int((self.right_border - self.left_border) / self.learning_step_alpha))]
        y = self.inputs
        plt.plot(x, y, color='red')
        x = x[self.count_of_nuerons_in_input_layer:]
        y = self.output_value
        plt.plot(x, y, color='blue')
        plt.show()

    def graph_error_iter(self):
        """
        Функция вывода графика MSE в зависимости от эпохи (по результатам обучения).
        :return:
        """
        x = [i for i in range(1, len(self.total_square_error))]
        y = self.total_square_error[1:]
        plt.plot(x, y, color="green")
        plt.show()

    def get_result(self):
        """
        Вывод результатов обучения.
        """
        for k in range(int((
                                   self.right_border - self.left_border) / self.learning_step_alpha) - self.count_of_nuerons_in_input_layer):
            print(f"t: {self.inputs[k + self.count_of_nuerons_in_input_layer]}\ty: {self.output_value[k]}\t"
                  f"E: {self.inputs[k + self.count_of_nuerons_in_input_layer] - self.output_value[k]}")

    def training(self):
        """
        Функция-обучение.
        """
        self.total_square_error = [self.error_must_be + 1]
        self.inputs = [self.function(self.left_border + i * self.learning_step_alpha)
                       for i in range(int((self.right_border - self.left_border) / self.learning_step_alpha))]

        while (self.total_square_error[-1] > self.error_must_be):
            self.output_value = []
            for k in range(int((
                                       self.right_border - self.left_border) / self.learning_step_alpha) - self.count_of_nuerons_in_input_layer):
                # Считаем взвешенную сумму в нейронах скрытого слоя
                list_weighted_sum_hidden = [
                    self.weighted_sum(i, self.inputs[k:k + self.count_of_nuerons_in_input_layer]) -
                    self.tresholds_in_hidden[i]
                    for i in range(self.count_of_nuerons_in_hidden_layer)
                ]

                # Выходы скрытого слоя
                output_hidden = [self.sigmoid(list_weighted_sum_hidden[i])
                                 for i in range(self.count_of_nuerons_in_hidden_layer)]

                # Взвешенная сумма в выходном нейроне
                weighted_sum_output = sum([output_hidden[i] * self.weights_from_hidden_to_output_layer[i]
                                           for i in range(self.count_of_nuerons_in_hidden_layer)])

                # Добавление значения в массив выходных значений (для отрисовки графика после обучения)
                self.output_value.append(weighted_sum_output - self.treshold_output)

                # Ошибки нейронов
                gamma_output = self.output_value[k] - self.inputs[k + self.count_of_nuerons_in_input_layer]
                gamma_hidden = [
                    gamma_output * self.output_value[k] *
                    (1 - self.output_value[k]) *
                    self.weights_from_hidden_to_output_layer[i]
                    for i in range(self.count_of_nuerons_in_hidden_layer)
                ]

                # Модификация весов и порогов
                self.weights_from_hidden_to_output_layer = [
                    self.weights_from_hidden_to_output_layer[i] - self.learning_step_alpha * output_hidden[
                        i] * self.derivate(weighted_sum_output) * gamma_output
                    for i in range(self.count_of_nuerons_in_hidden_layer)]

                self.treshold_output = self.treshold_output + self.learning_step_alpha * gamma_output * \
                                       self.derivate(weighted_sum_output)

                self.weights_from_input_to_hidden_layer = [
                    [
                        self.weights_from_input_to_hidden_layer[i][j] - self.learning_step_alpha * gamma_hidden[
                            j] *
                        self.derivate(list_weighted_sum_hidden[j]) *
                        self.inputs[k - self.count_of_nuerons_in_input_layer + i]
                        for j in range(self.count_of_nuerons_in_hidden_layer)
                    ]
                    for i in range(self.count_of_nuerons_in_input_layer)
                ]

                self.tresholds_in_hidden = [
                    self.tresholds_in_hidden[j] + self.learning_step_alpha * gamma_hidden[j] *
                    self.derivate(list_weighted_sum_hidden[j])
                    for j in range(self.count_of_nuerons_in_hidden_layer)
                ]

            # Подсчёт среднеквадратической ошибки
            self.total_square_error.append(0.5 * sum(
                [(self.output_value[i] - self.inputs[i + self.count_of_nuerons_in_input_layer]) ** 2
                 for i in range(len(self.output_value))]))

    def print_info(self):
        print("--Hidden layer--\nWeights:")
        for i in range(self.count_of_nuerons_in_input_layer):
            for j in range(self.count_of_nuerons_in_hidden_layer):
                print(f"w{i}{j} = {self.weights_from_input_to_hidden_layer[i][j]}", end='\t')
            print()

        print("\nThresholds:")
        for i in range(self.count_of_nuerons_in_hidden_layer):
            print(f"T{i} = {self.tresholds_in_hidden[i]}", end='\t')

        print("\n\n--Output layer--\nWeights:")
        for i in range(self.count_of_nuerons_in_hidden_layer):
            print(f"w{i} = {self.weights_from_hidden_to_output_layer[i]}")

        print(f"\nThreshold:\nT = {self.treshold_output}")

    def prognose(self):
        print("***ПРОГНОЗИРОВАНИЕ***")

        output_value = []
        num_of_values = int((self.right_border - self.left_border) / self.learning_step_alpha)
        prognose_inputs = [self.function(self.right_border + self.learning_step_alpha * i) for i in
                           range(num_of_values)]

        for k in range(len(prognose_inputs) - self.count_of_nuerons_in_input_layer):
            list_weighted_sum_hidden = [
                self.weighted_sum(i, prognose_inputs[k:k + self.count_of_nuerons_in_input_layer]) -
                self.tresholds_in_hidden[i]
                for i in range(self.count_of_nuerons_in_hidden_layer)
            ]

            output_hidden = [self.sigmoid(list_weighted_sum_hidden[i])
                             for i in range(self.count_of_nuerons_in_hidden_layer)]

            weighted_sum_output = sum([output_hidden[i] * self.weights_from_hidden_to_output_layer[i]
                                       for i in range(self.count_of_nuerons_in_hidden_layer)])

            output_value.append(weighted_sum_output - self.treshold_output)

        for i in range(len(output_value)):
            print(f"t: {output_value[i]}\t y:{prognose_inputs[i + self.count_of_nuerons_in_input_layer]}\t"
                  f" E:{output_value[i] - prognose_inputs[i + self.count_of_nuerons_in_input_layer]}")
