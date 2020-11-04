from backpropagation import *

myNN = MultilayerPerceptron(0.01, 0.01, 2, 6, 0, 5)
myNN.training()
myNN.print_info()