# This code is for building a neural network with backpropagation in python.
from random import random
from random import seed

# initialize network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    # add 1 to get n_input weights + a bias
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    # ouput layer has n_output layers,  each one with 1 weight and bias.
    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


seed(1)
network = initialize_network(2,1,2)
for layer in network:
    print(layer)