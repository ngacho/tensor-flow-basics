# This code is for building a neural network with backpropagation in python.
from random import random
from random import seed
from math import exp

# initialize network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    # add 1 to get n_input weights + a bias
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    # ouput layer has n_output layers,  each one with n_hidden layer weights and bias.
    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# sigmoid transfer function
def transfer_sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

# relu transfer functions.
def transfer_relu(activation):
    return max(0.0, activation)

# forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neurn[output] = transfer_sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


