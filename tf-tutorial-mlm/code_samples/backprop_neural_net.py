# This code is for building a neural network with backpropagation in python.
from random import random
from random import seed
from math import exp
# initialize network


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    # add 1 to get n_input weights + a bias
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]}
                    for i in range(n_hidden)]
    network.append(hidden_layer)
    # ouput layer has n_output layers,  each one with n_hidden layer weights and bias.
    output_layer = [{'weights': [random() for i in range(n_hidden+1)]}
                    for i in range(n_outputs)]
    network.append(output_layer)
    return network

# calculate neuron activation for an input


def activate(weights, inputs):
    # activation = sum(weight_i * input_i) + bias
    # get the last element (bias)
    activation = weights[-1]

    # loop up to the second last element (range loops [0,len(weights)-1))
    for i in range(len(weights)-1):
        # print( multiplying " + str(weights[i]) + " by " + str(inputs[i]))
        activation += weights[i] * inputs[i]

    return activation

# sigmoid transfer function


def transfer_sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

# relu transfer functions.


def transfer_relu(activation):
    return max(0.0, activation)


def transfer_relu_derivative(output):
    if output < 0:
        return 0
    elif output > 0:
        return 1


# forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    # work through each layer in our network.
    for layer in network:
        # new inputs for that layer.
        new_inputs = []
        # get each neuron in layer.
        for neuron in layer:
            # calculate activation
            activation = activate(neuron['weights'], inputs)
            # transfer function
            neuron['output'] = transfer_relu(activation)
            # the ouput
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        # if it's not the last element.
        if i != len(network) - 1:
            for j in range(len(layer)):
                # for each layer
                error = 0.0
                for neuron in network[i+1]:
                    print()
                    # for each neuron in the next layer (we call it x[j]).
                    # get the weight for the neuron of the preceding layer (y[j]), 
                    # multiply it by it by delta of neuron in the next layer, and add for all neurons in the following layer.
                    error += (neuron['weights'][j] * neuron['delta'])
                # append the error for neuron x[j] in our layer.
                errors.append(error)
        else:
            # get the output layer first
            for j in range(len(layer)):
                # for each neuron in the output layer, 
                # calculate the difference between the values of that output and the expected and place in a list.
                neuron = layer[j]
                # append the errors to a list.
                errors.append(neuron['output'] - expected[j])
                
        for j in range(len(layer)):
            neuron = layer[j]
            # for each neuron in the
            # add a delta error * derivative(output) component to the neuron
            neuron['delta'] = errors[j] * \
                transfer_relu_derivative(neuron['output'])
    return network


# a dictionary is a neuron.
# a list is a layer.
# test backpropagation of error

# test forward propagation
network = [
    [
        # neuron 0 with weight for activation 1, activation 2 and bias
        # hidden layer with 2 weights and bias
        {'weights': [0.13436424411240122,
                     0.8474337369372327, 0.763774618976614]},
        # neuron 1 with weight for input 1, input 2, and bias
        {'weights': [0.135335283237,
                     0.0497870683679, 0.763774618976614]}
    ],  # these produce two activations.
    [
        # output 0 with weight for activation 1, activation 2, and bias.
        # output layer 1 with one weight and bias.
        {'weights': [0.2550690257394217,
                     0.272531793034, 0.49543508709194095]},
        # output 1 with weight for activation 1, activation 2, and bias.
        # output layer 2 with one weight and bias.
        {'weights': [0.4494910647887381,
                     0.0301973834223, 0.651592972722763]}
    ]
]

row = [1, 0]
forward_propagate(network, row)
expected = [0, 1]
output = backward_propagate_error(network, expected)
print(output)