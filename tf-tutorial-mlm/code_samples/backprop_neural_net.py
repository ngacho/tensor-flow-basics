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
    # activation = sum(weight_i * input_i) + bias
    # get the last element (bias) 
    activation = weights[-1]

    # loop up to the second last element (range loops [0,len(weights)-1))
    for i in range(len(weights)-1):
        print("weight ", weights[i])

        # print( multiplying " + str(weights[i]) + " by " + str(inputs[i]))
        activation += weights[i] * inputs[i]

    print(activation)
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

# test forward propagation
network = [
            [
                {'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]},  # hidden layer with 2 weights and bias
            ],
		    [
                {'weights': [0.2550690257394217, 0.49543508709194095]},                   # output layer 1 with one weight and bias.
                {'weights': [0.4494910647887381, 0.651592972722763]}                      # output layer 2 with one weight and bias.
            ]
        ]

row = [1, 0]
output = forward_propagate(network, row)
print(output)


