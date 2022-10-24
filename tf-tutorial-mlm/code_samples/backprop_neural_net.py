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
    if output <= 0:
        return 0
    elif output > 0:
        return 1

# Calculate the derivative of an neuron output
def transfer_sigmoid_derivative(output):
	return output * (1.0 - output)
 


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
            neuron['output'] = transfer_sigmoid(activation)
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
                transfer_sigmoid_derivative(neuron['output'])
    return network

def update_weights(network, row, l_rate):
    
    # for each item in network list
    for i in range(len(network)):
        # Assume the row is passed with a bias term. 
        # to get the inputs, get everything that's been passed except the last element (bias).
        # this is for our first layer
        inputs = row[:-1]
        if i != 0:
            # if it's not the first network, switch inputs to the outputs of the prev layer.
            inputs = [neuron['output'] for neuron in network[i-1]]
            
        # for each neuron in network,
        # decrement the weight by l_rate * delta of that neuron * input corresponding to that weight.
        for neuron in network[i]:
            for j in range(len(inputs)):
                # update the weights.
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            # decrement the bias by l_rate * delta.
            neuron['weights'][-1] -= l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

    return network

def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

# inputs per data set without the expected value
n_inputs = len(dataset[0]) - 1
# unique expected values
n_outputs = len(set([row[-1] for row in dataset]))

network = initialize_network(n_inputs, 5, n_outputs)
trained_network = train_network(network, dataset, 0.3, 10, n_outputs)

# network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
# 	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]

for row in dataset:
    prediction = predict(trained_network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))