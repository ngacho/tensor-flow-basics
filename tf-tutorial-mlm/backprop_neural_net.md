# A Neural Network with backpropagation in Python.

## Backpropagation Algorithm.
The backpropagation algorithm is a supervised learning method for multilayer feed-forward networks from the field of Artificial Neural Networks (ANNs).

Feed forward neural networks are inspired by the formation processing of one or more neural cells, called a neuron. A neuron accepts input signals via the dendrites, which pass the electrical signal down to the cell body. The axon carries the signal out to synapses, which are the connections of a cell's axon to other cell's dentrites.

The principle of backpropagation approach is to model a given function by modifying internal weightings of input signals to produce an expected output signal. The system is trained using a supervised learning method, where the error between the system's output and a known expected output is presented to the system and used to modify its internal state.

Technically, the backpropagation algorithm is a methd for training the weights in a multilayer feed-forward neural network. As such, it requires a network structure to be defined of one or more layers where one layer is fully connected to the next layer. A standard network structure is one input layer, one hidden layer, and one output layer.

In classification problems, best results are achieved when the network has one neuron in the output layer for each class value. For example, a 2-class or binary classification problem with the class values of A and B. These expected outputs would have to be transformed into binary vectors with one column for each class value. Such as [1,0] and [0,1] for A and B respectively. This is called one hot encoding.

We'll be using the [Wheat Seeds Dataset](https://archive.ics.uci.edu/ml/datasets/seeds) and can be downloaded [here](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv)

The seeds dataset involves the prediction of species given measurement seeds from different varieties of wheat.

There're 201 records and 7 numerical input variables. It is a classification problem with 3 output classes. 
7 input variables:
- Area A
- Perimeter P
- Compactness c = 4 * pi * A / P^2
- Length of Kernel
- Width of Kernel
- Asymmetry coefficient
- Length of Kernel Groove.

## Tutorial
1. [Initialize Network](#initialize-network)
2. [Forward Propagate](#forward-propagate)
3. Back Propagate Error
4. Train Network
5. Predict
6. Seeds Dataset case study.

### Initialize Network

Each neuron has a set of weights that need to be maintained. One weight for each input connection and an additional weight of the bias. We will need to store additional properties for a neuron during training, therefore we will use a dictionary to represent each neuron and store properties by names such as 'weights' for the weights.

A network is organized into layers. The input layer is really just a row from our training data set. The first real layer is the hidden layer. This is followed by the output layer that has one neuron for each class value. We will organize layers as arrays of dictionaries and treat the whole network as an array of layers.

It is good practice to initialize network weights to small random numbers. In this case, we will use random numbers in the range of 0 to 1.

Sample code can be found [here](https://github.com/ngacho/tensor-flow-basics/blob/02c908c149324c2b2c83563453207b4585ec4d77/tf-tutorial-mlm/code_samples/backprop_neural_net.py#L6)

### Forward Propagate
We can calculate an ouput from a neural network by propagating an input signal through each layer until the output layer outputs its values. This is called forward propagation.

It is the technique we will need to generate predictions during training that will need to be corrected, and it is the method we will need after the network is trained to make predictions on new data.

We can break forward propagation down into three parts.
1. [Neuron Activation](#neuron-activation)
2. [Neuron Transfer](#neuron-transfer)
3. [Forward Propagation](#forward-propagation)

#### Neuron Activation.
The first step is to calculate the activation of one neuron given an input.

The input could be a row from our training data set, as in the case of each of the hidden layer. it may also be the outputs from each neuron in the hidden layer, in the case of output layer.

Neuron activation is calculated as the weighted sum of the inputs.

```
activation = sum(weight_i * input_i) + bias
```

Where **weight** is a network weight, **input** is an input, **i** is the index of a weight or an input and **bias** is a special weight that has no input to multiply with (or you can think of the input as always being 1.0).

![Here is a sample of the multiplication of neural networks](https://www.jeremyjordan.me/content/images/2018/01/Screen-Shot-2017-11-07-at-12.32.19-PM.png)

[sample activation function](https://github.com/ngacho/tensor-flow-basics/blob/501ff68c6dda2616c2b5eac711432b85184a81bc/tf-tutorial-mlm/code_samples/backprop_neural_net.py#L17)

#### Neuron Transfer
Once a neuron is activated, we need to transfer the activation to see what the neuron output actually is. Different transfer functions can be used. It's traditional to use the sigmoid activation function or tanh function to transfer outputs. More recently, the rectifier transfer function has been popular with large deep learning networks.

We will define a [sigmoid](https://github.com/ngacho/tensor-flow-basics/blob/02c908c149324c2b2c83563453207b4585ec4d77/tf-tutorial-mlm/code_samples/backprop_neural_net.py#L23) and a [rectified linear activation function](https://github.com/ngacho/tensor-flow-basics/blob/02c908c149324c2b2c83563453207b4585ec4d77/tf-tutorial-mlm/code_samples/backprop_neural_net.py#L26) for transfer and compare outcomes.


#### Forward propagation.
Forward propagating an input is straight forward. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer.

The code sample can be found [here](https://github.com/ngacho/tensor-flow-basics/blob/aa175a02d41c6034aaf9ab2650263e1e2b961705/tf-tutorial-mlm/code_samples/backprop_neural_net.py#L39)

## Back Propagate Error.
Backpropagation algorithm is named for the way in which weights are trained.

Error is calculated between the expected outputs and the outputs forward propagated from the network. These errors are then propagated backward through the network from the output layer to the hidden layer, assigning blame for errors and updating the weights as they go.

Two sections to achieve backpropagation
1. [Transfer Derivative](#transfer-derivative)
2. [Error Backpropagation](#error-backpropagation)

### Transfer Derivative
Given an output value from a neuron, we need to calculate it's slope.

With the sigmoid transfer function, the derivative is given by
```
derivative = return output * (1.0 - output)
```

For a ReLU, the derivative is given by
```
derivative = if output < 0:
                return 0
            else if output > 0
                return 1
            else 
                return undefined
```

### Error Backpropagation
The first step is to calculate the error for each output neuron, this will give us our error signal (input) to propagate it backwards through the network.

Error for a given neuron can be calculated as:
```
error = (output - expected) * derivative(output)
```

Where expected is the expected output value for the neuron, output is the output value for the neuron and derivative calculates the slope of the neuron's output value, as shown above.

The error calculation is used for neurons in the output layer. The expected value is the class value itself. In the hidden layer, things are a little more complicated.

The error signal for a neuron in the hidden layer is calculated as the weighted error of each neuron in the output layer. Think of the error traveling back along the weights of the output layer to the neurons in the hidden layer.

The back-propagated error signal is accumulated and then used to determine the error for the neuron in the hidden layer, as follows:
```
error = (weight_k * error_j) * derivative(output)
```

Where error_j is the error signal from the jth neuron in the output layer, weight_k is the weight that connects the kth neuron to the current neuron and output is the output for the current neuron.

