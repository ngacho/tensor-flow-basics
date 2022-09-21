# An Introduction to RELU via [this tutorial](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/).

A rectified linear activation function (RELU) is a short piecewise linear function that will output the input directly if positive, otherwise it will output zero. It has become the default activation function for many types of neural networks because a model that uses it to train often achieves better performance.

## Activation functions 
A neural network is comprised of layers of nodes and learns to map examples of inputs to outputs.

In a neural network, the activation function is responsible for transforming the summed weighted input from the node into the activation of the node or output for that input. The function defines the output of that node given an input or a set of inputs. A standard integrated circuit can be seen as a digitial network of activation functions that can be 0 or 1 depending on input.

For a given node, the inputs are multiplied by the weights in a node and summed together. This value is referred to as the summed activation of the node.

The summed activation is then transformed via an activation function and defines the specific output or _activation_ of the node.

The simplest activation function is referred to as the linear activation, where no transform is applied at all.

A network comprised of only linear activation functions is very easy to train, but cannot learn complex mapping functions.

They're still used in the output layer for network that predict a quantity.

Nonlinear activation functions are preferred as they allow the nodes to learn more complex structures in the data. Two widely used nonlinear activation functions are
- [sigmoid activation function.](#the-sigmoid-activation-function)
- [hyperbolic tangent function.](#the-hyperbolic-tangent-function)

### The sigmoid activation function

The sigmoid activation function, also called the logistic functon, is a popular activation function for neural networks. The input to the function is transformed into a value between 0.0 and 1.0. Inputs that are much larger than 1.0 are transformed to 1, similarly values much smaller than 0 are snapped to 0. The shape of the function for all possible inputs is an S-shape from zero up through 0.5 to 1.0.

### THe hyperbolic tangent function

The sigmoid activation function, also called the logistic function, is a similar shaped nonlinear activation function that outputs values between -1.0 and 1.0.

### Limitations of Sigmoid and Tanh Activation Functions.
