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

![Sigmoid Activation function sample](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)
### The hyperbolic tangent function

The sigmoid activation function, also called the logistic function, is a similar shaped nonlinear activation function that outputs values between -1.0 and 1.0. It has a higher range when compared to the sigmoid activation function.

![Hyperbolic tangent activation function](https://i.stack.imgur.com/vxfdW.png)

### Limitations of Sigmoid and Tanh Activation Functions.
A general problem with the sigmoid and tanh functions is that they saturate. This means that large values snap to 1, and small values snap to -1 hyperbolic tangnet functions and 0 for sigmoid functions respectively. 

Furthermore, the functions aare only really sensitive to changes around the mid-point of their input, 0.5 for sigmoid, and 0.0 for hyperbolic tangent functions.

Limited sensitivity and saturation happen regardless of whether the summed activation from node provided as input contains useful information or not. Once saturated, it becomes challenging for the learning algorithm to continue to adapt the weights to improve performance of the model.


**The vanishing gradient problem**
Vanishing gradient problem mostly occurs during the backpropagation when the value of the weights are changed.

Neural networks are trained using a stochastic gradient descent. This involves first calculating the prediction error made by the model, and using the error to estimate a gradient used to update each weight in the network so that less error is made next time. This error gradient is propagated backward through network from the output layer to the input layer.

It's desirable to train neural networks with many layers as the addition of more layers increases the capacity of the network, making it capable of learning a large training dataset and efficiently representing more complex mapping functions from inputs to outputs.

A problem with training networks with many layers is that the gradient diminishes dramatically as it is propagated backward through the network. The error may be so small by the time it reaches layers close to the input of the model it may have very little effect hence the __vanishing gradient problem__

Layers deep in large networks using this nonlinear activation functions fail to receive useful gradient information. Error is back propagated through the network and used to update the weights. The amount of error decreases dramatically with each additional layer through which it is propagated, given the derivative of the chosen activation function.

For the sigmoid function, the range is 0 to 1. We know that the maximum threshold value is 1, and the minimum value is 0. So when we increase the input values, the predicted output must lie near to the upper threshold value which is 1.

When neuron outputs are very small, the patterns created during optimization will be smaller and smaller towards the upper layers, making the learning process very slow, and make them converge to their optimum.

The vanishing gradient problem may be manifest in a Multilayer Perceptron by a slow rate of improvement of a model during training and perhaps premature convergence-continued training does not result in further improvement.

### Rectified Linear Activation Function.
