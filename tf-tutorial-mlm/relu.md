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

## Rectified Linear Activation Function.
In order to use stochastic gradient descent with [backpropagation of errors](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/) to train deep neural networks, an activation function, but is, in fact, a nonlinear function allowing complex relationships in the data to be learned.

The function must also provide more sensitivity to the activation sum input and avoid easy saturation.

A node or unit that implements this activation function is referred to as rectified linear activation unit (RELU). Often, networks that use the rectifier functions for the hidden layers are referred to as rectified networks.

A Rectified Linear Activation function is a simple calculation that returns the value provided as input directly, or the value 0.0 if the input is 0.0 or less

```
if input > 0:
    return input
else:
    return 0.
```

or 

```
g(x) = max(0, x)
```

The function is linear for values greater than 0, meaning it has a lot of desirable properties of a linear activation function when training a neural network using backpropagation. Yet, it is a nonlinear function as negative values are always output as 0.

### Coding a Rectified Linear Activation function.

In python
```
def rectified(x):
    return max(0.0, x)
```

A sample relu plot generating integers from -10 to 10 and plotting a relu activation function can be found [here]()

The derivative of the rectified linear function is also easy to calculate. Recall that the derivative of the activation function is required when updating the weights of a node as part of the backpropagation of error. The derivative of the function is the slope. 0 for negative valus and 1 for positive values.


### Advantages of the Rectified Linear Activation function.
- Computational simplicity. 
    - Only requires a max function unlike tanh and sigmoid which require use of an exponential function.

- Representational Sparsity.
    - an important benefit is that it's capable of outputting a true zero value.
    - negative inputs that can output ztrue zero values allowing the activation of hidden layers in neural networks to contain one or more zero values. This is called sparse representation and is a desirable property in representational learning as it can accelerate learning and simplify the model.
- Linear Behavior
    - a neural network is easier to optimize when its behavior is linear or close to linear.
- Train Deep Networks


### Tips while using the rectified linear activation function.

When in doubt, start with ReLU in your neural network.

Use ReLU with MLPs, CNNs, but probably not RNNs.

Try a smaller bias input Value. The bias is the input on the node that has a fixed value. The biasa has the effect of shifting the activation function and it is traditionally set to the bias input value of 1.0.
When using ReLU, consider setting the bias to a small value, such as 0.1

Use "He Weight Initialization"
Before training a neural network, the weights of the network must be initialized to small random values.
When using ReLU in your network and initializing weights to small random values centered on zero, then by default half of the units in the networks will output a zero value. He initialization (specifically +/- sqrt(2/n)) whenre n is the number of nodes in the prior layer known as the fan-in.

Scale Input Data
It's good practice to scale input data prior to using a neural network. 

This may involve standardizing variables to have a zero mean and unit variance or normalizing each value to the scale 0-to-1.

Use Weight Penalty.
By design, output from ReLU is unbounded in the positive domain. This means, that in some cases, the output can continue to grow in size. As such, it may be a goood idea to use a form of weight regularization.

Key among the limitations of ReLU is the case where large weight updates can mean that the summed input to the activation function is always negative, regardless of the input to the network. This means that a node with this problem will forever output an activation value of 0.0. This is referred to as "dying ReLU".

An extension can be to relax the non-linear output of the function to allow small negative values in some ways.

Other extensions include:
- Leaky ReLU (lReLU)that modifies the function to allow small negative values when the input is less than zero.

- Exponential Linear Unit (ELU) that uses a parameterized exponential function to transition from the positive to small negative values.

- Parametric ReLU learns parameters that control the shape and leaky-ness of the function.


