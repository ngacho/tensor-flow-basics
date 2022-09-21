# This folder contains code and notes from [this](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/) machine learning tutorial.

## Five step model life cycle.
1. [Define the model](#define-the-model)
2. [Compile the model](#compile-the-model)
3. [Fit the model](#fit-the-model)
4. [Evaluate the model](#evaluate-the-model)
5. [Make predictions.](#make-a-prediction)

### Define the model.
Selecting the type of model you need.<br>
Choose architecture or network topology.

Using keras, this means:<br>
- defining model
- configuring each layer with a number of nodes and an activation function.
- connecting layers together into a cohesive model.

### Compile the Model.
Requires selecting a loss function you want to optimize
- a mean squared error
- cross-entropy
Require selection of an algorithm to perform the optimization procedure
    - stochastic gradient descent
    - or its modern variation eg adam.

Select any performance metrics to keep track of during the model training purpose

Using keras api, this means
- calling a function to compile model with chosen configuration.
- optimizer can be specified as string for a known optimizer class, e.g sgd for stochastic gradient descent or configure optimizer class and use that
```
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy')
```

- three common loss functions:
    - binary_crossentropy for binary classification
    - sparse_categorical_crossentropy for multiclass classification
    - mse for mean squared error for regression.

- metrics are defined as list of strings for known metric functions or list of functions to call to evaluate predictions

### Fit the model.
Fitting a model requires we select the training configuration eg 
- number of epochs (loops thru training dataset)
- batcg size (number of samples in epoch used to estimate model error)

Training applies the chosen optimization algo to minimize the chosen loss function and update the model using backpropagation of error algorithm.

 Could take many hours depending on complexity of model, hardware, and size of training dataset.


Using keras, call function to perform training process
```
    model.fit(X, y, epochs=100, batch_size=32)
```

### Evaluate the model
Evaluating a model requires you choose a holdout dataset used 
 to evaluate model. This is not used in training set.

Speed is inversely proportional to amount of data you want 
 to use for evaluation

call a function with holdout dataset and getting a loss and other metrics
```
    loss = model.evaluate(X, y, verbose=0)
```

### Make a prediction.
Get new data for which prediction is required.

Call function to make a prediction of class label, probability or numerical values.
```
    yhat = model.predict(X)        
```

## Main ways to Use Keras API.
- [sequential model api (simple)](#sequential-model-api)
- [functional model api (advanced)](#functional-model-api)

### Sequential Model API
Called sequential because it involves defining a sequential class and adding layers
 one by one in a linear manner, from input to output.

Using keras:
This model defines a sequential Multilayer Perceptron (MLP) model that accepts eight input, has one hidden layer with 10 nodes, then an output layer with one node to predict a numerical value.

```
    # define model
    model = Sequential()
    # Input shape defines the visible layer of the network.
    # Model expects input for one sample to be a vector of eight numbers.
    model.add(Dense(10, input_shape=(8,)))
    model.add(Dense(1))
```

### Functional Model API
Involves explicitly connecting the output of one layer to the input of another
Each connection is specified.

An input layer must be defined via the input class, shape of an input sample is specified. We retain a reference to the input layer when defining the model.
 
```
    # define a layer
    x_in = Input(shape=(8,))
```

A fully connected layer can be connected to the input 
```
    x = Dense(10)(x_in)
```

Then connect to an output layer
```
    x_out = Dense(1)(x)    
```

A complete example
```
    # define the layers
    x_in = Input(shape=(8,))
    x = Dense(10)(x_in)
    x_out = Dense(1)(x)
    # define the model
    model = Model(inputs=x_in, outputs=x_out)
```


## Developing Deep Learning Models
Developing evaluating and predicting with deep learning models:
 - [Multilayer Perceptrons (MLP)](#multilayer-perceptrons)
 - Convolutional Neural Networks (CNN)
 - Recurrent Neural Networks (RNN)


### Multilayer Perceptrons
Standard fully connected neural network model

It comprises layers of nodes where each node is connected to all outputs from the previous layer, 
 and the output of each node is connected to all inputs for nodes in the next layer.

Created with one or more Dense layers.
Model is appropriate for tabular data, that is data as it looks in a table or spreadsheet with 
 one column for each variable and 
 one row for each variable.

 Three predictive modeling problems you may want to explore with an MLP:
- [binary classification](#binary-classification)
- multiclass classification
- regression

#### Binary Classification.
We will use the Ionosphere binary (two-class) classification dataset to demonstrate an MLP for binary classification.

This dataset involves predicting whether a structure is in the atmosphere or not, given radar returns.


