# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

# five-step model-life-cycle
"""
1. Define the model
2. Compile the model
3. Fit the model
4. Evaluate the model
5. Make predictions.
"""

# Define the model.
"""
Selecting the type of model you need
Choose architecture or network topology.

Using keras, this means:
    - defining model.
    - configuring each layer with a number of nodes and an activation function.
    - connecting layers together into a cohesive model.
"""

# Compile the model
"""
Requires selecting a loss function you want to optimize
    - a mean squared error
    - cross-entropy
Require selection of an algorithm to perform the optimization procedure
    - stochastic gradient descent
    - or its modern variation eg adam.
Select any performance metrics to keep track of during the model training purpose

Using keras api, this means
    - calling a function to compile model with chosen configuration.
    - optimizer can be specified as string for a known optimizer class, e.g sgd for stochastic gradient descent
        or configure optimizer class and use that

        ```opt = SGD(learning_rate=0.01, momentum=0.9)
            model.compile(optimizer=opt, loss='binary_crossentropy')
        ```

    - three common loss functions:
        binary_crossentropy for binary classification
        sparse_categorical_crossentropy for multiclass classification
        mse for mean squared error for regression.

    - metrics are defined as list of strings for known metric functions 
    or list of functions to call to evaluate predictions
"""

# fit the model.
"""
Fitting a model requires we select the training configuration eg 
    - number of epochs (loops thru training dataset)
    - batcg size (number of samples in epoch used to estimate model error)

Training applies the chosen optimization algo to minimize the chosen loss
 function and update the model using backpropagation of error algorithm.

 Could take many hours depending on 
    complexity of model, hardware, and size of training dataset.


Using keras, 
    call function to perform training process
    ```
        model.fit(X, y, epochs=100, batch_size=32)
    ```

"""


# Evaluate the model
"""
Evaluating a model requires you choose a holdout dataset used 
 to evaluate model. This is not used in training set.

Speed is inversely proportional to amount of data you want 
 to use for evaluation

call a function with holdout dataset and getting a loss and other metrics

    ```
        loss = model.evaluate(X, y, verbose=0)
    ```
"""


# Make a prediction.
"""
Get new data for which prediction is required.

Call function to make a prediction of class label, probability or numerical values.

    ```
        yhat = model.predict(X)        
    ```

"""


# Main ways to use tf.keras api to build models: sequential and functional.
"""
    - sequential model api (simple)
    - functional model api (advanced)
"""

# Sequential model api.
"""
Called sequential because it involves defining a sequential class and adding layers
 one by one in a linear manner, from input to output.

Using keras:
    ```
        This model defines a sequential MLP model that accepts eight input, 
         has one hidden layer with 10 nodes, then an output layer with
         with one node to predict a numerical value.

        # define model
        model = Sequential()
        Input shape defines the visible layer of the network.
        Model expects input for one sample to be a vector of eight numbers.
        model.add(Dense(10, input_shape=(8,)))
        model.add(Dense(1))
    ```

"""

# Functional Model API.
"""
Involves explicitly connecting the output of one layer to the input of another
Each connection is specified.

An input layer must be defined via the input class,
 shape of an input sample is specified.
 we retain a reference to the input layer when defining the model.
 

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

"""