import tensorflow as tf
from matplotlib import pyplot as plt

# import dataset.
mnist = tf.keras.datasets.mnist
# test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))

plt.show()

# stack layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# model returns a vector of logits or log-odds scores, one for each class.
predictions = model(x_train[:1]).numpy()

# The tf.nn.softmax function converts these logits to probabilities for each class: 
tf.nn.softmax(predictions).numpy()

# a loss function for training using losses.SparseCategoricalCrossentropy, which takes a vector of logits and a True index and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This loss is equal to the negative log probability of the true class: 
# The loss is zero if the model is sure of the correct class.

# efore you start training, configure and compile the model using Keras Model.compile. 
# Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, 
# and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)