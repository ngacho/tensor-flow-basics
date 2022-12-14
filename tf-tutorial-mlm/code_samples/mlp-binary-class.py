from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# load dataset
dataset_path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv"
df = read_csv(dataset_path, header=None)

# split into input and output columns
x, y = df.values[:,:-1], df.values[:,-1]
# x an array is a value between -1 and 1
# y is an array of labels


# ensure all x data are floating poitn values
x = x.astype('float32')

# encode strings into integers(0s == b and 1s == g)
y = LabelEncoder().fit_transform(y)

# split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

# determine the number of input features
n_features = x_train.shape[1]

# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=0)

# evaluate the model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy : %.3f' %acc)

# make a prediction
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
y_hat = model.predict([row])
print('Predicted : %.3f' %y_hat)