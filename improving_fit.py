import keras
from keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Common problems before overfitting occurs

# 1. Training does not start (gets stuck at low accuracy)

# load MNIST dataset
train_images, train_labels = mnist.load_data()[0]
# Preprocess the data as 60000 rows of 784 pixels each
# and normalize pixel values to be between 0 and 1
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255

# build model
model = keras.models.Sequential([
    layers.Dense(512, activation='relu'), # hidden layer with 512 neurons
    layers.Dense(10, activation='softmax') # output layer with 10 neurons for 10 classes (digits 0-9)
])

# Example of model with too high learning rate

# compile model
model.compile(optimizer=keras.optimizers.RMSprop(1.), # using RMSprop optimizer with learning rate 1.0
                # for faster convergence but leads to accuracy getting stuck at 25%
                loss='sparse_categorical_crossentropy', # using sparse categorical crossentropy loss function
                # since labels are integers (0-9) and not one-hot encoded
                metrics=['accuracy'])

# fit model to training data
# using 20% of the training data for validation
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# Example of model with lower learning rate

# compile model
model.compile(optimizer=keras.optimizers.RMSprop(1e-2), # using RMSprop optimizer with learning rate 0.01
                loss='sparse_categorical_crossentropy', # using sparse categorical crossentropy loss function
                # since labels are integers (0-9) and not one-hot encoded
                metrics=['accuracy'])

# fit model to training data
# using 20% of the training data for validation
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# 2. Training starts but no generalization occurs (similar to random classifier)

# 3. Training and validation accuracy go up but validation accuracy stalls (not overfitting yet)

# Example of model with too few neurons (not enough representational power)

# build model with output layer - applies linear transformation to input data
# and softmax activation function to compute probabilities for each class
model = keras.models.Sequential([
    layers.Dense(10, activation='softmax') # output layer with 10 neurons for 10 classes (digits 0-9)
])

# compile model
model.compile(optimizer='rmsprop', # using RMSprop optimizer
                loss='sparse_categorical_crossentropy', # using sparse categorical crossentropy loss function
                metrics=['accuracy'])

history_small_model = model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_split=0.2)

# build model
model = keras.models.Sequential([
    layers.Dense(96, activation='relu'), # hidden layer with 96 neurons
    layers.Dense(96, activation='relu'), # hidden layer with 96 neurons
    layers.Dense(10, activation='softmax') # output layer with 10 neurons for 10 classes (digits 0-9)
])

# compile model
model.compile(optimizer='rmsprop', # using RMSprop optimizer
                loss='sparse_categorical_crossentropy', # using sparse categorical crossentropy loss function
                metrics=['accuracy'])

history_large_model = model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_split=0.2)

val_loss_small = history_small_model.history['val_loss']
val_loss_large = history_large_model.history['val_loss']
epochs = range(1, len(val_loss_small) + 1)
plt.plot(epochs, val_loss_small, 'b--', label='Validation loss (small model)') # plot validation loss for small model
plt.plot(epochs, val_loss_large, 'b-', label='Validation loss (large model)') # plot validation loss for large model
plt.title('Validation loss for small and large models') # set title for the plot
plt.xlabel('Epochs') # set x-axis label
plt.ylabel('Validation loss') # set y-axis label
plt.legend() # show legend
plt.show() # display the plot