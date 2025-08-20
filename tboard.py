from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

# create a simple MNIST model using the Functional API
def get_mnist_model():
    # use comma otherwise number in parentheses
    inputs = keras.Input(shape=(28*28,))
    features = layers.Dense(512, activation='relu')(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# load data from MNIST dataset: 28x28 grayscale images of handwritten digits (0-9)
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28*28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255
# set aside validation data
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

# get and compile model
model = get_mnist_model()
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# use TensorBoard callback
tensorboard = keras.callbacks.TensorBoard(log_dir='/Users/tanmay/Desktop/Code/Python/ML')

# pass callback to model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels), callbacks=[tensorboard])

# load saved model
model = keras.models.load_model('checkpoint_path.keras')

# locally host using Tensorboard:
# tensorboard --logdir /Users/tanmay/Desktop/Code/Python/ML