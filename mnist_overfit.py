import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

(train_images, train_labels), _ = mnist.load_data()

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255

 # add 784 random noise channels (pixels) after the original 784 channels
 # each row is an image, and this array is doubled in size from 784 to 1568
train_images_with_noise_channels = np.concatenate(
    [train_images, np.random.random(size=(60000, 28*28))], axis=1)
# add 784 zero channels after the original 784 channels
train_images_with_zeros_channels = np.concatenate(
    [train_images, np.zeros(shape=(60000, 28*28))], axis=1)

def get_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = get_model()
history_noise = model.fit(
    train_images_with_noise_channels, train_labels,
    epochs=10, batch_size=128, validation_split=0.2)

model = get_model()
history_zeros = model.fit(
    train_images_with_zeros_channels, train_labels,
    epochs=10, batch_size=128, validation_split=0.2)

val_acc_noise = history_noise.history['val_accuracy']
val_acc_zeros = history_zeros.history['val_accuracy']
epochs = range(1, len(val_acc_noise) + 1)
# noise channels contribute to overfitting, so validation accuracy is lower
plt.plot(epochs, val_acc_noise, 'b-',
         label='Validation accuracy with noise channels')
# zeros channels do not contribute to overfitting, so validation accuracy is higher
plt.plot(epochs, val_acc_zeros, 'b--',
            label='Validation accuracy with zeros channels')
plt.title('Effect of noise channels on validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# training on shuffled labels does not affect training accuracy since the model
# learns by overfitting to the training data but validation accuracy is the same as random chance
random_train_labels = train_labels[:]
np.random.shuffle(random_train_labels)

model = get_model()
history_random = model.fit(
    train_images, random_train_labels,
    epochs=10, batch_size=128, validation_split=0.2)

train_loss_random = history_random.history['loss']
plt.plot(epochs, train_loss_random, 'b-', label='Training loss with random labels')
val_acc_random = history_random.history['val_accuracy']
plt.plot(epochs, val_acc_random, 'b--',
         label='Validation accuracy with random labels')
plt.title('Effect of random labels on validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()