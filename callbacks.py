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

# customize with callbacks to fit method to schedule actions at specific points during training

# callbacks allow dynamic control over training process based on current state of training
# callback - object passed to model in call to fit method, called by model at various points during training
# has access to all available data about state of model and performance
# can take action: interrupt training, save model, load different weight set, etc.
# Model checkpointing - saves current state of model at different points during training
# Early stopping - interrupts training if validation loss no longer improves, saves best model from training
# Dynamically adjusting value of parameters during training e.g. learning rate of optimizer
# Logging training/validation metrics during training, visualizing representations learned by model as they are updated e.g. fit progress bar
# Keras.callbacks includes ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger

# EarlyStopping with ModelCheckpoint
# list of callbacks - interrupt training 2 epochs after best accuracy and save model with minimum loss
callbacks_list = [
    # interrupt training when improvement stops
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', # monitor validation accuracy
        patience=2 # interrupt training when accuracy stops improving for 2 epochs
    ),
    # save current weights after every epoch
    keras.callbacks.ModelCheckpoint(
        filepath='checkpoint_path.keras', # path to destination model file
        monitor='val_loss',
        save_best_only=True # overwrite model file only if val_loss improves - keep best model in training
    )
]

model = get_mnist_model()
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# pass callbacks to model with callbacks argument in fit method call
model.fit(train_images, train_labels, epochs=10, callbacks=callbacks_list, validation_data=(val_images, val_labels))

# load saved model
model = keras.models.load_model('checkpoint_path.keras')

# custom callbacks
# subclass keras.callbacks.Callback and implement any of the following methods:
# on_epoch_begin(epoch, logs) - called at start of every epoch
# on_epoch_end(epoch, logs) - called at end of every epoch
# on_batch_begin(batch, logs) - called before processing each batch
# on_batch_end(batch, logs) - called after processing each batch
# on_train_begin(logs) - called at start of training
# on_train_end(logs) - called at end of training
# logs dictionary contains info about previous batch/epoch/training run (training/validation metrics etc.)

# custom callback - track loss per batch for an epoch
class LossHistory(keras.callbacks.Callback):
    # initialize list to store batch losses
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    # add loss to list of batch losses
    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get('loss'))
    
    # plot batch loss for current epoch
    def on_epoch_end(self, epoch, logs):
        # clear figure
        plt.clf()
        # plot loss of each batch
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses, label='Training loss for each batch')
        # set axis labels
        plt.xlabel(f'Batch (epoch {epoch})')
        plt.xlabel('loss')
        # draw legend
        plt.legend()
        # save figure
        plt.savefig(f'plot_at_epoch_{epoch}')
        # reset batch loss list for next epoch
        self.per_batch_losses = []

# get and compile model
model = get_mnist_model()
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# fit model with LossHistory callback - default batch size of 32
model.fit(train_images, train_labels, epochs=10, callbacks=[LossHistory()], validation_data=(val_images, val_labels))