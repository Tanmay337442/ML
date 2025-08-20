import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# load data from MNIST dataset: 28x28 grayscale images of handwritten digits (0-9)
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28*28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28*28)).astype('float32') / 255
# set aside validation data
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

loss_fn = keras.losses.SparseCategoricalCrossentropy()
# metric object used to track avg per-batch loss during training and evaluation
loss_tracker = keras.metrics.Mean(name='loss')

class CustomModel(keras.Model):
    # override train_step method
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            # forward pass with training=True
            # self since model is this class
            predictions = self(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # update loss tracker metric which tracks avg loss
        loss_tracker.update_state(loss)
        # return avg loss by querying loss tracker metric
        return {'loss': loss_tracker.result()}
    
    # allows model to automatically call reset_state on metrics at start of each epoch and start of evaluate call
    @property
    def metrics(self):
        return [loss_tracker] # list metrics to reset across epochs

# build model
inputs = keras.Input(shape=(28*28,))
features = layers.Dense(512, activation='relu')(inputs)
features = layers.Dropout(0.5)(features)
outputs = layers.Dense(10, activation='softmax')(features)
model = CustomModel(inputs, outputs)

# compile model and train using fit method
model.compile(optimizer=keras.optimizers.RMSprop())
model.fit(train_images, train_labels, epochs=3)

# after calling compile model has access to:
# self.compiled_loss - loss function passed to compile()
# self.compiled_metrics - wrapper for list of metrics passed - allows calling self.compiled_metrics.update_state()
# to update all metrics at once
# self.metrics - list of metrics passed to compile() - includes metric tracking loss

class CustomModel(keras.Model):
    # override train_step method
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            # forward pass with training=True
            # self since model is this class
            predictions = self(inputs, training=True)
            # compute loss via self.compiled_loss - deprecated
            loss = self.compiled_loss(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # update model metrics via self.compiled_metrics - deprecated
        self.compiled_metrics.update_state(targets, predictions)
        # return dict mapping metric names to current values
        return {'loss': loss, **{m.name: m.result() for m in self.metrics}}

# build model
inputs = keras.Input(shape=(28*28,))
features = layers.Dense(512, activation='relu')(inputs)
features = layers.Dropout(0.5)(features)
outputs = layers.Dense(10, activation='softmax')(features)
model = CustomModel(inputs, outputs)

# compile model and train using fit method
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_images, train_labels, epochs=3)