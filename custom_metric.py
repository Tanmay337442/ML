import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

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

# create model
model = get_mnist_model()
# compile model with optimizer, loss function, and metrics
# sparse since labels are integers (0-9)
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# fit method to train model, provide validation data for monitoring performance on unseen data
model.fit(train_images, train_labels, epochs=3,
          validation_data=(val_images, val_labels))
# evaluate model on test data for loss and metrics
test_metrics = model.evaluate(test_images, test_labels)
# compute classification probabilities for test images
predictions = model.predict(test_images)

# customize with custom metrics
# measure difference between performance on training and test data

# subclassed metric - has internal state stored in TensorFlow variables, but not updated by backpropagation
class RootMeanSquaredError(keras.metrics.Metric):
    # define state variables in constructor - use add_weight to create variables that are tracked by the metric
    def __init__(self, name='rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name='mse_sum', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros', dtype='int32')
    
    # implement state update logic
    # y_true is targets for one batch, y_pred is predictions for one batch
    # sample_weight not used here
    def update_state(self, y_true, y_pred, sample_weight=None):
        # categorical predictions and integer labels
        # convert y_true to one-hot encoding
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        # calculate squared error for this batch
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        # add squared error to the running total
        self.mse_sum.assign_add(mse)
        # update total number of samples seen so far
        num_samples = tf.shape(y_pred)[0]
        # update the total sample count
        self.total_samples.assign_add(num_samples)
    
    # implement result calculation logic
    def result(self):
        # calculate root mean squared error
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))
    
    # reset metric state without reinstantiating the object
    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)

model = get_mnist_model()

# use new metric when compiling model
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', RootMeanSquaredError()])
model.fit(train_images, train_labels, epochs=3,
          validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)