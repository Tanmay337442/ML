import tensorflow as tf
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

# supervised-learning training step
# def train_step(inputs, targets):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, training=True)
#         loss = loss_fn(targets, predictions)
#     gradients = tape.gradient(loss, model.trainable_weights)
#     optimizer.apply_gradients(zip(gradients, model.trainable_weights))

# low-level metric usage
metric = keras.metrics.SparseCategoricalAccuracy()
targets = [0, 1, 2]
predictions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
metric.update_state(targets, predictions)
current_result = metric.result()
print(f'Result: {current_result:.2f}')

# track average of scalar value e.g. loss
values = [0, 1, 2, 3, 4]
mean_tracker = keras.metrics.Mean()
for value in values:
    mean_tracker.update_state(value)
print(f'Mean of values: {mean_tracker.result():.2f}')

# get model
model = get_mnist_model()
# prepare loss function, optimizer, list of metrics to monitor, Mean metric tracker for loss avg
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.RMSprop()
metrics = [keras.metrics.SparseCategoricalAccuracy()]
loss_tracking_metric = keras.metrics.Mean()

# training step on one batch
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        # forward pass with training=True
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    # backward pass with model.trainable_weights
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    # keep track of metrics
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()
    # keep track of loss avg
    loss_tracking_metric.update_state(loss)
    logs['loss'] = loss_tracking_metric.result()
    return logs

# reset method to be run at start of each epoch and before evaluation
def reset_metrics():
    for metric in metrics:
        metric.reset_state()
    loss_tracking_metric.reset_state()

# organize training dataset into batches and set number of epochs
training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
training_dataset = training_dataset.batch(32)
epochs = 3

# run training loop
for epoch in range(epochs):
    reset_metrics()
    for inputs_batch, targets_batch in training_dataset:
        logs = train_step(inputs_batch, targets_batch)
    print(f'Results at the end of epoch {epoch}')
    for key, value in logs.items():
        print(f'...{key}: {value:.4f}')

# evaluate model
def test_step(inputs, targets):
    # now training=False
    predictions = model(inputs, training=False)
    loss = loss_fn(targets, predictions)
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs['val_' + metric.name] = metric.result()
    loss_tracking_metric.update_state(loss)
    logs['val_loss'] = loss_tracking_metric.result()
    return logs

# organize validation set into batches and reset metrics
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32)
reset_metrics()

# evaluation loop
for inputs_batch, targets_batch in val_dataset:
    test_step(inputs_batch, targets_batch)

# print results
print('Evaluation results:')
for key, value in logs:
    print(f'...{key}: {value:.4f}')

# fit() and evaluate() support more features including large-scale distributed computation and performance optimizations
# TensorFlow function compilation - compile into computation graph which can be globally optimized instead of executing
# line by line (eager evaluation)

# test step with @tf.function decorator
@tf.function
def test_step(inputs, targets):
    # now training=False
    predictions = model(inputs, training=False)
    loss = loss_fn(targets, predictions)
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs['val_' + metric.name] = metric.result()
    loss_tracking_metric.update_state(loss)
    logs['val_loss'] = loss_tracking_metric.result()
    return logs

# organize validation set into batches and reset metrics
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32)
reset_metrics()

# evaluation loop
for inputs_batch, targets_batch in val_dataset:
    test_step(inputs_batch, targets_batch)

# print results
print('Evaluation results:')
for key, value in logs:
    print(f'...{key}: {value:.4f}')