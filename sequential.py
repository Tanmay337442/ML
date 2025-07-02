import keras
from keras import layers
import numpy as np

# Sequential models are linear stacks of layers
# can be used for simple models with a single input and output
# build a simple sequential model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# build model incrementally
model = keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# model.weights returns ValueueError - model has not been built yet

model.build(input_shape=(None, 3))  # specify input shape with any batch size
model.weights  # now it works
model.summary()  # print model summary for debugging

# can name models and layers
model = keras.Sequential(name='test_model')
model.add(layers.Dense(64, activation='relu', name='first_layer'))
model.add(layers.Dense(10, activation='softmax', name='last_layer'))
model.build((None, 3))
model.summary()

# cannot print summary until model is built
# declare model with input shape
# common debugging step when layers transform input shape in complex ways e.g. convolutional layers
model = keras.Sequential()
model.add(keras.Input(shape=(3,)))  # specify input shape - shape of each sample not batch
model.add(layers.Dense(64, activation='relu'))
model.summary()  # now it works

# can also use Functional API to build models - multiple inputs and outputs, non-linear topology (does not run sequentially)
# sumbolic tensor - does not contain data but encodes shape and dtype of actual tensors
inputs = keras.Input(shape=(3,), name='test_input')
# Keras layers are callable with __call__ - takes symbolic tensor as input and returns symbolic tensor as output with
# updated shape and dtype
features = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10, activation='softmax')(features)
model = keras.Model(inputs=inputs, outputs=outputs)