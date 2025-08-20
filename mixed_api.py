from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np

# can mix Sequential, Functional, and Model subclassing APIs
# subclassed model/layer in Functional model
class Classifier(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        # define activation function and number of units based on number of classes
        if num_classes == 2:
            num_units = 1
            # for binary classification, use sigmoid activation
            activation = 'sigmoid'
        else:
            num_units = num_classes
            # for multi-class classification, use softmax activation
            activation = 'softmax'
        # define dense layer with number of units and activation function
        self.dense = layers.Dense(num_units, activation=activation)

    def call(self, inputs):
        return self.dense(inputs)

# use Classifier in a Functional model
inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation='relu')(inputs)
outputs = Classifier(num_classes=10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

# can also use Functional model in a subclassed layer/model
inputs = keras.Input(shape=(64,))
outputs = layers.Dense(1, activation='sigmoid')(inputs)
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)

# subclassed model
class MyModel(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(64, activation='relu')
        self.classifier = binary_classifier
    
    # define the call method to specify the forward pass
    def call(self, inputs):
        # apply dense layer to inputs
        features = self.dense(inputs)
        # apply classifier to features - 64 features to 1 output
        return self.classifier(features)
    
model = MyModel()