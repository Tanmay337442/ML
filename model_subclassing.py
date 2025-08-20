from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np

# Model subclassing allows for more flexibility in defining custom behavior
# can have call method use layers in loop, recursively call
# subclassing - layer connectivity hidden in call method
# cannot call summary method, cannot visualize model topology
# cannot access nodes of graph of layers for feature extraction
# similar to Layer subclassing
# layer is building block of model
# model is top-level object, trained and exported for inference - fit, evaluate, predict

# system to rank customer support tickets by priority and send to appropriate department
class CustomerTicketModel(keras.Model):
    # initialize model with number of departments
    def __init__(self, num_departments):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation='relu')
        self.priority_scorer = layers.Dense(1, activation='sigmoid')
        self.department_classifier = layers.Dense(num_departments, activation='softmax')
    
    # method for calling model on inputs - __call__ is called when model is called
    # __call__ does preprocessing, then calls call method, finally returns outputs
    def call(self, inputs):
        title = inputs['title']
        text_body = inputs['text_body']
        tags = inputs['tags']

        # combine inputs into single tensor by concatenating them
        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department

# input dimensions
vocabulary_size = 10000
num_tags = 100
num_departments=4

# set number of samples
num_samples = 1280

# dummy input data
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# dummy target data
priority_data = np.random.rand(num_samples, 1)  # a random priority score between 0 and 1 for each sample
# random one-hot encoded (0 or 1) department for each sample
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

# instantiate model
model = CustomerTicketModel(num_departments)
# forward pass without training
priority, department = model({
    'title': title_data, 'text_body': text_body_data, 'tags': tags_data})

# compile model
model.compile(optimizer='rmsprop',
              loss=['mean_squared_error', 'categorical_crossentropy'],
              metrics=['mean_absolute_error', 'accuracy'])

# fit model to data over one epoch
# structure of input data matches call method
# structure of target data matches output returned by call method
model.fit({'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
          [priority_data, department_data],
          epochs=1)

# evaluate model on data - returns loss and metrics
model.evaluate({'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
               [priority_data, department_data])

# make predictions
priority_preds, department_preds = model.predict({
    'title': title_data, 'text_body': text_body_data, 'tags': tags_data})