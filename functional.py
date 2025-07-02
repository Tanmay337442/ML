from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np
import pydot

# Functional API best for complex models - multiple inputs/outputs
# system to rank customer support tickets by priority and send to appropriate department
# inputs: title (text input), body (text input), tags (categorical input, one-hot encoded)
# outputs: priority score (scalar 0-1 sigmoid), department (softmax over departments)
vocabulary_size = 10000
num_tags = 100
num_departments = 4

# define model inputs
title = keras.Input(shape=(vocabulary_size,), name='title')
text_body = keras.Input(shape=(vocabulary_size,), name='text_body')
tags = keras.Input(shape=(num_tags,), name='tags')

# combine inputs into single tensor by concatenating them
features = layers.Concatenate()([title, text_body, tags])

# apply intermediate later to recombine input features into richer representations - 64 neurons
features = layers.Dense(64, activation='relu')(features)

# define model outputs
priority = layers.Dense(1, activation='sigmoid', name='priority')(features)
department = layers.Dense(num_departments, activation='softmax', name='department')(features)

model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])
plot_model(model, 'ticket_classifier_shapes.png', show_shapes=True)
# train model

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

# compile model
# loss functions:  mean squared error for priority (regression) and categorical crossentropy for department (classification)
# metrics: mean absolute error for priority and accuracy for department
model.compile(optimizer='rmsprop',
              loss=['mean_squared_error', 'categorical_crossentropy'],
              metrics=['mean_absolute_error', 'accuracy'])

# fit model to data over one epoch
model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1)

# evaluate model on data - returns loss and metrics
model.evaluate([title_data, text_body_data, tags_data],
               [priority_data, department_data])

# make predictions
priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])

# for unordered inputs, use dictionary to pass inputs by name
model.compile(optimizer='rmsprop',
              loss={'priority': 'mean_squared_error',
                    'department': 'categorical_crossentropy'},
              metrics={'priority': ['mean_absolute_error'],
                       'department': ['accuracy']})
# fit model with dictionary inputs
model.fit({'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
          {'priority': priority_data, 'department': department_data},
          epochs=1)

# evaluate model with dictionary inputs
model.evaluate({'title': title_data, 'text_body': text_body_data, 'tags': tags_data},
               {'priority': priority_data, 'department': department_data})

# make predictions with dictionary inputs
priority_preds, department_preds = model.predict(
    {'title': title_data, 'text_body': text_body_data, 'tags': tags_data})

# Functional model is explicit graph data structure - can inspect layer connections and reuse previous nodes
# used for model visualization and feature extraction

# visualize connectivity (topology) of model
plot_model(model, 'ticket_classifier.png')
plot_model(model, 'ticket_classifier_shapes.png', show_shapes=True)

