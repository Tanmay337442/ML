import keras
from keras import layers
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

# train_data is a list of reviews, which are lists of word indices (0-9999)
(train_data, train_labels) = imdb.load_data(num_words=10000)[0]

# multi-hot encoding
def vectorize_sequences(sequences, dimension=10000):
    # set results to have each review as a row with the number of most common words in the vocabulary
    results = np.zeros((len(sequences), dimension))
    # i is index of the review, sequence is the list of words in the review
    for i, sequence in enumerate(sequences):
        # set the words in the review to 1 in the results array
        # i is the index of the review, sequence is the list of words in the review as indices in the vocabulary
        results[i, sequence] = 1
    return results

train_data = vectorize_sequences(train_data)

# layer_output *= np.random.randint(0, high=2, size=layer_output.shape) - at training time
# layer_output *= 0.5 - at testing time

# layer_output *= np.random.randint(0, high=2, size=layer_output.shape) - at training time
# layer_output /= 0.5 - at training time

model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),  # dropout layer with 50% dropout rate
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),  # dropout layer with 50% dropout rate
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history_dropout = model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4)