import keras
from keras import layers, regularizers
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

# Example of reducing/increasing network size to see how it affects validation accuracy

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

model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history_original = model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4)

model = keras.Sequential([
    layers.Dense(4, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history_small = model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4)

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history_large = model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4)

epochs = range(1, 21)
val_loss_original = history_original.history['val_loss']
val_loss_small = history_small.history['val_loss']
val_loss_large = history_large.history['val_loss']
plt.plot(epochs, val_loss_original, 'b-', label='Validation loss (original model)')
plt.plot(epochs, val_loss_small, 'b--', label='Validation loss (small model)')
plt.plot(epochs, val_loss_large, 'b-.', label='Validation loss (large model)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Example of adding L2 regularization to see how it affects validation accuracy

# build model with L2 regularization - adds 0.002 * weight_coeficient_value ** 2 to total loss
# to penalize large weights and prevent overfitting
# only added at training time, not for testing
model = keras.Sequential([
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation='relu'),
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history_l2 = model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4)

regularizers.l1(0.001)  # L1 regularization (not used in this example)
regularizers.l1_l2(l1=0.001, l2=0.001)  # simultaneous L1 and L2 regularization (not used in this example)

val_loss_l2 = history_l2.history['val_loss']
plt.plot(epochs, val_loss_original, 'b-', label='Validation loss (original model)')
plt.plot(epochs, val_loss_l2, 'b--', label='Validation loss (L2 regularization)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()