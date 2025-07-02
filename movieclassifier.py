from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# keep the top 10,000 most frequent words
# train_data and test_data are lists of reviews, which are lists of word indices (0-9999)
# train labels and test labels are 0s and 1s - binary for positive and negative
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def decode_review(review_num):
    # decode word indices back to words
    word_index = imdb.get_word_index() # maps words to indices
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # maps indices to words
    # indices are offset by 3 because 0, 1, and 2 are reserved for "padding", "start of sequence", and "unknown" in train data
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[review_num]]) # decode the first review
    print(decoded_review) # print the decoded review

def vectorize_sequences(sequences, dimension=10000):
    # turn lists of reviews into a 2D numpy array of shape (len(sequences), dimension)
    # each index that is set to 1 corresponds to a word in the review
    results = np.zeros((len(sequences), dimension)) # create an array of zeros
    for i, sequence in enumerate(sequences): # for each review
        results[i, sequence] = 1. # set the indices of the words in the review to 1
    return results

# vectorize the training and test data
x_train = vectorize_sequences(train_data) # vectorize the training data
x_test = vectorize_sequences(test_data) # vectorize the test data
# vectorize the labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# build the model
model = keras.Sequential([
    layers.Dense(16, activation='relu'), # hidden layer with 16 neurons
    layers.Dense(16, activation='relu'), # hidden layer with 16 neurons
    layers.Dense(1, activation='sigmoid') # output layer with 1 neuron
])

model.compile(optimizer='rmsprop', # optimizer
              loss='binary_crossentropy', # loss function
              metrics=['accuracy']) # metrics to track

# set aside 10,000 samples from the training data for validation
x_val = x_train[:10000] # validation data
partial_x_train = x_train[10000:] # training data
y_val = y_train[:10000] # validation labels
partial_y_train = y_train[10000:] # training labels

# use remaining 40,000 samples for training
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss') # plot training loss
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') # plot validation loss
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc') # plot training accuracy
plt.plot(epochs, val_acc, 'b', label='Validation acc') # plot validation accuracy
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()