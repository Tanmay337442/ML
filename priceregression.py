import keras
from tensorflow.keras.datasets import boston_housing
import numpy as np

# Load Boston housing dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

# Normalize the data so model does not have to adapt to different scales
# not min-max normalization since it is sensitive to outliers
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean # center the data around 0
train_data /= std # scale the data to unit variance (units of standard deviation so data is in the same scale)
test_data -= mean
test_data /= std

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

# K-fold cross-validation since the dataset is small
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] # set apart data for validation
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples] # set apart targets for validation

    # combine all data and targets before and after validation fold
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                          train_data[(i + 1) * num_val_samples:]],
                                          axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                          train_targets[(i + 1) * num_val_samples:]],
                                          axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))

num_epochs = 500
all_mae_histories = []

for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] # set apart data for validation
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples] # set apart targets for validation

    # combine all data and targets before and after validation fold
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                          train_data[(i + 1) * num_val_samples:]],
                                          axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                          train_targets[(i + 1) * num_val_samples:]],
                                          axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# Plotting the validation MAE history - all 500 epochs
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Plotting the validation MAE history after 10 epochs (first 10 epochs have high error)
truncate_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncate_mae_history) + 1), truncate_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Retrain the model with all data
model = build_model()
model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mse_score)
print(test_mae_score)
predictions = model.predict(test_data)
print(predictions[0])