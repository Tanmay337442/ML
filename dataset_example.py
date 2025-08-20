import numpy as np
import tensorflow as tf
random_numbers = np.random.normal(size=(1000, 16))
# from_tensor_slices creates Dataset from np array or tuple/dict of np arrays
dataset = tf.data.Dataset.from_tensor_slices(random_numbers)
# single samples
for i, element in enumerate(dataset):
    print(element.shape)
    if i >= 2:
        break
# batch data with batch method
batched_dataset = dataset.batch(32)
for i, element in enumerate(batched_dataset):
    print(element.shape)
    if i >= 2:
        break
# other useful methods
# .shuffle(buffer_size) - shuffle elements within buffer
# .prefetch(buffer_size) - prefetch buffer of elements in GPU memory - better device utilization
# .map(callable)- applies arbitrary transformation to each element of dataset - function callable takes single element from dataset as imput

# reshape elements in dataset from (16,) to (4,4)
reshaped_dataset = dataset.map(lambda x: tf.reshape(x, (4, 4)))
for i, element in enumerate(reshaped_dataset):
    print(element.shape)
    if i >= 2:
        break