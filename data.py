# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
import numpy as np

# Globals:
BUFFER_SIZE = 60000
BATCH_SIZE = 256

def get_dataset():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                        1).astype('float32')
    # We are normalizing the images to the range of [0, 1]
    # train_images = np.zeros_like(train_images[:BATCH_SIZE,:,:,:], dtype = np.int)
    train_images = np.concatenate([np.zeros([BATCH_SIZE, 3,4,1], dtype=np.float), np.ones([BATCH_SIZE, 1,4,1], dtype=np.float)], axis = 1)
    # train_images = np.round(train_images / 256).astype(np.int)

    # return train_images
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)