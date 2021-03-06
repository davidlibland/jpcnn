# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
import numpy as np
from jpcnn.image_utils import display_images

# Globals:

BUFFER_SIZE = 60000
BATCH_SIZE = 256

def get_dataset(batch_size = BATCH_SIZE, basic_test_data = False):
    global BUFFER_SIZE
    if basic_test_data:
        train_images = np.concatenate([np.zeros([batch_size, 3,4,1], dtype=np.float), np.ones([batch_size, 1,4,1], dtype=np.float)], axis = 1)
    else:
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28,
                                            1).astype('float32')[train_labels==0,:,:,:]
        # We are normalizing the images to the range of [0, 1]
        train_images = np.round(train_images / 256).astype(np.float)
    BUFFER_SIZE = train_images.shape[0]
    assert train_images.shape[1] == train_images.shape[2], "Images should be square"
    image_dim = train_images.shape[1]
    print("Buffer size: %d" % BUFFER_SIZE)
    print("Image size: %d" % image_dim)

    display_images('training_sample.png', train_images[:16,:,:,0])

    # return train_images
    return (tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        BUFFER_SIZE).batch(batch_size),
            image_dim)