# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
import numpy as np
from jpcnn.image_utils import save_and_display_images

# Globals:
BATCH_SIZE = 256

# Horizontal white line: nl=1, nr=1, nf=50, lr=1e-2
test_data = np.concatenate([np.zeros([3,4], dtype=np.float32), np.ones([1,4], dtype=np.float32)], axis = 0)
# diagonal line: nl=1, nr=1, nf=50, lr=1e-2
test_data = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
], dtype=np.float32)
# small zero: nl=1, nr=1, nf=50, lr=1e-2
test_data = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
], dtype=np.float32)
# zero: nl=2, nr=2, nf=32, lr=1e-4
# test_data = np.array([
#     [1, 1, 1, 1, 1, 1],
#     [1, 1, 0, 0, 1, 1],
#     [1, 0, 1, 1, 0, 1],
#     [1, 0, 1, 1, 0, 1],
#     [1, 1, 0, 0, 1, 1],
#     [1, 1, 1, 1, 1, 1],
# ], dtype=np.float32)
# small gradient: nl=1, nr=1, nf=50, lr=1e-2
test_data = np.array([
    [0, 40, 80, 120],
    [40, 80, 120, 160],
    [80, 120, 160, 200],
    [120, 160, 200, 240]
], dtype=np.float32)


def get_dataset(batch_size = BATCH_SIZE, image_preprocessor = None, basic_test_data = False, dtype: str= "float32"):
    train_labels = None
    if basic_test_data:
        all_images = np.stack([test_data]*batch_size, axis = 0)
        all_images = np.expand_dims(all_images, 3)
        all_images = all_images.astype(dtype)/40
    else:
        (all_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        all_images = all_images.reshape(all_images.shape[0], 28, 28,
                                            1).astype(dtype)  # [train_labels==0,:,:,:]
        # We are normalizing the images to the range of [0, 1]
        # train_images = np.round(train_images / 256).astype(dtype)
        all_images = all_images.astype(dtype)
    buffer_size = all_images.shape[0]
    assert all_images.shape[1] == all_images.shape[2], "Images should be square"
    image_dim = all_images.shape[1]
    # Normalize images to the range of [0, 1]
    image_max = tf.reduce_max(all_images)
    image_min = tf.reduce_min(all_images)
    assert image_max > image_min, "Must have nontrivial images"
    if image_max > 1 or image_min < 0:
        all_images = (all_images - image_min)/(image_max - image_min)
    print("Buffer size: %d" % buffer_size)
    print("Mini Batch size: %d" % batch_size)
    print("Image size: %d" % image_dim)

    save_and_display_images(".", "training_sample.png", all_images[:16, :, :, 0])

    if image_preprocessor:
        all_images = image_preprocessor(all_images)

    if train_labels is not None:
        one_hot_labels = tf.one_hot(train_labels, max(train_labels))
        dataset = tf.data.Dataset.from_tensor_slices((all_images, one_hot_labels)) \
            .shuffle(buffer_size) \
            .batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(all_images) \
            .shuffle(buffer_size) \
            .batch(batch_size)
    # return train_images

    return (dataset, image_dim, buffer_size)
