# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Globals:
BUFFER_SIZE = 60000
BATCH_SIZE = 256

def get_dataset():
    global BUFFER_SIZE
    # train_images = np.concatenate([np.zeros([BATCH_SIZE, 3,4,1], dtype=np.float), np.ones([BATCH_SIZE, 1,4,1], dtype=np.float)], axis = 1)
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

    fig = plt.figure(figsize = (4, 4))

    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(train_images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('training_sample.png')
    plt.show()

    # return train_images
    return (tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE),
            image_dim)