import numpy as np
from matplotlib import pyplot as plt


def display_images(name, image_data):
    dims = int(np.sqrt(image_data.shape[0]))
    assert dims * dims == int(image_data.shape[0]), \
        "There should be a square number of images"
    fig = plt.figure(figsize = (dims, dims), facecolor = 'lightgray')
    for i in range(image_data.shape[0]):
        plt.subplot(dims, dims, i + 1)
        plt.imshow(image_data[i, :, :], cmap = 'gray')
        plt.axis('off')
    plt.savefig(name)
    plt.show()