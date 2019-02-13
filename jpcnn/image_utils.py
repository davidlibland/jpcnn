import numpy as np
from matplotlib import pyplot as plt
import os


def save_and_display_images(root_dir, file_name, image_data, display=True):
    """image_data is of shape [N, W, H], where N is a (square) batch size."""
    dims = int(np.sqrt(int(image_data.shape[0])))
    assert dims * dims == int(image_data.shape[0]), \
        "There should be a square number of images"
    plt.figure(figsize = (dims, dims), facecolor = 'lightgray')
    for i in range(image_data.shape[0]):
        plt.subplot(dims, dims, i + 1)
        plt.imshow(image_data[i, :, :], cmap = 'gray')
        plt.axis('off')
    if root_dir is not None:
        plt.savefig(os.path.join(root_dir, file_name))
    if display:
        plt.show()
