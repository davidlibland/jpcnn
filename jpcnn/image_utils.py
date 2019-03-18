from typing import List
import os

import numpy as np
from matplotlib import pyplot as plt


def save_and_display_images(root_dir, file_name, image_data, display=True, sample_labels: List[int]=None):
    """image_data is of shape [N, W, H], where N is a (square) batch size."""
    dims = int(np.sqrt(int(image_data.shape[0])))
    assert dims * dims == int(image_data.shape[0]), \
        "There should be a square number of images"
    plt.figure(figsize = (dims, dims), facecolor = 'lightgray')
    for i in range(image_data.shape[0]):
        ax = plt.subplot(dims, dims, i + 1)
        ax.imshow(image_data[i, :, :], cmap = 'gray')
        ax.axis('off')
        if sample_labels is not None:
            ax.set_title(sample_labels[i])
    plt.tight_layout()
    if root_dir is not None:
        plt.savefig(os.path.join(root_dir, file_name))
    if display:
        plt.show()
