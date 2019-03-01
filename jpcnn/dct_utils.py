from typing import List

import tensorflow as tf
import numpy as np

from jpcnn.nn import get_shape_as_list


def partition_axis(x, n: int, axis: int):
    """Partitions the specified axis into n slices, the slice's
    index is appended to the end of the tensor"""
    num_dims = len(get_shape_as_list(x))
    xs = tf.split(x, num_or_size_splits = n, axis = axis)
    assert isinstance(xs, list)
    assert len(xs) == n, "There should be %d partitions, not %d" % (n, len(xs))
    x_stacked = tf.stack(xs, axis = num_dims)
    assert int(tf.shape(x_stacked)[num_dims]) == n, \
        "The partitions weren't stacked appropriately."
    permutation = list(range(num_dims + 1))
    permutation[axis] = num_dims
    permutation[num_dims] = axis
    result = tf.transpose(x_stacked, perm = permutation)
    assert int(tf.shape(result)[axis]) == n, \
        "The specified axis wasn't partitioned correctly."
    # assert tf.reduce_all(tf.equal(x - unpartition_axis(result, axis), 0)), "%s\n %s" %( x, unpartition_axis(result, axis))
    return result


def unpartition_axis(x, axis: int):
    """Reverse of partition_axis"""
    num_dims = len(list(tf.shape(x)))
    permutation = list(range(num_dims))
    permutation[axis] = num_dims - 1
    permutation[num_dims - 1] = axis
    x_untransposed = tf.transpose(x, perm = permutation)
    xs = tf.unstack(x_untransposed, axis = num_dims - 1)
    result = tf.concat(xs, axis = axis)
    assert len(list(tf.shape(result))) == num_dims - 1, \
        "There should be one fewer dimensions after unpartioning."
    assert int(tf.shape(result)[axis]) == \
           int(tf.shape(x)[axis]) * int(tf.shape(x)[-1]), \
        "The unpartitioned array should concatenate the final axis along the" \
        "partition axis."
    return result


def dct(x, axes: List[int]):
    """Applies a dct to the specified axes"""
    num_dims = len(get_shape_as_list(x))
    def one_axis(x, axis):
        permutation = list(range(num_dims))
        permutation[-1] = axis
        permutation[axis] = num_dims - 1
        x_transposed = tf.transpose(x, perm = permutation)
        x_dct_trans = tf.spectral.dct(x_transposed, norm="ortho")
        return tf.transpose(x_dct_trans, perm = permutation)
    if len(axes) > 0:
        return dct(one_axis(x, axes[-1] % num_dims), axes[:-1])
    return x


def idct(x, axes: List[int]):
    """Applies a idct to the specified axes"""
    num_dims = len(list(tf.shape(x)))
    def one_axis(x, axis):
        permutation = list(range(num_dims))
        permutation[-1] = axis
        permutation[axis] = num_dims - 1
        x_transposed = tf.transpose(x, perm = permutation)
        x_dct_trans = tf.spectral.idct(x_transposed, norm="ortho")
        return tf.transpose(x_dct_trans, perm = permutation)
    if len(axes) > 0:
        return idct(one_axis(x, axes[-1] % num_dims), axes[:-1])
    return x


def jpeg_compression(x, strides: List[int], compression):
    """
    Assumes x is of shape [N, H, W, C] (batch, height, width, channels)
    The output is of shape [N, H/s_h, W/s_w, C, s_h, s_w],
    where strides = [s_h, s_w]
    The strides must be divisors of the height and width.
    """
    if isinstance(compression, list):
        compression = tf.constant(compression, dtype = tf.float32)
    assert tf.reduce_all(compression > 0), \
        "Compression matrix must have positive values"
    in_shape = get_shape_as_list(x)[1:3]
    num_height_parts = in_shape[0] // strides[0]
    num_width_parts = in_shape[1] // strides[1]
    height_part = partition_axis(x, num_height_parts, 1)
    partitioned_x = partition_axis(height_part, num_width_parts, 2)
    dct_x = dct(partitioned_x, [-2, -1])
    quant_qual_reshaped = compression[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    pre_compressed_x = dct_x / quant_qual_reshaped
    compressed_x = tf.round(pre_compressed_x)
    return compressed_x


def jpeg_reconstruction(x, compression):
    """
    Approximate inverse to jpeg_compression.
    """
    if isinstance(compression, list):
        compression = tf.constant(compression, dtype = tf.float32)
    assert tf.reduce_all(compression > 0), \
        "Compression matrix must have positive values"
    quant_qual_reshaped = compression[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    decompressed_x = x * quant_qual_reshaped
    idct_x = idct(decompressed_x, [-2, -1])
    unpart_width_x = unpartition_axis(idct_x, 2)
    return unpartition_axis(unpart_width_x, 1)


def flat_compress(x, compression):
    """
    Assumes x is of shape [N, H, W, C] (batch, height, width, channels)
    The resulting array will be of shape [N, H', W', C']
    where H' = H/s_h, W' = W/s_w, `compression` is of shape (s_h, s_w),
    and C' is ordered from lowest to highest frequency (cycling through
    channels on each frequency before moving to the next frequency).
    """
    strides = get_shape_as_list(compression)
    compressed = jpeg_compression(x, strides, compression)
    in_shape = get_shape_as_list(compressed)
    freq_ordered = diagonal_flatten(compressed)
    # freq_ordered goes from lowest to highest frequencies in the last axis
    freq_prioritized = tf.transpose(freq_ordered, perm = [0, 1, 2, 4, 3])
    # freq_prioritized places the frquencies before the channels prior to
    # flattening; thus we cycle through channels faster than frequencies
    # in the flattened array.
    return tf.reshape(freq_prioritized, in_shape[:3]+[-1])


def flat_reconstruct(x, compression):
    """Approximate inverse to `flat_compress`"""
    strides = get_shape_as_list(compression)
    compressed_length = strides[0] * strides[1]
    in_shape = get_shape_as_list(x)
    channels = in_shape[-1]//compressed_length
    pre_compressed_shape = in_shape[:3]+[channels]+strides
    reshaped = tf.reshape(x, in_shape[:3] + [compressed_length, channels])
    freq_deprioritized = tf.transpose(reshaped, perm = [0, 1, 2, 4, 3])
    unflattened = undiagonal_flatten(freq_deprioritized, y_shape = pre_compressed_shape)
    return jpeg_reconstruction(unflattened, compression)


def basic_compression(
        min_comp: float,
        max_comp: float,
        patch_size: List[int]
) -> List[List[float]]:
    """Builds a basic compression matrix, with entry [i,j] depending linearly
    on i+j, the minimum in the top left, and the maximum in the lower right."""
    b = min_comp
    max_ij = (patch_size[0]+patch_size[1] - 2)
    a = (max_comp-min_comp)/max_ij
    def lin_transformation(n):
        return a*n + b
    compression = []
    for i in range(patch_size[0]):
        compression.append([])
        for j in range(patch_size[1]):
            compression[-1].append(lin_transformation(i+j))
    return compression


def diagonal_flatten(x):
    """
    Traverses the last two axes diagonally from lower left to upper right,
    as in the following ordering:
    [[0, 1, 3],
     [2, 4, 6],
     [5, 7, 8]]
    """
    x_shape = get_shape_as_list(x)
    _, perm = get_diagonal_flatten_permutations(x_shape)
    flat_x = tf.reshape(x, x_shape[:-2] + [-1])
    return tf.gather(flat_x, indices = perm, axis = -1)


def undiagonal_flatten(x, y_shape=None):
    """
    Traverses the last two axes diagonally from lower left to upper right,
    as in the following ordering:
    [[0, 1, 3],
     [2, 4, 6],
     [5, 7, 8]]
    """
    x_shape = get_shape_as_list(x)
    if y_shape is None:
        sqrt = int(np.sqrt(x_shape[-1]))
        y_shape = x_shape[:-1] + [sqrt, sqrt]
    perm, _ = get_diagonal_flatten_permutations(y_shape)
    unperm_x = tf.gather(x, indices = perm, axis = -1)
    return tf.reshape(unperm_x, y_shape)


def get_diagonal_flatten_index_maps(x_shape):
    weights = dict()
    for i in range(x_shape[-2]):
        for j in range(x_shape[-1]):
            n = i + j  # this is the "weight" of the diagonal.
            m = n * (n + 1) / 2  # The number of subdiagonal elements.
            # This is the weight assigned to the i,j-th entry of
            # an infinite matrix:
            weights[(i, j)] = m + j
    # Viewing our matrix as a subset of the infinite matrix gives the correct
    # ordering, so sort according to it, and then take the final position as
    # the index:
    index_mapping = sorted(weights.keys(), key = weights.get)
    mapping = {
        (i, j): k
        for k, (i, j) in enumerate(index_mapping)
    }
    inverse_mapping = {
        k: (i, j)
        for k, (i, j) in enumerate(index_mapping)
    }
    return mapping, inverse_mapping


def get_diagonal_flatten_permutations(x_shape):
    mapping, inverse_mapping = get_diagonal_flatten_index_maps(x_shape)
    def flattened_index(i,j):
        return i * x_shape[-1] + j
    forward_perm = {
        flattened_index(i, j): k
        for (i, j), k in mapping.items()
    }
    reverse_perm = {
        k: flattened_index(i, j)
        for (i, j), k in mapping.items()
    }
    return [forward_perm[i] for i in range(len(forward_perm))], \
           [reverse_perm[i] for i in range(len(reverse_perm))]
