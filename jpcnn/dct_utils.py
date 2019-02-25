from typing import List

import tensorflow as tf

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
    The output is of shape [N, H/s_h, W/s_w, C'], where strides = [s_h, s_w]
    The strides must be divisors of the height and width.
    # Docstring incorrect: the shape is extended by 2d
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
    return tf.transpose(compressed_x, perm = [0, 1, 2, 4, 5, 3])


def jpeg_reconstruction(x, compression):
    """
    Assumes x is of shape [N, H, W, C] (batch, height, width, channels)
    The output is of shape [N, H/s_h, W/s_w, C'], where strides = [s_h, s_w]
    The strides must be divisors of the height and width.
    # Docstring incorrect: the shape is shrunk by 2d
    """
    if isinstance(compression, list):
        compression = tf.constant(compression, dtype = tf.float32)
    assert tf.reduce_all(compression > 0), \
        "Compression matrix must have positive values"
    quant_qual_reshaped = compression[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    reshaped_x = tf.transpose(x, perm = [0, 1, 2, 5, 3, 4])
    decompressed_x = reshaped_x * quant_qual_reshaped
    idct_x = idct(decompressed_x, [-2, -1])
    unpart_width_x = unpartition_axis(idct_x, 2)
    return unpartition_axis(unpart_width_x, 1)


def flat_compress(x, compression):
    strides = get_shape_as_list(compression)
    compressed = jpeg_compression(x, strides, compression)
    in_shape = get_shape_as_list(compressed)
    return tf.reshape(compressed, in_shape[:3]+[-1])


def flat_reconstruct(x, compression):
    strides = get_shape_as_list(compression)
    in_shape = get_shape_as_list(x)
    reshaped = tf.reshape(x, in_shape[:3] + strides + [-1])
    return jpeg_reconstruction(reshaped, compression)


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
