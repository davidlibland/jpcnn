import tensorflow as tf


def partition_axis(x, n, partition_axis):
    """Partitions the specified axis into n slices, the slice's
    index is appended to the end of the tensor"""
    num_dims = len(list(tf.shape(x)))
    xs = tf.split(x, num_or_size_splits = n, axis = partition_axis)
    x_stacked = tf.stack(xs, axis = num_dims)
    permutation = list(range(num_dims + 1))
    permutation[partition_axis] = num_dims
    permutation[num_dims] = partition_axis
    return tf.transpose(x_stacked, perm = permutation)


def unpartition_axis(x, partition_axis):
    """Reverse of partition_axis"""
    num_dims = len(list(tf.shape(x)))
    permutation = list(range(num_dims))
    permutation[partition_axis] = num_dims - 1
    permutation[num_dims - 1] = partition_axis
    x_untransposed = tf.transpose(x, perm = permutation, conjugate = True)
    xs = tf.unstack(x_untransposed, axis = num_dims - 1)
    return tf.concat(xs, axis = partition_axis)

