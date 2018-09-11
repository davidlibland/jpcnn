import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope


def get_name(layer_name: str, counters: dict):
    """ utlity for keeping track of layer names """
    if counters is None:
        raise ValueError("No counter dict was provided to get_name.")
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

@add_arg_scope
def crop2d_layer(x, croppings, counters=None):
    xshape = list(map(int, x.get_shape()))
    ty = croppings[0][0]
    by = croppings[0][1]
    lx = croppings[1][0]
    rx = croppings[1][1]

    return x[:, ty: xshape[1]-by, lx: xshape[2] - rx, :]

@add_arg_scope
def pad2d_layer(x, paddings, mode="CONSTANT", counters=None):
    paddings = list(map(list, paddings))
    return tf.pad(x, [[0, 0]] + paddings + [[0, 0]], mode=mode)


@add_arg_scope
def shift_layer(x, x_shift=0, y_shift=0, counters=None):
    nx, px = max(-x_shift, 0), max(x_shift, 0)
    ny, py = max(-y_shift, 0), max(y_shift, 0)

    xc = crop2d_layer(x, croppings=((ny, py), (nx, px)))
    xp = pad2d_layer(xc, paddings=((py, ny), (px, nx)), mode="CONSTANT")

    return xp


@add_arg_scope
def conv_layer(x, num_filters, kernel_size, strides, pad="SAME", nonlinearity=None, counters=None):
    name = get_name('conv', counters)
    kernel_size=tuple(kernel_size)
    strides=tuple(strides)
    with tf.variable_scope(name):
        xshape = list(map(int, x.get_shape()))
        W = tf.get_variable(name = "W", shape = kernel_size + (xshape[-1], num_filters), initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float64)
        b = tf.get_variable(name = "b", shape = [num_filters], initializer=tf.constant_initializer(0), dtype=tf.float64)
        conv_x = tf.nn.conv2d(x, W, (1, ) + strides + (1,), pad)
        out_x = tf.nn.bias_add(conv_x, b)
        if nonlinearity is not None:
            out_x = nonlinearity(out_x)
        else:
            out_x = tf.nn.leaky_relu(out_x)
        return out_x


@add_arg_scope
def deconv_layer(x, num_filters, kernel_size, strides, pad="SAME", nonlinearity=None, counters=None):
    name = get_name('deconv', counters)
    xshape = list(map(int, x.get_shape()))
    if pad=='SAME':
        output_shape = [xshape[0], xshape[1]*strides[0], xshape[2]*strides[1], num_filters]
    else:
        output_shape = [xshape[0], xshape[1]*strides[0] + kernel_size[0]-1, xshape[2]*strides[1] + kernel_size[1]-1, num_filters]
    with tf.variable_scope(name):
        W = tf.get_variable(name = "W", shape = kernel_size + (xshape[-1], num_filters), initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float64)
        b = tf.get_variable(name = "b", shape = [num_filters], initializer=tf.constant_initializer(0), dtype=tf.float64)
        conv_x = tf.nn.conv2d_transpose(x, W, output_shape, (1, ) + strides + (1,), pad)
        out_x = tf.nn.bias_add(conv_x, b)
        if nonlinearity is not None:
            out_x = nonlinearity(out_x)
        return out_x


@add_arg_scope
def dense_layer(x, num_units, nonlinearity=None, counters=None):
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        xshape = list(map(int, x.get_shape()))
        W = tf.get_variable(name = "W", shape = [xshape[1], num_units], initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float64)
        b = tf.get_variable(name = "b", shape = [num_units], initializer=tf.constant_initializer(0), dtype=tf.float64)
        x_W = tf.matmul(x, W)
        out_x = tf.nn.bias_add(x_W, b)
        if nonlinearity is not None:
            out_x = nonlinearity(out_x)
        return out_x


@add_arg_scope
def nin_layer(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    xshape = list(map(int, x.get_shape()))
    x = tf.reshape(x, [np.prod(xshape[:-1]), xshape[-1]])
    x = dense_layer(x, num_units, **kwargs)
    return tf.reshape(x, xshape[:-1]+[num_units])


@add_arg_scope
def shift_conv_2D(x, filters, kernel_size, strides=(1, 1), shift_types=None, counters=None):
    if shift_types is None:
        shift_types = []
    if "down" in shift_types:
        pad_y = (kernel_size[0]-1, 0)
    elif "up" in shift_types:
        pad_y = (0, kernel_size[0]-1)
    else:
        pad_y = (int((kernel_size[0]-1)/2), int((kernel_size[0]-1)/2))
    if "right" in shift_types:
        pad_x = (kernel_size[1]-1, 0)
    elif "left" in shift_types:
        pad_x = (0, kernel_size[1]-1)
    else:
        pad_x = (int((kernel_size[1]-1)/2), int((kernel_size[1]-1)/2))

    pad_x = pad2d_layer(x, paddings=[pad_y, pad_x], mode="CONSTANT", counters = counters)
    conv_x = conv_layer(pad_x, filters, kernel_size, strides, pad="VALID", counters = counters)

    return conv_x


@add_arg_scope
def shift_deconv_2D(x, filters, kernel_size, strides=(1, 1), shift_types=None,
                    counters=None):
    if shift_types is None:
        shift_types = []
    if "down" in shift_types:
        crop_y = (0, kernel_size[0]-1)
    elif "up" in shift_types:
        crop_y = (kernel_size[0]-1, 0)
    else:
        crop_y = (int((kernel_size[0]-1)/2), int((kernel_size[0]-1)/2))
    if "right" in shift_types:
        crop_x = (0, kernel_size[1]-1)
    elif "left" in shift_types:
        crop_x = (kernel_size[1]-1, 0)
    else:
        crop_x = (int((kernel_size[1]-1)/2), int((kernel_size[1]-1)/2))
    deconv_x = deconv_layer(x, filters, kernel_size, strides, pad="VALID", counters=counters)
    crop_x = crop2d_layer(deconv_x, croppings=(crop_y, crop_x))

    return crop_x


@add_arg_scope
def skip_layer(x, y, nonlinearity=tf.nn.leaky_relu, counters=None):
    if nonlinearity is not None:
        x = nonlinearity(x)
    xshape = list(map(int, x.get_shape()))
    ynl = nonlinearity(y)
    c2 = nin_layer(ynl, xshape[-1], counters=counters)

    return x + c2
