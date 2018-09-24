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
def conv_layer(x, num_filters, kernel_size, strides, pad="SAME", nonlinearity=None, counters=None, init=False, init_scale=1.):
    name = get_name('conv', counters)
    kernel_size=tuple(kernel_size)
    strides=tuple(strides)
    with tf.variable_scope(name):
        xshape = list(map(int, x.get_shape()))
        V = tf.get_variable(name = "V", shape = kernel_size + (xshape[-1], num_filters), initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float64)
        g = tf.get_variable(name = "g", shape = [num_filters], initializer=tf.constant_initializer(1.), dtype=tf.float64)
        b = tf.get_variable(name = "b", shape = [num_filters], initializer=tf.constant_initializer(0.), dtype=tf.float64)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, (1,) + strides + (1,), pad), b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.identity(x)

        if nonlinearity is not None:
            x = nonlinearity(x)
        else:
            x = tf.nn.leaky_relu(x)
        return x


@add_arg_scope
def deconv_layer(x, num_filters, kernel_size, strides, pad="SAME", nonlinearity=None, counters=None, init=False, init_scale=1.):
    name = get_name('deconv', counters)
    xshape = list(map(int, x.get_shape()))
    if pad=='SAME':
        output_shape = [xshape[0], xshape[1]*strides[0], xshape[2]*strides[1], num_filters]
    else:
        output_shape = [xshape[0], xshape[1]*strides[0] + kernel_size[0]-1, xshape[2]*strides[1] + kernel_size[1]-1, num_filters]
    with tf.variable_scope(name):
        xshape = list(map(int, x.get_shape()))
        V = tf.get_variable(name = "V", shape = kernel_size + (num_filters, xshape[-1]), initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float64)
        g = tf.get_variable(name = "g", shape = [num_filters], initializer=tf.constant_initializer(1.), dtype=tf.float64)
        b = tf.get_variable(name = "b", shape = [num_filters], initializer=tf.constant_initializer(0.), dtype=tf.float64)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(x, W, output_shape, (1,) + strides + (1,), padding=pad)
        x = tf.nn.bias_add(x, b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.identity(x)

        if nonlinearity is not None:
            x = nonlinearity(x)
        else:
            x = tf.nn.leaky_relu(x)
        return x


@add_arg_scope
def dense_layer(x, num_units, nonlinearity=None, counters=None, init=False, init_scale=1.):
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        xshape = list(map(int, x.get_shape()))
        V = tf.get_variable(name = "V", shape = [xshape[1], num_units], initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float64)
        g = tf.get_variable(name = "g", shape = [num_units], initializer=tf.constant_initializer(1.), dtype=tf.float64)
        b = tf.get_variable(name = "b", shape = [num_units], initializer=tf.constant_initializer(0.), dtype=tf.float64)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        if init: # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale/tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g*scale_init), b.assign_add(-m_init*scale_init)]):
                x = tf.identity(x)

        if nonlinearity is not None:
            x = nonlinearity(x)
        return x


@add_arg_scope
def nin_layer(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    xshape = list(map(int, x.get_shape()))
    x = tf.reshape(x, [np.prod(xshape[:-1]), xshape[-1]])
    x = dense_layer(x, num_units, **kwargs)
    return tf.reshape(x, xshape[:-1]+[num_units])


@add_arg_scope
def batch_normalization(x, training=True, counters=None, bn_epsilon=1e-3):
    """A batch normalization layer"""
    name = get_name('batch_norm', counters)
    with tf.variable_scope(name):
        xshape = list(map(int, x.get_shape()))
        print(xshape)
        x_flatshape = [xshape[0], np.prod(xshape[1:])]
        print(x_flatshape)
        x_flat = tf.reshape(x, x_flatshape)
        batch_mean, batch_var = tf.nn.moments(x_flat, [0], name=name)
        scale = tf.get_variable(name="bn_scale", shape=x_flatshape, initializer=tf.constant_initializer(1), dtype=tf.float64)
        beta = tf.get_variable(name="bn_offset", shape=x_flatshape, initializer=tf.constant_initializer(1), dtype=tf.float64)
        bn_x = tf.nn.batch_normalization(x_flat, batch_mean, batch_var, beta,
                                        scale, bn_epsilon)
        return tf.reshape(bn_x, xshape)


@add_arg_scope
def gated_resnet(x, a=None, nonlinearity=tf.nn.leaky_relu, conv=conv_layer, dropout_p=0.9, counters=None, **kwargs):
    x_shape = list(map(int, x.get_shape()))
    num_filters = x_shape[-1]

    y1 = conv(nonlinearity(x), num_filters, counters = counters)
    if a is not None:  # Add short cut connections:
        y1 += nin_layer(nonlinearity(a), num_filters, counters = counters)
    y1 = nonlinearity(y1)
    # Do dropout here
    y2 = conv(y1, num_filters * 2)

    # Add extra conditioning here, perhaps

    y2_a, y2_b = tf.split(y2, 2, 3)
    y3 = y2_a * tf.nn.sigmoid(y2_b)  # gating
    return x + y3


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
