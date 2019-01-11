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
def crop2d_layer(x, croppings, counters=None, init=False):
    xshape = list(map(int, x.get_shape()))
    ty = croppings[0][0]
    by = croppings[0][1]
    lx = croppings[1][0]
    rx = croppings[1][1]

    return x[:, ty: xshape[1]-by, lx: xshape[2] - rx, :]

@add_arg_scope
def pad2d_layer(x, paddings, mode="CONSTANT", counters=None, init=False):
    paddings = list(map(list, paddings))
    return tf.pad(x, [[0, 0]] + paddings + [[0, 0]], mode=mode)


@add_arg_scope
def shift_layer(x, x_shift=0, y_shift=0, counters=None, init=False):
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
        V = tf.get_variable(name = "V", shape = kernel_size + (xshape[-1], num_filters), initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float32)
        g = tf.get_variable(name = "g", shape = [num_filters], initializer=tf.constant_initializer(1.), dtype=tf.float32)
        b = tf.get_variable(name = "b", shape = [num_filters], initializer=tf.constant_initializer(0.), dtype=tf.float32)

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
        V = tf.get_variable(name = "V", shape = kernel_size + (num_filters, xshape[-1]), initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float32)
        g = tf.get_variable(name = "g", shape = [num_filters], initializer=tf.constant_initializer(1.), dtype=tf.float32)
        b = tf.get_variable(name = "b", shape = [num_filters], initializer=tf.constant_initializer(0.), dtype=tf.float32)

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
        V = tf.get_variable(name = "V", shape = [xshape[1], num_units], initializer=tf.random_normal_initializer(0, 0.05), dtype=tf.float32)
        g = tf.get_variable(name = "g", shape = [num_units], initializer=tf.constant_initializer(1.), dtype=tf.float32)
        b = tf.get_variable(name = "b", shape = [num_units], initializer=tf.constant_initializer(0.), dtype=tf.float32)

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
def batch_normalization(x, training=True, counters=None, bn_epsilon=1e-3, init=False):
    """A batch normalization layer"""
    name = get_name('batch_norm', counters)
    with tf.variable_scope(name):
        xshape = list(map(int, x.get_shape()))
        x_flatshape = [xshape[0], np.prod(xshape[1:])]
        x_flat = tf.reshape(x, x_flatshape)
        batch_mean, batch_var = tf.nn.moments(x_flat, [0], name=name)
        scale = tf.get_variable(name="bn_scale", shape=x_flatshape, initializer=tf.constant_initializer(1), dtype=tf.float32)
        beta = tf.get_variable(name="bn_offset", shape=x_flatshape, initializer=tf.constant_initializer(1), dtype=tf.float32)
        bn_x = tf.nn.batch_normalization(x_flat, batch_mean, batch_var, beta,
                                        scale, bn_epsilon)
        return tf.reshape(bn_x, xshape)


@add_arg_scope
def gated_resnet(x, a=None, nonlinearity=tf.nn.leaky_relu, conv=conv_layer, dropout_p=0.9, counters=None, **kwargs):
    x_shape = list(map(int, x.get_shape()))
    num_filters = x_shape[-1]

    y1 = conv(nonlinearity(x), num_filters=num_filters)
    if a is not None:  # Add short cut connections:
        y1 += nin_layer(nonlinearity(a), num_filters)
    y1 = nonlinearity(y1)
    if dropout_p > 0:
        y1 = tf.nn.dropout(y1, keep_prob=1. - dropout_p)
    y2 = conv(y1, num_filters = num_filters * 2, init_scale=0.1)

    # Add extra conditioning here, perhaps

    y2_a, y2_b = tf.split(y2, 2, 3)
    y3 = y2_a * tf.nn.sigmoid(y2_b)  # gating
    return x + y3


@add_arg_scope
def shift_conv_2D(x, num_filters, kernel_size, strides=(1, 1), shift_types=None, counters=None, **kwargs):
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

    pad_x = pad2d_layer(x, paddings=[pad_y, pad_x], mode="CONSTANT")
    conv_x = conv_layer(pad_x, num_filters, kernel_size, strides, pad= "VALID", **kwargs)

    return conv_x


@add_arg_scope
def shift_deconv_2D(x, num_filters, kernel_size, strides=(1, 1), shift_types=None,
                    counters=None, **kwargs):
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
    deconv_x = deconv_layer(x, num_filters, kernel_size, strides, pad= "VALID", **kwargs)
    crop_x = crop2d_layer(deconv_x, croppings=(crop_y, crop_x))

    return crop_x


@add_arg_scope
def skip_layer(x, y, nonlinearity=tf.nn.leaky_relu, counters=None, init=False, dropout_p=0.9):
    if nonlinearity is not None:
        x = nonlinearity(x)
    xshape = list(map(int, x.get_shape()))
    c2 = nin_layer(y, xshape[-1], nonlinearity=nonlinearity)

    return x + c2

@add_arg_scope
def down_shifted_conv2d(x, num_filters, kernel_size=(2, 3), strides=(1, 1), **kwargs):
    return shift_conv_2D(
                x,
                num_filters = num_filters,
                kernel_size = kernel_size,
                strides = strides,
                shift_types = ["down"],
                **kwargs
    )

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, kernel_size=(2, 2), strides=(1, 1), **kwargs):
    return shift_conv_2D(
            x,
            num_filters = num_filters,
            kernel_size = kernel_size,
            strides = strides,
            shift_types = ["down", "right"],
            **kwargs
        )


def discretized_mix_logistic_loss(x, l, mixture_sizes):
    # ToDo: add discretization w/ control of granularity
    xshape = list(map(int, x.get_shape()))
    lshape = list(map(int, l.get_shape()))
    assert xshape[:-1] == lshape[:-1], \
        "Target shape must be compatible with shape of distribution params."
    num_logistics = xshape[-1] // 3  # 2 params for each logistic + 1 mix param
    logit_probs = x[...,:num_logistics]
    means = x[..., num_logistics: 2*num_logistics]
    log_scales = x[..., 2*num_logistics:]
    target_indices = [i for j, l in enumerate(mixture_sizes) for i in [j]*l]
    component_targets = tf.gather(l, target_indices, axis=-1)
    centered_targets = component_targets - means
    inv_scale = tf.exp(-log_scales)
    normalized_targets_m = (centered_targets - 0.5) * inv_scale
    normalized_targets_p = (centered_targets + 0.5) * inv_scale

    # Note: mixture_log_loss & mixture_log_probs compute the following:
    # cdf_m = tf.nn.sigmoid(normalized_targets_m)
    # cdf_p = tf.nn.sigmoid(normalized_targets_p)
    # target_prob = cdf_p - cdf_m  # Note: difference could be numerically unstable
    # mixture_log_probs = logit_probs + tf.log(target_prob)

    mixture_log_loss = sum([
        log_expm1(inv_scale),
        - tf.math.softplus(normalized_targets_p),
        - tf.math.softplus(-normalized_targets_m)
    ])

    mixture_log_probs = logit_probs + mixture_log_loss
    stacked = tf.stack([mixture_log_probs, logit_probs], axis = -1)
    components = tf.split(stacked, mixture_sizes, axis=-2)
    mixed_probs = sum(tf.reduce_logsumexp(comp, axis=-2) for comp in components)
    normalized_log_prob = mixed_probs[...,0]-mixed_probs[...,1]
    return -normalized_log_prob



def sample_from_discretized_mix_logistic(x, mixture_sizes):
    """
    We represent a mixture of logistics as a 1-dim array whose
    length is divisible by 3: An array of the form
        [p_1, ..., p_n, m_1, ..., m_n, log(s_1), ..., log(s_n)]

    parametrizes the distribution:
        x_j -> sum_i p_i* logistic_pdf((x-m)/s)/s
    where logisitic_pdf(y)=0.25*sech(y/s) is the pdf of the
    centered logistic distribution function,
    here the index i ranges from offset[j] to offset[j] + mixture_sizes[j]
    where offset[j] = sum(mixture_sizes[:j])

    mixture_sizes is a list of positive integers each indicating the number of
    mixtures in the corresponding output.
    # ToDo: add discretization w/ control of granularity
    """
    xshape = list(map(int, x.get_shape()))
    num_logistics = xshape[-1] // 3  # 2 params for each logistic + 1 mixture

    # First we sample the softmax indicators:
    softmax_indices = sample_multinomials(x[...,:num_logistics], mixture_sizes)
    logistic_means = tf.batch_gather(
        params = x[..., num_logistics: 2*num_logistics],
        indices = softmax_indices,
    )
    logistic_scales = tf.exp(tf.batch_gather(
        params = x[..., 2*num_logistics:],
        indices = softmax_indices,
    ))
    u = tf.random_uniform(logistic_means.get_shape(),
                          minval=1e-5, maxval=1. - 1e-5)
    smooth_x = logistic_means + logistic_scales*(tf.log(u) - tf.log(1. - u))
    # discretize (todo)
    return smooth_x


def sample_multinomials(logit_probs, num_multinomials):
    """Samples from sequence of multinomial distributions, num_multinomials
    is a list of integers indicating the size of each multinomial set.

    For instance to sample from a binomial and trinomial distribution with
    logit probs (2, -1), and (1, 1, 2), respectively, one would call:
    sample_multinomials([2, -1, 1, 1, 2], [2, 3])
    """
    uniform_samples = tf.random_uniform(
        logit_probs.get_shape(),
        minval = 1e-5,
        maxval = 1. - 1e-5
    )
    all_gumbel_samples = logit_probs - tf.log(-tf.log(uniform_samples))
    split_gumbel_samples = tf.split(all_gumbel_samples, num_multinomials, axis=-1)
    print([0]+num_multinomials[:-1])
    offsets = partition_offsets(num_multinomials)
    indices = [tf.cast(tf.argmax(gumbel_samples, axis=-1), tf.int32) + offset
               for offset, gumbel_samples in zip(offsets, split_gumbel_samples)]
    return tf.stack(indices, axis=-1)


def partition_offsets(partition_sizes):
    offsets = tf.cast(tf.scan(
        lambda a, x: a + x,
        tf.concat([[0] + partition_sizes[:-1]], axis = 0)
    ), dtype = tf.int32)
    return offsets

@tf.custom_gradient
def log_expm1(x):
    """Numerically stable log(expm1(_))"""
    def grad(dy):
        return dy*tf.where(x > 0, -1./tf.expm1(-x), tf.exp(x)/tf.expm1(x))
    return tf.where(x > 50, x+tf.log1p(-tf.exp(-x)), tf.log(tf.expm1(x))), grad
