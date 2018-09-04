import tensorflow as tf


def shift_layer(x, x_shift=0, y_shift=0, variables=None):
    nx, px = max(-x_shift, 0), max(x_shift, 0)
    ny, py = max(-y_shift, 0), max(y_shift, 0)
    crop_layer = tf.keras.layers.Cropping2D(cropping=((ny, py), (nx, px)))
    pad_layer = tf.keras.layers.ZeroPadding2D(padding=((py, ny), (px, nx)))

    xc = crop_layer(x)
    xp = pad_layer(xc)

    if variables is None:
        raise Warning("variables not being collected")
    else:
        variables += crop_layer.variables + pad_layer.variables
    return xp


def shift_conv_2D(x, filters, kernel_size, strides=(1, 1), shift_types=None,
                 data_format=None, variables=None):
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
    pad_layer = tf.keras.layers.ZeroPadding2D(padding = (pad_y, pad_x))
    conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="valid", data_format=data_format)

    if variables is None:
        raise Warning("variables not being collected")
    else:
        variables += pad_layer.variables + conv_layer.variables
    pad_x = pad_layer(x)
    return conv_layer(pad_x)


def shift_deconv_2D(x, filters, kernel_size, strides=(1, 1), shift_types=None,
                 data_format=None, variables=None):
    if shift_types is None:
        shift_types = []
    if "down" in shift_types:
        pad_y = (0, kernel_size[0]-1)
    elif "up" in shift_types:
        pad_y = (kernel_size[0]-1, 0)
    else:
        pad_y = (int((kernel_size[0]-1)/2), int((kernel_size[0]-1)/2))
    if "right" in shift_types:
        pad_x = (0, kernel_size[1]-1)
    elif "left" in shift_types:
        pad_x = (kernel_size[1]-1, 0)
    else:
        pad_x = (int((kernel_size[1]-1)/2), int((kernel_size[1]-1)/2))
    deconv_layer = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding="valid", data_format=data_format)
    crop_layer = tf.keras.layers.Cropping2D(cropping = (pad_y, pad_x))

    if variables is None:
        raise Warning("variables not being collected")
    else:
        variables += crop_layer.variables + deconv_layer.variables

    deconv_x = deconv_layer(x)
    return crop_layer(deconv_x)


def skip_layer(x, y, nonlinearity, conv, variables=None):
    if conv is None:
        # conv = tf.keras.layers.Conv2D(
        #     filters = 5,
        #     kernel_size = (1, 1) # Network in a network layer
        # )
        pass
    # this requires it to be a conv layer:
    # num_filters = conv.filters
    num_filters = tf.shape(x)[3]
    nin = tf.keras.layers.Conv2D(
        filters = num_filters,
        kernel_size = (1, 1) # Network in a network layer
    )

    xnl = nonlinearity(x)
    c1 = xnl #conv(xnl)
    ynl = nonlinearity(y)
    c2 = nin(ynl)

    if variables is None:
        raise Warning("variables not being collected")
    else:
        variables += nin.variables
    return c1 + c2
