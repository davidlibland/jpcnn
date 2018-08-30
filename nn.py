import tensorflow as tf


def shift_layer(x, y):
    nx, px = max(-x, 0), max(x, 0)
    ny, py = max(-y, 0), max(y, 0)
    crop_layer = tf.keras.layers.Cropping2D((ny, py), (nx, px))
    pad_layer = tf.keras.layers.ZeroPadding2D(padding=((py, ny), (px, nx)))
    def helper(v):
        vc = crop_layer(v)
        vp = pad_layer(vc)
        return vp
    return helper


class ShiftLayer(tf.keras.layers.Layer):
    def __init__(self, x_shift=0, y_shift=0):
        super().__init__()
        nx, px = max(-x_shift, 0), max(x_shift, 0)
        ny, py = max(-y_shift, 0), max(y_shift, 0)
        self.crop_layer = tf.keras.layers.Cropping2D(cropping=((ny, py), (nx, px)))
        self.pad_layer = tf.keras.layers.ZeroPadding2D(padding=((py, ny), (px, nx)))

    def call(self, inputs, **kwargs):
        xc = self.crop_layer(inputs)
        xp = self.pad_layer(xc)
        return xp


class ShiftConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), shift_types=None,
                 data_format=None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
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
        self.pad_layer = tf.keras.layers.ZeroPadding2D(padding = (pad_y, pad_x))
        self.conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding="valid", data_format=data_format)

    def build(self, input_shape):
        self.pad_layer.build(input_shape)
        print("input_shape")
        print(input_shape)
        pad_output_shape = self.pad_layer.compute_output_shape(input_shape)
        self.conv_layer.build(pad_output_shape)

    def call(self, inputs, **kwargs):
        pad_x = self.pad_layer(inputs)
        return self.conv_layer(pad_x)

    def compute_output_shape(self, input_shape):
        padded_shape = self.pad_layer.compute_output_shape(input_shape)
        return self.conv_layer.compute_output_shape(padded_shape)


class ShiftDeconv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), shift_types=None,
                 data_format=None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
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
        self.deconv_layer = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding="valid", data_format=data_format)
        self.crop_layer = tf.keras.layers.Cropping2D(cropping = (pad_y, pad_x))

    def build(self, input_shape):
        self.deconv_layer.build(input_shape)
        deconv_output_shape = self.deconv_layer.compute_output_shape(input_shape)
        self.crop_layer.build(deconv_output_shape)

    def call(self, inputs, **kwargs):
        deconv_x = self.deconv_layer(inputs)
        return self.crop_layer(deconv_x)

    def compute_output_shape(self, input_shape):
        deconv_shape = self.deconv_layer.compute_output_shape(input_shape)
        return self.crop_layer.compute_output_shape(deconv_shape)


# def nin(x, num_units, **kwargs):
#     """ a network in network layer (1x1 CONV) """
#     s = int_shape(x)
#     x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
#     x = dense(x, num_units, **kwargs)
#     return tf.reshape(x, s[:-1]+[num_units])
#
# class Skiplayer():
#     def __init__(self, nonlinearity, conv):
#         self.nonlinearity = nonlinearity
#         self.conv = conv
#
#     def __call__(self, x, y):


class Skiplayer(tf.keras.layers.Layer):
    def __init__(self, nonlinearity, conv):
        super().__init__()
        self.nonlinearity = nonlinearity
        if conv is None:
            conv = tf.keras.layers.Conv2D(
                filters = 5,
                kernel_size = (1, 1) # Network in a network layer
            )
        self.conv = conv
        # this requires it to be a conv layer:
        self.num_filters = self.conv.filters
        self.nin = tf.keras.layers.Conv2D(
            filters = self.num_filters,
            kernel_size = (1, 1) # Network in a network layer
        )

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError(
                'A skip layer should be called ' 
                'on a list of inputs.')
        if len(input_shape) != 2:
            raise ValueError('A skip layer should be called '
                             'on a list of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        xs, ys, = input_shape[0], input_shape[1]
        self.conv.build(xs)
        self.nin.build(ys)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            raise ValueError('A skip layer should be called ' 
                             'on a list of two inputs.')
        x, y = inputs[0], inputs[1]
        xnl = self.nonlinearity(x)
        c1 = self.conv(xnl)
        ynl = self.nonlinearity(y)
        c2 = self.nin(ynl)
        return c1 + c2
