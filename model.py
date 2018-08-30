import tensorflow as tf
import nn


class PixelCNN(tf.keras.Model):
    def __init__(self, num_filters, num_layers):
        super().__init__()

        # Initial layers:
        self.down_shift = nn.ShiftLayer(y_shift = 1)
        self.right_shift = nn.ShiftLayer(x_shift = 1)
        self.offset_initial_upper_feed = nn.ShiftConv2D(
            filters = num_filters,
            kernel_size = (2, 3),
            strides = (1, 1),
            shift_types = ["down"]
        )
        self.offset_initial_narrow_upper_feed = nn.ShiftConv2D(
            filters = num_filters,
            kernel_size = (1, 3),
            strides = (1, 1),
            shift_types = ["down"]
        )
        self.offset_initial_narrow_right_feed = nn.ShiftConv2D(
            filters = num_filters,
            kernel_size = (2, 1),
            strides = (1, 1),
            shift_types = ["down", "right"]
        )

        # normal pass
        self.merge_feed = []
        self.upward_feed = []
        self.upright_feed = []
        for i in range(num_layers):
            self.merge_feed += [nn.Skiplayer(
                nonlinearity = tf.keras.layers.LeakyReLU(),
                conv = nn.ShiftConv2D(
                    filters = num_filters,
                    kernel_size = (2, 2),
                    strides = (1, 1),
                    shift_types = ["down","right"]
                )
            )]
            self.upward_feed += [nn.ShiftConv2D(
                filters = num_filters,
                kernel_size = (2, 3),
                strides = (2, 2),
                shift_types = ["down"]
            )]
            self.upright_feed += [nn.ShiftConv2D(
                filters = num_filters,
                kernel_size = (2, 2),
                strides = (2, 2),
                shift_types = ["down","right"]
            )]

        # transpose pass
        self.t_merge_feed = []
        self.t_upward_feed = []
        self.t_upright_feed = []
        for i in range(num_layers):
            self.t_merge_feed += [nn.Skiplayer(
                nonlinearity = tf.keras.layers.LeakyReLU(),
                conv = nn.ShiftConv2D(
                    filters = num_filters,
                    kernel_size = (2, 2),
                    strides = (1, 1),
                    shift_types = ["down","right"]
                )
            )]
            self.t_upward_feed += [nn.ShiftDeconv2D(
                filters = num_filters,
                kernel_size = (2, 3),
                strides = (2, 2),
                shift_types = ["down"]
            )]
            self.t_upright_feed += [nn.ShiftDeconv2D(
                filters = num_filters,
                kernel_size = (2, 2),
                strides = (2, 2),
                shift_types = ["down","right"]
            )]
        self.final_collapse = tf.keras.layers.Conv2D(
            filters = 1, # greyscale output
            kernel_size = (1, 1) # Network in a network layer
        )
        self.final_activation = tf.keras.layers.Activation("sigmoid")


    def build(self, input_shape):
        print(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=True):
        # Initial layers:

        u = self.down_shift(self.offset_initial_upper_feed(inputs))
        ul = self.down_shift(self.offset_initial_narrow_upper_feed(inputs)) \
            + self.right_shift(self.offset_initial_narrow_right_feed(inputs))

        # normal pass
        for merge, upl, uprl in zip(self.merge_feed, self.upward_feed, self.upright_feed):
            u = upl(u)
            ul = merge([uprl(ul), u])

        # transpose pass
        for merge, upl, uprl in zip(self.t_merge_feed, self.t_upward_feed, self.t_upright_feed):
            u = upl(u)
            ul = merge([uprl(ul), u])

        logits = self.final_collapse(ul)
        return self.final_activation(logits)

def pixel_cnn_loss(input, output):
    return tf.losses.sigmoid_cross_entropy(input, output)

optimizer = tf.train.AdamOptimizer(1e-4)
