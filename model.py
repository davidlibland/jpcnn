import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
import nn


def model(inputs, num_filters, num_layers, num_resnet=3, dropout_p=0.9, training=True, init=False):
    # Initial layers:
    counters = {}
    if not training:
        dropout_p = 0
    arg_scope_layers = [nn.conv_layer, nn.deconv_layer, nn.gated_resnet, nn.dense_layer, nn.shift_layer, nn.shift_conv_2D, nn.shift_deconv_2D, nn.nin_layer, nn.skip_layer]
    with arg_scope(arg_scope_layers, counters=counters, init=init):
        inputs = tf.pad(inputs, [[0,0],[0,0],[0,0],[0,1]], "CONSTANT", constant_values=1)  # add channel of ones to distinguish image from padding later on
        u = nn.shift_layer(nn.shift_conv_2D(
            inputs,
            filters = num_filters,
            kernel_size = (2, 3),
            strides = (1, 1),
            shift_types = ["down"],
        ), y_shift = 1)
        ul = nn.shift_layer(nn.shift_conv_2D(
            inputs,
            filters = num_filters,
            kernel_size = (1, 3),
            strides = (1, 1),
            shift_types = ["down"]
        ), y_shift = 1) \
            + nn.shift_layer(nn.shift_conv_2D(
            inputs,
            filters = num_filters,
            kernel_size = (2, 1),
            strides = (1, 1),
            shift_types = ["down", "right"]
        ), x_shift = 1)

        # normal pass
        # for merge, upl, uprl in zip(self.merge_feed, self.upward_feed, self.upright_feed):
        for i in range(num_layers):
            u = nn.shift_conv_2D(
                u,
                filters = num_filters,
                kernel_size = (2, 3),
                strides = (1, 1),
                shift_types = ["down"]
            )
            ul = nn.skip_layer(
                x = nn.shift_conv_2D(
                    ul,
                    filters = num_filters,
                    kernel_size = (2, 2),
                    strides = (1, 1),
                    shift_types = ["down", "right"]
                ),
                y = u,
                nonlinearity = tf.nn.leaky_relu,
                dropout_p = dropout_p
            )

        # transpose pass
        # for merge, upl, uprl in zip(self.t_merge_feed, self.t_upward_feed, self.t_upright_feed):
        for i in range(num_layers):
            u = nn.shift_deconv_2D(
                u,
                filters = num_filters,
                kernel_size = (2, 3),
                strides = (1, 1),
                shift_types = ["down"]
            )
            ul = nn.skip_layer(
                x = nn.shift_deconv_2D(
                    ul,
                    filters = num_filters,
                    kernel_size = (2, 2),
                    strides = (1, 1),
                    shift_types = ["down", "right"]
                ),
                y = u,
                nonlinearity = tf.nn.leaky_relu,
                dropout_p = dropout_p
            )

        logits = nn.nin_layer(
            ul,
            1
        )
        return tf.sigmoid(logits)

def pixel_cnn_loss(input, output):
    return tf.losses.sigmoid_cross_entropy(input, output)

optimizer = tf.train.AdamOptimizer(1e-4)
