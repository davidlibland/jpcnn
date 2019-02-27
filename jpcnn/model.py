import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
import jpcnn.nn as nn


def model(inputs, labels, num_filters, num_layers, num_resnet=1, dropout_p=0.9, training=True, init=False):
    # Initial layers:
    in_shape = list(map(int, tf.shape(inputs)))
    counters = {}
    if not training:
        dropout_p = 0
    arg_scope_layers = [nn.conv_layer, nn.deconv_layer, nn.gated_resnet, nn.dense_layer, nn.shift_layer, nn.shift_conv_2D, nn.shift_deconv_2D, nn.nin_layer, nn.skip_layer]
    down_shifted_conv2d = lambda x, **kwargs: nn.shift_conv_2D(
                x,
                kernel_size = (2, 3),
                strides = (1, 1),
                shift_types = ["down"],
                **kwargs
    )
    down_right_shifted_conv2d = lambda x, **kwargs: nn.shift_conv_2D(
            x,
            kernel_size = (2, 2),
            strides = (1, 1),
            shift_types = ["down", "right"],
            **kwargs
        )

    with arg_scope(arg_scope_layers, counters=counters, init=init, labels=labels):
        inputs = tf.pad(inputs, [[0,0],[0,0],[0,0],[0,1]], "CONSTANT", constant_values=1)  # add channel of ones to distinguish image from padding later on
        u_list = [nn.shift_layer(nn.shift_conv_2D(
            inputs,
            num_filters = num_filters,
            kernel_size = (2, 3),
            strides = (1, 1),
            shift_types = ["down"],
        ), y_shift = 1)]
        ul_list = [nn.shift_layer(nn.shift_conv_2D(
            inputs,
            num_filters = num_filters,
            kernel_size = (1, 3),
            strides = (1, 1),
            shift_types = ["down"]
        ), y_shift = 1) \
            + nn.shift_layer(nn.shift_conv_2D(
            inputs,
            num_filters = num_filters,
            kernel_size = (2, 1),
            strides = (1, 1),
            shift_types = ["down", "right"]
        ), x_shift = 1)]

        # normal pass
        # for merge, upl, uprl in zip(self.merge_feed, self.upward_feed, self.upright_feed):
        for i in range(num_layers):
            for j in range(num_resnet):
                u_list.append(nn.gated_resnet(
                    u_list[-1],
                    dropout_p = dropout_p,
                    conv = down_shifted_conv2d
                ))
                ul_list.append(nn.gated_resnet(
                    ul_list[-1],
                    u_list[-1],
                    dropout_p = dropout_p,
                    conv = down_right_shifted_conv2d
                ))
            if i != num_layers-1:
                u_list.append(nn.shift_conv_2D(
                    u_list[-1],
                    num_filters = num_filters,
                    kernel_size = (2, 3),
                    strides = (2, 2),
                    shift_types = ["down"]
                ))
                ul_list.append(nn.shift_conv_2D(
                    ul_list[-1],
                    num_filters = num_filters,
                    kernel_size = (2, 2),
                    strides = (2, 2),
                    shift_types = ["down", "right"]
                ))

        u = u_list.pop()
        ul = ul_list.pop()
        # transpose pass
        # for merge, upl, uprl in zip(self.t_merge_feed, self.t_upward_feed, self.t_upright_feed):
        for i in range(num_layers):
            # add an extra resnet per transpose layer (except on 1st pass)
            c_num_resnet = num_resnet + min(i, 1)
            for j in range(c_num_resnet):
                u = nn.gated_resnet(
                    u_list.pop(),
                    dropout_p = dropout_p,
                    conv = down_shifted_conv2d
                )
                ul = nn.gated_resnet(
                    ul,
                    tf.concat([u, ul_list.pop()], 3),
                    dropout_p = dropout_p,
                    conv = down_right_shifted_conv2d
                )
            if i != num_layers-1:
                u = nn.shift_deconv_2D(
                    u,
                    num_filters = num_filters,
                    kernel_size = (2, 3),
                    strides = (2, 2),
                    shift_types = ["down"]
                )
                ul = nn.shift_deconv_2D(
                    ul,
                    num_filters = num_filters,
                    kernel_size = (2, 2),
                    strides = (2, 2),
                    shift_types = ["down", "right"]
                )

        assert len(u_list) == 0, "All u layers should be connected."
        assert len(ul_list) == 0, "All ul layers should be connected."
        logits = nn.nin_layer(
            ul,
            3 * in_shape[-1]  # 3 per channel
        )
        return logits

def pixel_cnn_loss(input, output):
    return tf.losses.sigmoid_cross_entropy(input, output)
