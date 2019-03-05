import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
import jpcnn.nn as nn


def model(inputs, labels, num_filters, num_layers, num_resnet=1, num_blocks=1, mixtures_per_channel=1, dropout_p=0.9, training=True, init=False):
    # Initial layers:
    in_shape = list(map(int, tf.shape(inputs)))
    counters = {}
    if not training:
        dropout_p = 0
    arg_scope_layers = [nn.lt_conv_layer, nn.lt_deconv_layer, nn.lt_gated_resnet,
                        nn.lt_dense_layer, nn.shift_layer, nn.lt_shift_conv_2D,
                        nn.lt_shift_deconv_2D, nn.lt_nin_layer, nn.lt_skip_layer,
                        nn.slt_dense_layer, nn.slt_nin_layer, nn.shift_conv_2D,
                        nn.shift_deconv_2D, nn.gated_resnet, nn.conv_layer,
                        nn.deconv_layer, nn.dense_layer, nn.nin_layer]
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
    lt_down_right_shifted_conv2d = lambda x, **kwargs: nn.lt_shift_conv_2D(
            x,
            kernel_size = (2, 2),
            strides = (1, 1),
            shift_types = ["down", "right"],
            **kwargs
        )

    with arg_scope(arg_scope_layers, counters=counters, init=init, labels=labels, num_blocks=num_blocks):
        #inputs = tf.pad(inputs, [[0,0],[0,0],[0,0],[0,1]], "CONSTANT", constant_values=1)  # add channel of ones to distinguish image from padding later on
        u_list = [nn.shift_layer(nn.shift_conv_2D(
            inputs,
            num_filters = num_filters * num_blocks,
            kernel_size = (2, 3),
            strides = (1, 1),
            shift_types = ["down"],
        ), y_shift = 1)]
        ul_list = [nn.shift_layer(nn.shift_conv_2D(
            inputs,
            num_filters = num_filters * num_blocks,
            kernel_size = (1, 3),
            strides = (1, 1),
            shift_types = ["down"]
        ), y_shift = 1) \
            + nn.shift_layer(nn.shift_conv_2D(
            inputs,
            num_filters = num_filters * num_blocks,
            kernel_size = (2, 1),
            strides = (1, 1),
            shift_types = ["down", "right"]
        ),x_shift = 1)]
        dl_list = [
            nn.shift_layer(nn.lt_shift_conv_2D(
                inputs,
                num_filters = num_filters,
                kernel_size = (2, 2),
                strides = (1, 1),
                shift_types = ["down", "right"],
                include_diagonals = False
            ), x_shift = 1) + ul_list[-1]
        ]

        nn.assert_finite(u_list[-1])
        nn.assert_finite(ul_list[-1])
        nn.assert_finite(dl_list[-1])

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
                dl_list.append(nn.lt_gated_resnet(
                    dl_list[-1],
                    tf.concat([u_list[-1], ul_list[-1]], 3),
                    dropout_p = dropout_p,
                    conv = lt_down_right_shifted_conv2d
                ))

                nn.assert_finite(u_list[-1])
                nn.assert_finite(ul_list[-1])
            if i != num_layers-1:
                u_list.append(nn.shift_conv_2D(
                    u_list[-1],
                    num_filters = num_filters * num_blocks,
                    kernel_size = (2, 3),
                    strides = (2, 2),
                    shift_types = ["down"]
                ))
                ul_list.append(nn.shift_conv_2D(
                    ul_list[-1],
                    num_filters = num_filters * num_blocks,
                    kernel_size = (2, 2),
                    strides = (2, 2),
                    shift_types = ["down", "right"]
                ))
                dl_list.append(nn.lt_shift_conv_2D(
                    dl_list[-1],
                    num_filters = num_filters,
                    kernel_size = (2, 2),
                    strides = (2, 2),
                    shift_types = ["down", "right"]
                ))

        u = u_list.pop()
        ul = ul_list.pop()
        dl = dl_list.pop()

        nn.assert_finite(u)
        nn.assert_finite(ul)
        nn.assert_finite(dl)
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
                dl = nn.lt_gated_resnet(
                    dl,
                    tf.concat([u, ul, dl_list.pop()], 3),
                    dropout_p = dropout_p,
                    conv = lt_down_right_shifted_conv2d
                )

                nn.assert_finite(u)
                nn.assert_finite(ul)
            if i != num_layers-1:
                u = nn.shift_deconv_2D(
                    u,
                    num_filters = num_filters * num_blocks,
                    kernel_size = (2, 3),
                    strides = (2, 2),
                    shift_types = ["down"]
                )
                ul = nn.shift_deconv_2D(
                    ul,
                    num_filters = num_filters * num_blocks,
                    kernel_size = (2, 2),
                    strides = (2, 2),
                    shift_types = ["down", "right"]
                )
                dl = nn.lt_shift_deconv_2D(
                    dl,
                    num_filters = num_filters,
                    kernel_size = (2, 2),
                    strides = (2, 2),
                    shift_types = ["down", "right"]
                )

        nn.assert_finite(ul)
        nn.assert_finite(dl)
        assert len(u_list) == 0, "All u layers should be connected."
        assert len(ul_list) == 0, "All ul layers should be connected."
        assert len(dl_list) == 0, "All dl layers should be connected."
        logits = nn.lt_nin_layer(
            dl,
            mixtures_per_channel * 3 * in_shape[-1] // num_blocks  # 3 per channel
        )
        nn.assert_finite(logits)
        return logits

def pixel_cnn_loss(input, output):
    return tf.losses.sigmoid_cross_entropy(input, output)
