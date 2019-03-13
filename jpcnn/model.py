import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
import jpcnn.nn as nn
from jpcnn.dct_utils import get_block_sizes


def model(inputs, labels, avg_num_filters, num_layers, num_resnet=1, compression=None, mixtures_per_channel=1, dropout_p=0.9, training=True, init=False):
    # Set up params for masking:
    assert compression is not None, "Compression needs to be passed to the model"

    counters = {}
    if not training:
        dropout_p = 0
    arg_scope_layers = [nn.masked_conv_layer, nn.masked_deconv_layer, nn.masked_gated_resnet,
                        nn.masked_dense_layer, nn.shift_layer, nn.masked_shift_conv_2D,
                        nn.masked_shift_deconv_2D, nn.masked_nin_layer, nn.masked_skip_layer,
                        nn.masked_dense_layer, nn.masked_nin_layer, nn.shift_conv_2D,
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
    masked_down_right_shifted_conv2d = lambda x, **kwargs: nn.masked_shift_conv_2D(
            x,
            kernel_size = (2, 2),
            strides = (1, 1),
            shift_types = ["down", "right"],
            **kwargs
        )

    with arg_scope(arg_scope_layers, counters=counters, init=init, labels=labels):
        # add channel of ones to distinguish image from padding later on
        inputs = tf.pad(inputs, [[0,0],[0,0],[0,0],[1,0]], "CONSTANT", constant_values=1)
        block_sizes = get_block_sizes(avg_num_filters, compression)
        extended_block_sizes = [1] + block_sizes  # add extra block for the channel of ones.
        block_heights = extended_block_sizes
        block_widths = extended_block_sizes
        num_filters = sum(extended_block_sizes)

        # Initial layers:
        in_shape = list(map(int, tf.shape(inputs)))
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
        ),x_shift = 1)]
        dl_list = [
            nn.masked_shift_conv_2D(
                inputs,
                block_heights = in_shape[-1]*[1],
                block_widths = block_widths,
                kernel_size = (2, 2),
                strides = (1, 1),
                shift_types = ["down", "right"],
                include_diagonals = False
            ) + ul_list[-1]
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
                dl_list.append(nn.masked_gated_resnet(
                    dl_list[-1],
                    tf.concat([u_list[-1], ul_list[-1]], 3),
                    dropout_p = dropout_p,
                    block_sizes = extended_block_sizes,
                    conv = masked_down_right_shifted_conv2d
                ))

                nn.assert_finite(u_list[-1])
                nn.assert_finite(ul_list[-1])
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
                dl_list.append(nn.masked_shift_conv_2D(
                    dl_list[-1],
                    block_widths = block_widths,
                    block_heights = block_heights,
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
                dl = nn.masked_gated_resnet(
                    dl,
                    a = tf.concat([u, ul], 3),
                    masked_a = dl_list.pop(),
                    block_sizes = extended_block_sizes,
                    dropout_p = dropout_p,
                    conv = masked_down_right_shifted_conv2d
                )

                nn.assert_finite(u)
                nn.assert_finite(ul)
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
                dl = nn.masked_shift_deconv_2D(
                    dl,
                    block_heights = block_heights,
                    block_widths = block_widths,
                    kernel_size = (2, 2),
                    strides = (2, 2),
                    shift_types = ["down", "right"]
                )

        nn.assert_finite(ul)
        nn.assert_finite(dl)
        assert len(u_list) == 0, "All u layers should be connected."
        assert len(ul_list) == 0, "All ul layers should be connected."
        assert len(dl_list) == 0, "All dl layers should be connected."
        # Each mixture should have 3 params,
        final_block_heights = block_heights
        final_block_heights[0] += num_filters  # add ul above the blocks.
        final_block_widths = [0] + [ 3 * mixtures_per_channel for _ in block_sizes]
        logits = nn.masked_nin_layer(
            tf.concat([ul, dl], axis=3),
            block_heights = block_heights,
            block_widths = final_block_widths,
        )
        nn.assert_finite(logits)
        return logits

def pixel_cnn_loss(input, output):
    return tf.losses.sigmoid_cross_entropy(input, output)
