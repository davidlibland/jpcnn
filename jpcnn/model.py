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
                        nn.masked_dense_layer, nn.masked_nin_layer]
    masked_down_shifted_conv2d = lambda x, **kwargs: nn.masked_shift_conv_2D(
                x,
                kernel_size = (2, 3),
                strides = (1, 1),
                shift_types = ["down"],
                **kwargs
    )
    matrix_masked_conv2d = lambda x, **kwargs: nn.masked_shift_conv_2D(
            x,
            kernel_size = (3, 3),
            strides = (1, 1),
            mask_type="matrix",
            **kwargs
        )
    tensor_masked_conv2d = lambda x, **kwargs: nn.masked_shift_conv_2D(
            x,
            kernel_size = (3, 3),
            strides = (1, 1),
            mask_type="tensor",
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
        lf_list = [nn.masked_shift_conv_2D(
                inputs,
                block_heights = in_shape[-1]*[1],
                block_widths = block_widths,
                kernel_size = (3, 3),
                strides = (1, 1),
                include_diagonals = False,
                mask_type="matrix"
        )]
        u_list = [nn.shift_layer(nn.masked_shift_conv_2D(
            inputs,
            block_heights = in_shape[-1]*[1],
            block_widths = block_widths,
            num_filters = num_filters,
            kernel_size = (2, 3),
            strides = (1, 1),
            shift_types = ["down"],
        ), y_shift = 1)]
        m_list = [nn.masked_shift_conv_2D(
                inputs,
                block_heights = in_shape[-1]*[1],
                block_widths = block_widths,
                kernel_size = (3, 3),
                strides = (1, 1),
                include_diagonals = False,
                mask_type="tensor"
            )]

        nn.assert_finite(lf_list[-1])
        nn.assert_finite(u_list[-1])
        nn.assert_finite(m_list[-1])

        # normal pass
        # for merge, upl, uprl in zip(self.merge_feed, self.upward_feed, self.upright_feed):
        for i in range(num_layers):
            for j in range(num_resnet):
                lf_list.append(nn.masked_gated_resnet(
                    lf_list[-1],
                    dropout_p = dropout_p,
                    block_sizes = extended_block_sizes,
                    conv = matrix_masked_conv2d
                ))
                u_list.append(nn.masked_gated_resnet(
                    u_list[-1],
                    masked_a_list=[lf_list[-1]],
                    dropout_p = dropout_p,
                    block_sizes = extended_block_sizes,
                    conv = masked_down_shifted_conv2d
                ))
                m_list.append(nn.masked_gated_resnet(
                    m_list[-1],
                    masked_a_list = [u_list[-1], lf_list[-1]],
                    dropout_p = dropout_p,
                    block_sizes = extended_block_sizes,
                    conv = tensor_masked_conv2d
                ))

                nn.assert_finite(u_list[-1])
                nn.assert_finite(lf_list[-1])
                nn.assert_finite(m_list[-1])
            if i != num_layers-1:
                lf_list.append(nn.masked_shift_conv_2D(
                    lf_list[-1],
                    block_widths = block_widths,
                    block_heights = block_heights,
                    num_filters = num_filters,
                    kernel_size = (3, 3),
                    strides = (2, 2),
                    mask_type="matrix"
                ))
                u_list.append(nn.masked_shift_conv_2D(
                    u_list[-1],
                    block_widths = block_widths,
                    block_heights = block_heights,
                    num_filters = num_filters,
                    kernel_size = (2, 3),
                    strides = (2, 2),
                    shift_types = ["down"]
                ))
                m_list.append(nn.masked_shift_conv_2D(
                    m_list[-1],
                    block_widths = block_widths,
                    block_heights = block_heights,
                    num_filters = num_filters,
                    kernel_size = (3, 3),
                    strides = (2, 2),
                    mask_type="tensor"
                ))

        u = u_list.pop()
        lf = lf_list.pop()
        ml = m_list.pop()

        nn.assert_finite(u)
        nn.assert_finite(lf)
        nn.assert_finite(ml)
        # transpose pass
        # for merge, upl, uprl in zip(self.t_merge_feed, self.t_upward_feed, self.t_upright_feed):
        for i in range(num_layers):
            # add an extra resnet per transpose layer (except on 1st pass)
            c_num_resnet = num_resnet + min(i, 1)
            for j in range(c_num_resnet):
                lf = nn.masked_gated_resnet(
                    lf,
                    masked_a_list = [lf_list.pop()],
                    block_sizes = extended_block_sizes,
                    dropout_p = dropout_p,
                    conv = matrix_masked_conv2d
                )
                u = nn.masked_gated_resnet(
                    u,
                    masked_a_list= [lf,u_list.pop()],
                    dropout_p = dropout_p,
                    block_sizes = extended_block_sizes,
                    conv = masked_down_shifted_conv2d
                )
                ml = nn.masked_gated_resnet(
                    ml,
                    masked_a_list= [lf, u, m_list.pop()],
                    block_sizes = extended_block_sizes,
                    dropout_p = dropout_p,
                    conv = tensor_masked_conv2d
                )

                nn.assert_finite(lf)
                nn.assert_finite(u)
                nn.assert_finite(ml)
            if i != num_layers-1:
                lf = nn.masked_shift_deconv_2D(
                    lf,
                    block_heights = block_heights,
                    block_widths = block_widths,
                    num_filters = num_filters,
                    kernel_size = (3, 3),
                    strides = (2, 2),
                    mask_type="matrix"
                )
                u = nn.shift_deconv_2D(
                    u,
                    num_filters = num_filters,
                    kernel_size = (2, 3),
                    strides = (2, 2),
                    shift_types = ["down"]
                )
                ml = nn.masked_shift_deconv_2D(
                    ml,
                    block_heights = block_heights,
                    block_widths = block_widths,
                    num_filters = num_filters,
                    kernel_size = (3, 3),
                    strides = (2, 2),
                    mask_type="tensor"
                )

        nn.assert_finite(ml)
        assert len(u_list) == 0, "All u layers should be connected."
        assert len(lf_list) == 0, "All lf layers should be connected."
        assert len(m_list) == 0, "All ml layers should be connected."
        # Each mixture should have 3 params,
        final_block_widths = [0] + [ 3 * mixtures_per_channel for _ in block_sizes]
        logits = nn.masked_nin_layer(
            ml,
            block_heights = block_heights,
            block_widths = final_block_widths,
        )
        nn.assert_finite(logits)
        return logits

def pixel_cnn_loss(input, output):
    return tf.losses.sigmoid_cross_entropy(input, output)
