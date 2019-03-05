import sys

import tensorflow as tf
tf.enable_eager_execution()
# fix random seed for reproducibility
seed = 4
tf.set_random_seed(seed)
from jpcnn.config import JPCNNConfig
from jpcnn.data import get_dataset, BATCH_SIZE
from jpcnn.dct_utils import (
    flat_compress, flat_reconstruct, basic_compression,
    mean_inv_compression,
)
from jpcnn.file_utils import (
    build_checkpoint_file_name,
    load_or_save_conf,
)
from jpcnn.image_utils import save_and_display_images
from jpcnn.model import model, pixel_cnn_loss
from jpcnn.nn import get_shape_as_list, assert_finite
import time
import numpy as np

from jpcnn.nn import (
    discretized_mix_logistic_loss,
    sample_from_discretized_mix_logistic,
)


def generate_and_save_images(model, epoch, test_input, container, root_dir, compression, mixtures_per_channel, display_images=False):
    height = test_input.shape[1]
    width = test_input.shape[2]
    channels = test_input.shape[-1]
    predictions = np.zeros_like(test_input)
    cap_adjustments = []
    for j in range(height):
        for i in range(width):
            with container.as_default():
                ij_likelihood = model(predictions)[:, j, i, :]
            ij_sample = sample_from_discretized_mix_logistic(ij_likelihood, [mixtures_per_channel] * channels)
            predictions[:,j,i,:] = ij_sample

            # crop values to [0,1] interval:
            reconstruction = flat_reconstruct(predictions, compression)
            capped_sample = tf.maximum(tf.minimum(reconstruction, 1), 0)
            recompression = flat_compress(capped_sample, compression)
            cap_adjustments.append(float(tf.reduce_max(tf.abs(ij_sample - recompression[:,j,i,:]))))
            predictions[:,j,i,:] = recompression[:,j,i,:]

    print("Cap adjustment: %s" % max(cap_adjustments))
    decompressed_images = flat_reconstruct(predictions, compression)[:, :, :, 0]
    save_and_display_images(root_dir, 'image_at_epoch_{:04d}.png'.format(epoch),
                            decompressed_images, display = display_images)


def train(train_dataset, val_dataset, conf: JPCNNConfig, ckpt_file: str=None, access_token=None):
    rng = np.random.RandomState(conf.seed)
    noise = rng.beta(1,1,[16, conf.image_dim, conf.image_dim, 1]).astype("float32")
    noise = image_processors(conf.compression)[0](noise)
    comp_shapes = get_shape_as_list(conf.compression)
    num_blocks = comp_shapes[0] * comp_shapes[1]
    sample_labels = None
    optimizer = tf.train.AdamOptimizer(conf.lr)
    container = tf.contrib.eager.EagerVariableStore()

    conf, dir_name, do_sync = load_or_save_conf(ckpt_file, conf, access_token)
    summary_writer = tf.contrib.summary.create_file_writer("{}/logs".format(dir_name), flush_millis = 10000)
    summary_writer.set_as_default()

    # Data dependent initialization:
    for i, data in enumerate(train_dataset):
        if isinstance(data, tuple):
            images, labels = data
        else:
            images = data
            labels = None
        with container.as_default():
            image_var = tf.contrib.eager.Variable(images)
            model(image_var, labels, training = True, num_layers = conf.num_layers,
                  avg_num_filters = conf.num_filters, num_resnet = conf.num_resnet,
                  mixtures_per_channel = conf.mixtures_per_channel,
                  init = True, compression = conf.compression)
        # pull sample labels:
        if i == 0 and labels is not None:
            sample_labels = labels[:16]

    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(
        var_list = container.trainable_variables()+[global_step],
        save_relative_paths = True,
        sharded = False
    )
    if ckpt_file is not None:
        saver.restore(None, ckpt_file)
    while global_step < conf.epochs:
        global_step.assign_add(1)
        print("Starting Epoch: {0:d}".format(int(global_step)))
        start = time.time()
        total_train_loss = []
        for data in train_dataset:
            if isinstance(data, tuple):
                images, labels = data
            else:
                images = data
                labels = None

            with tf.GradientTape() as gr_tape, container.as_default():
                image_var = tf.contrib.eager.Variable(images)
                im_shape = list(map(int, tf.shape(image_var)))

                logits = model(image_var, labels,
                               training = True,
                               num_layers=conf.num_layers,
                               avg_num_filters=conf.num_filters,
                               num_resnet=conf.num_resnet,
                               compression = conf.compression,
                               mixtures_per_channel = conf.mixtures_per_channel)

                loss = tf.reduce_mean(discretized_mix_logistic_loss(
                    logits, images, [conf.mixtures_per_channel] * im_shape[-1])
                )
                rescaled_loss = mean_inv_compression(conf.compression) * loss
                assert_finite(loss)
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("loss", loss)
                tf.contrib.summary.scalar("rescaled_loss", rescaled_loss)
            total_train_loss.append(np.array(rescaled_loss))
            print(float(rescaled_loss))
            print("trying to compute gradients")
            gradients = gr_tape.gradient(
                loss,
                container.trainable_variables()
            )
            print("done computing gradients")

            optimizer.apply_gradients(
                zip(gradients, container.trainable_variables())
            )
            print("done applying gradients")
            with tf.contrib.summary.always_record_summaries():
                def safe_div(x, y):
                    if x is None:
                        return 0
                    rel_grads = (abs(x) / (abs(y) + 1e-10)).numpy().reshape(-1)
                    valid_grads = rel_grads[tf.is_finite(rel_grads)]
                    avg_rel_grad = tf.reduce_mean(valid_grads)
                    if not tf.is_finite(avg_rel_grad):
                        print("error, non finite grad")
                        return 0
                    return avg_rel_grad
                # print("Finite Gradients: %s"% tf.reduce_all(tf.is_finite(gradients)))
                # print("Finite Variables: %s"% tf.reduce_all(tf.is_finite(container.trainable_variables())))
                rel_grads = list(map(safe_div, gradients, container.trainable_variables()))
                tf.contrib.summary.histogram("relative_gradients", rel_grads)

        if int(global_step) % conf.ckpt_interval == 0:
            # display.clear_output(wait=True)
            generate_and_save_images(
                lambda pred: model(pred, sample_labels, num_layers=conf.num_layers,
                                   avg_num_filters=conf.num_filters,
                                   num_resnet=conf.num_resnet,
                                   compression=conf.compression,
                                   mixtures_per_channel = conf.mixtures_per_channel),
                int(global_step),
                noise,
                container,
                dir_name,
                conf.compression,
                conf.mixtures_per_channel,
                display_images = conf.display_images
            )
            ckpt_name = build_checkpoint_file_name(dir_name, conf.description)
            fp = saver.save(None, ckpt_name, global_step=global_step)
            print("Model Saved at {}".format(fp))
            if conf.dropbox_sync:
                do_sync()

            # Compute test_set loss:
            total_val_loss = []
            for test_data in val_dataset:
                if isinstance(test_data, tuple):
                    test_images, test_labels = test_data
                else:
                    test_images = test_data
                    test_labels = None
                with container.as_default():
                    image_var = tf.contrib.eager.Variable(test_images)
                    im_shape = list(map(int, tf.shape(image_var)))

                    logits = model(image_var, test_labels, training = False,
                                             num_layers=conf.num_layers,
                                             avg_num_filters=conf.num_filters,
                                             num_resnet=conf.num_resnet,
                                             compression=conf.compression,
                                             mixtures_per_channel =
                                             conf.mixtures_per_channel)

                    loss = tf.reduce_mean(discretized_mix_logistic_loss(
                        logits, test_images,
                        [conf.mixtures_per_channel] * im_shape[-1])
                    )
                    rescaled_loss = mean_inv_compression(conf.compression) * loss
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("validation loss", loss)
                    tf.contrib.summary.scalar("rescaled validation loss", rescaled_loss)
                total_val_loss.append(rescaled_loss)
            avg_val_loss = tf.reduce_mean(total_val_loss)
            print('Validation Loss for epoch {} is {}'.format(int(global_step), avg_val_loss))


        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(int(global_step),
                                                      time.time()-start))

        print('Training Loss for epoch {} is {}'.format(int(global_step), np.mean(total_train_loss)))
    # display.clear_output(wait = True)
    generate_and_save_images(
        lambda pred: model(pred, sample_labels, num_layers=conf.num_layers,
                           avg_num_filters=conf.num_filters,
                           num_resnet=conf.num_resnet,
                           compression=conf.compression,
                           mixtures_per_channel =
                           conf.mixtures_per_channel),
        conf.epochs,
        noise, 
        container,
        dir_name,
        conf.compression,
        conf.mixtures_per_channel,
        display_images = conf.display_images
    )
def image_processors(compression):
    def compressor(im):
        padded_im = im #tf.pad(im, [[0, 0], [0, 0], [0, 0], [0, 1]], "CONSTANT",
                        #   constant_values = 1)  # add channel of ones to distinguish image from padding later on
        return flat_compress(padded_im, compression)
    def reconstructor(c_im):
        return flat_reconstruct(c_im, compression)
    return compressor, reconstructor

if __name__ == "__main__":
    compression = basic_compression(.1, .3, [2, 2])
    full_dataset, image_dim, buffer_size = get_dataset(
        basic_test_data = True,
        image_processors = image_processors(compression)
    )
    num_test_elements = buffer_size//BATCH_SIZE//5
    print("Training Set Size: %s" % (buffer_size - num_test_elements * BATCH_SIZE))
    print("Validation Set Size: %s" % (num_test_elements * BATCH_SIZE))
    val_dataset = full_dataset.take(num_test_elements)
    train_dataset = full_dataset.skip(num_test_elements)
    # train_dataset=full_dataset
    # val_dataset=full_dataset
    dropbox_access_token = None
    if len(sys.argv) > 1:
        dropbox_access_token = sys.argv[1]
        print("Access Token: %s" % dropbox_access_token)
    train(train_dataset, val_dataset, JPCNNConfig(
        image_dim=image_dim,
        compression = compression,
        seed = seed,
        num_test_elements = num_test_elements
    ), access_token = dropbox_access_token)
    # train(train_dataset, JPCNNConfig(image_dim=image_dim), ckpt_file = "Checkpoint-20181011-004410/params_tmp.ckpt-15")
