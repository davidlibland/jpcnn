import tensorflow as tf
from jpcnn.config import JPCNNConfig
from jpcnn.data import get_dataset
from jpcnn.dct_utils import (
    jpeg_compression, jpeg_reconstruction,
    flat_compress, flat_reconstruct,
)
from jpcnn.file_utils import (
    build_checkpoint_file_name,
    load_or_save_conf,
)
from jpcnn.image_utils import display_images
from jpcnn.model import model, pixel_cnn_loss
import time
import numpy as np

from jpcnn.nn import (
    discretized_mix_logistic_loss,
    sample_from_discretized_mix_logistic,
)

tf.enable_eager_execution()

jpeg_block_size = 2
compression = np.ones([jpeg_block_size, jpeg_block_size])
in_shape = None


def generate_and_save_images(model, epoch, test_input, container, root_dir):
    height = test_input.shape[1]
    width = test_input.shape[2]
    channels = test_input.shape[-1]
    predictions = np.zeros_like(test_input)
    for j in range(height):
        for i in range(width):
            with container.as_default():
                ij_likelihood = model(predictions)[:, j, i, :]
            # print(ij_likelihood.mean(), ij_likelihood.std())
            ij_sample = sample_from_discretized_mix_logistic(ij_likelihood, [1] * channels)
            predictions[:,j,i,:] = ij_sample
    print(tf.reduce_min(predictions), tf.reduce_max(predictions))
    display_images(root_dir, 'image_at_epoch_{:04d}.png'.format(epoch), flat_reconstruct(predictions, [jpeg_block_size]*2, compression)[:,:,:,0] )


def train(dataset, conf: JPCNNConfig, ckpt_file: str=None):
    noise = np.random.beta(1,1,[16, conf.image_dim, conf.image_dim, 1]).astype("float32")
    noise = flat_compress(noise, [jpeg_block_size] * 2, compression)
    optimizer = tf.train.AdamOptimizer(conf.lr)
    container = tf.contrib.eager.EagerVariableStore()

    conf, dir_name, do_sync = load_or_save_conf(ckpt_file, conf)
    summary_writer = tf.contrib.summary.create_file_writer("{}/logs".format(dir_name), flush_millis = 10000)
    summary_writer.set_as_default()

    # Data dependent initialization:
    for i, images in enumerate(dataset):
        images = flat_compress(images, [jpeg_block_size]*2, compression)
        with container.as_default():
            image_var = tf.contrib.eager.Variable(images)
            model(image_var, training = True, num_layers = conf.num_layers,
                  num_filters = conf.num_filters, num_resnet = conf.num_resnet,
                  init = True)

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
        total_loss = []
        for images in dataset:
            images = flat_compress(images, [jpeg_block_size]*2, compression)

            with tf.GradientTape() as gr_tape, container.as_default():
                image_var = tf.contrib.eager.Variable(images)
                im_shape = list(map(int, tf.shape(image_var)))

                logits = model(image_var, training = True,
                                         num_layers=conf.num_layers,
                                         num_filters=conf.num_filters,
                                         num_resnet=conf.num_resnet)

                loss = tf.reduce_mean(discretized_mix_logistic_loss(logits, images, [1] * im_shape[-1]))
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("loss", loss)
            total_loss.append(np.array(loss))
            print(float(loss))

            gradients = gr_tape.gradient(
                loss,
                container.trainable_variables()
            )

            optimizer.apply_gradients(
                zip(gradients, container.trainable_variables())
            )
            with tf.contrib.summary.always_record_summaries():
                def safe_div(x, y):
                    if x is None:
                        return 0
                    return tf.reduce_mean(abs(x) / (abs(y) + 1e-10))
                rel_grads = list(map(safe_div, gradients, container.trainable_variables()))
                tf.contrib.summary.histogram("relative_gradients", rel_grads)

        if int(global_step) % conf.ckpt_interval == 0:
            # display.clear_output(wait=True)
            generate_and_save_images(
                lambda pred: model(pred, num_layers=conf.num_layers,
                                   num_filters=conf.num_filters,
                                   num_resnet=conf.num_resnet),
                int(global_step),
                noise,
                container,
                dir_name
            )
            ckpt_name = build_checkpoint_file_name(dir_name, conf.description)
            fp = saver.save(None, ckpt_name, global_step=global_step)
            print("Model Saved at {}".format(fp))
            # do_sync()

        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(int(global_step),
                                                      time.time()-start))

        print('Loss for epoch {} is {}'.format(int(global_step), np.mean(total_loss)))
    # display.clear_output(wait = True)
    generate_and_save_images(
        lambda pred: model(pred, num_layers=conf.num_layers,
                           num_filters=conf.num_filters,
                           num_resnet=conf.num_resnet),
        conf.epochs,
        noise, 
        container,
        dir_name
    )


if __name__ == "__main__":
    train_dataset, image_dim = get_dataset(basic_test_data = True)
    # train_dataset = tf.batch(lambda x: flat_compress(x, [jpeg_block_size]*2, compression), train_dataset)
    train(train_dataset, JPCNNConfig(image_dim=image_dim))
    # train(train_dataset, JPCNNConfig(image_dim=image_dim), ckpt_file = "Checkpoint-20181011-004410/params_tmp.ckpt-15")