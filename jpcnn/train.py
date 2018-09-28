import tensorflow as tf
from jpcnn.data import get_dataset
from jpcnn.file_utils import (
    build_checkpoint_file_name,
    load_or_save_conf,
)
from jpcnn.image_utils import display_images
from jpcnn.model import model, pixel_cnn_loss
import time
import numpy as np
from dataclasses import dataclass

tf.enable_eager_execution()


@dataclass
class JPCNNConfig:
    image_dim: int
    num_layers: int=2
    num_resnet: int=2
    num_filters: int=64
    lr: float=1e-2
    epochs: int=500
    description: str="tmp"
    ckpt_interval: int=1


def generate_and_save_images(model, epoch, test_input, container):
    height = test_input.shape[1]
    width = test_input.shape[2]
    predictions = np.zeros_like(test_input)
    for j in range(height):
        for i in range(width):
            with container.as_default():
                ij_likelihood = model(predictions)[:, j, i, :]
            # print(ij_likelihood.mean(), ij_likelihood.std())
            ij_sample = np.random.binomial(1, ij_likelihood)
            predictions[:,j,i,:] = ij_sample

    display_images('image_at_epoch_{:04d}.png'.format(epoch), predictions[:,:,:,0])


def train(dataset, conf: JPCNNConfig, ckpt_file: str=None):
    print("image dim: %s " % image_dim)
    noise = np.random.beta(1,1,[16, image_dim, image_dim, 1])
    optimizer = tf.train.AdamOptimizer(conf.lr)
    container = tf.contrib.eager.EagerVariableStore()

    conf, dir_name, start_epoch = load_or_save_conf(ckpt_file, conf)
    # Data dependent initialization:
    for i, images in enumerate(dataset):
        with container.as_default():
            image_var = tf.contrib.eager.Variable(images)
            model(image_var, training = True, num_layers = conf.num_layers,
                  num_filters = conf.num_filters, num_resnet = conf.num_resnet,
                  init = True)

    saver = tf.train.Saver(
        var_list = container.trainable_variables(),
        save_relative_paths = True
    )
    if ckpt_file is not None:
        saver.restore(None, ckpt_file)
    for epoch in range(start_epoch, conf.epochs):
        start = time.time()
        total_loss = []
        for images in dataset:

            with tf.GradientTape() as gr_tape, container.as_default():
                image_var = tf.contrib.eager.Variable(images)

                generated_images = model(image_var, training = True,
                                         num_layers=conf.num_layers,
                                         num_filters=conf.num_filters,
                                         num_resnet=conf.num_resnet)

                loss = pixel_cnn_loss(images, generated_images)
            total_loss.append(np.array(loss))

            gradients = gr_tape.gradient(
                loss,
                container.trainable_variables()
            )

            optimizer.apply_gradients(
                zip(gradients, container.trainable_variables())
            )

        if epoch % conf.ckpt_interval == 0:
            # display.clear_output(wait=True)
            generate_and_save_images(
                lambda pred: model(pred, num_layers=conf.num_layers,
                                   num_filters=conf.num_filters,
                                   num_resnet=conf.num_resnet),
                epoch + 1,
                noise,
                container
            )
            ckpt_name = build_checkpoint_file_name(dir_name, conf.description,
                                                   epoch)
            saver.save(None, ckpt_name)

        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                      time.time()-start))

        print('Loss for epoch {} is {}'.format(epoch + 1, np.mean(total_loss)))
    # display.clear_output(wait = True)
    generate_and_save_images(
        lambda pred: model(pred, num_layers=conf.num_layers,
                           num_filters=conf.num_filters,
                           num_resnet=conf.num_resnet),
        conf.epochs,
        noise, 
        container
    )


if __name__ == "__main__":
    train_dataset, image_dim = get_dataset(basic_test_data = True)
    train(train_dataset, JPCNNConfig(image_dim=image_dim), ckpt_file = "Checkpoint-20180928-113325/params_tmp_5.ckpt")