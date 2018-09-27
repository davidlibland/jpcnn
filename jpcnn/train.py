import tensorflow as tf
from jpcnn.data import get_dataset
from jpcnn.image_utils import display_images
from jpcnn.model import model, optimizer, pixel_cnn_loss
import time
import numpy as np

tf.enable_eager_execution()

num_layers = 1
num_filters = 50
num_resnet = 1


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


def train(dataset, epochs, image_dim, num_filters=num_filters, num_layers=num_layers, num_resnet=num_resnet):
    print("image dim: %s " % image_dim)
    noise = np.random.beta(1,1,[16, image_dim, image_dim, 1])
    container = tf.contrib.eager.EagerVariableStore()

    # Data dependent initialization:
    for i, images in enumerate(dataset):
        with container.as_default():
            image_var = tf.contrib.eager.Variable(images)
            model(image_var, training = True, num_layers = num_layers,
                  num_filters = num_filters, num_resnet = num_resnet, init = True)

    for epoch in range(epochs):
        start = time.time()
        total_loss = []
        for images in dataset:

            with tf.GradientTape() as gr_tape, container.as_default():
                image_var = tf.contrib.eager.Variable(images)

                generated_images = model(image_var, training = True, num_layers = num_layers, num_filters = num_filters, num_resnet=num_resnet)

                loss = pixel_cnn_loss(images, generated_images)
            total_loss.append(np.array(loss))

            gradients = gr_tape.gradient(
                loss,
                container.trainable_variables()
            )

            optimizer.apply_gradients(
                zip(gradients, container.trainable_variables())
            )

        if epoch % 1 == 0:
            # display.clear_output(wait=True)
            generate_and_save_images(
                lambda pred: model(pred, num_layers = num_layers,
                                   num_filters = num_filters, num_resnet=num_resnet),
                epoch + 1,
                noise,
                container
            )

        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                      time.time()-start))

        print('Loss for epoch {} is {}'.format(epoch + 1, np.mean(total_loss)))
    # display.clear_output(wait = True)
    generate_and_save_images(
        lambda pred: model(pred, num_layers = num_layers,
                           num_filters = num_filters, num_resnet=num_resnet),
        epochs,
        noise, 
        container
    )


if __name__ == "__main__":
    train_dataset, image_dim = get_dataset(basic_test_data = True)
    train(train_dataset, 500, image_dim)