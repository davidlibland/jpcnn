import tensorflow as tf
from data import get_dataset, BATCH_SIZE
from model import model, optimizer, pixel_cnn_loss
import time
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

num_layers = 0
num_filters = 50


train_dataset = get_dataset()


def generate_and_save_images(model, epoch, test_input, container):
    height = test_input.shape[1]
    width = test_input.shape[2]
    predictions = np.zeros_like(test_input)
    for j in range(height):
        for i in range(width):
            with container.as_default():
                ij_likelihood = model(predictions, num_layers = num_layers, num_filters = num_filters)[:, j, i, :]
            # print(ij_likelihood.mean(), ij_likelihood.std())
            ij_sample = np.random.binomial(1, ij_likelihood)
            predictions[:,j,i,:] = ij_sample

    fig = plt.figure(figsize = (4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train(dataset, epochs, image_dim, num_layers=num_layers, num_filters=num_filters):
    noise = np.random.beta(1,1,[16, image_dim, image_dim, 1])
    container = tf.contrib.eager.EagerVariableStore()

    # Data dependent initialization:
    for i, images in enumerate(dataset):
        with container.as_default():
            image_var = tf.contrib.eager.Variable(images)
            model(image_var, training = True, num_layers = num_layers,
                  num_filters = num_filters, init = True)

    for epoch in range(epochs):
        start = time.time()
        total_loss = []
        for images in dataset:

            with tf.GradientTape() as gr_tape, container.as_default():
                image_var = tf.contrib.eager.Variable(images)

                generated_images = model(image_var, training = True, num_layers = num_layers, num_filters = num_filters)

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
            generate_and_save_images(model, epoch + 1, noise, container)

        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                      time.time()-start))

        print('Loss for epoch {} is {}'.format(epoch + 1, np.mean(total_loss)))
    # display.clear_output(wait = True)
    generate_and_save_images(model,
                             epochs,
                             noise, container)


if __name__ == "__main__":
    train(train_dataset, 500, 28)
