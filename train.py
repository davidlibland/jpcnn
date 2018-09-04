import tensorflow as tf
from data import get_dataset, BATCH_SIZE
from model import model, optimizer, pixel_cnn_loss
import time
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()


# model = PixelCNN(5, 2)
num_layers = 2
num_filters = 5


# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

train_dataset = get_dataset()
# model.build(train_dataset.shape)
# print(len(list(iter(train_dataset))))

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
# model.fit(train_dataset.astype(np.float), train_dataset, epochs=1000, batch_size=BATCH_SIZE)


# grads = tf.keras.backend.gradients(model.output, model.offset_initial_upper_feed.pad_layer.input)
# print(grads)

def generate_and_save_images(model, epoch, test_input):
    height = test_input.shape[1]
    width = test_input.shape[2]
    predictions = np.zeros_like(test_input.copy())
    for j in range(height):
        for i in range(width):
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

all_params = []
model = tf.make_template('model', model, variables = all_params)
x_init = tf.constant(np.random.beta(1,1,[BATCH_SIZE,28,28,1]))
# run once for data dependent initialization of parameters
# all_params = []
# init_pass = model(x_init, num_filters = num_filters, num_layers = num_layers, variables = all_params)

# keep track of moving average
# all_params = tf.trainable_variables()

def train(dataset, epochs, image_dim):
    noise = np.random.beta(1,1,[16,28,28,1])
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:

            with tf.GradientTape() as gr_tape:
                generated_images = model(images, training = True, num_layers = num_layers, num_filters = num_filters)

                loss = pixel_cnn_loss(images, generated_images)

            gradients = gr_tape.gradient(
                loss,
                all_params
            )

            optimizer.apply_gradients(
                zip(gradients, all_params)
            )

        if epoch % 1 == 0:
            # display.clear_output(wait=True)
            generate_and_save_images(model, epoch + 1, noise)

        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                      time.time()-start))

    # display.clear_output(wait = True)
    generate_and_save_images(model,
                             epochs,
                             noise)

train(train_dataset, 10, 28)

