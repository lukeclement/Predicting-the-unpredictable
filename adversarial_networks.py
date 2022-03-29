import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, backend as k
import matplotlib.pyplot as plt
import bubble_prediction
import create_network
import loss_functions
import dat_to_training


def generate_images(model, epoch, input_images_index, name):
    images = np.zeros((16, 4, 64, 64, 3))
    for i in range(len(input_images_index)):
        for j in range(0, 4):
            images[i, j, :, :, :] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_False_0_0.001_12/data_{}.npy".format(input_images_index[i] + j), 64)
    predictions = model(images, training=False)

    fig = plt.figure(figsize=(5, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='Blues')
        plt.axis('off')
    plt.savefig("{}_predictions_at_epoch_{:04d}.png".format(name, epoch))


def main():
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9
    )
    network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
    discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)

    image_size = 64
    image_frames = 4
    timestep = 5
    resolution = 0.001

    network = create_network.create_basic_network(layers.LeakyReLU(), image_frames, image_size)
    discriminator = create_network.create_special_discriminator(image_size)

    print(network.summary())
    print(type(network))
    print(type(discriminator))
    training_data = dat_to_training.create_training_data(
        image_frames, timestep, image_size=image_size,
        excluded_sims=[12], variants=[0], resolution=resolution, flips_allowed=False, easy_mode=False)
    print(discriminator.summary())
    train_network(training_data[0], network, discriminator, network_optimizer, discriminator_optimizer, 500, 'basic')
    network.save("models/basic_network")
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9
    )
    network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
    discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)

    network = create_network.create_u_network(layers.LeakyReLU(), image_frames, image_size, encode_size=10,
                                              kernel_size=5)
    discriminator = create_network.create_special_discriminator(image_size)
    print(network.summary())
    train_network(training_data[0], network, discriminator, network_optimizer, discriminator_optimizer, 500, "u-net")
    network.save("models/u_network")


@tf.function
def train_step(input_images, expected_output, network, discriminator, net_op, disc_op):
    with tf.GradientTape() as net_tape, tf.GradientTape() as disc_tape:
        predictions = network(input_images, training=True)

        real_output = discriminator(expected_output, training=True)
        actual_output = discriminator(predictions, training=True)
        network_disc_loss = loss_functions.generator_loss(actual_output)
        network_mse = k.mean(losses.mean_squared_error(expected_output, predictions), axis=0)
        network_loss = network_disc_loss
        disc_loss = loss_functions.discriminator_loss(real_output, actual_output)

    net_grad = net_tape.gradient(network_loss, network.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    net_op.apply_gradients(zip(net_grad, network.trainable_variables))
    disc_op.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

    return network_loss, disc_loss


def train_network(dataset, network, discriminator, net_op, disc_op, epochs, name):
    times_so_far = []
    ref_index = [4, 42, 69, 72, 104, 254, 298, 339, 347, 420, 481, 482, 555, 663, 681, 701]
    ref_index_float = np.linspace(3, 810, 16, endpoint=True)
    for i, r in enumerate(ref_index_float):
        ref_index[i] = int(r)
    for epoch in range(epochs):
        start = time.time()
        gen_losses = []
        disc_losses = []
        print("Running epoch {}...".format(epoch + 1))
        for questions, answers in dataset:
            gen_loss, disc_loss = train_step(questions, answers,
                                             network, discriminator,
                                             net_op, disc_op)
            gen_losses.append(k.mean(gen_loss))
            disc_losses.append(k.mean(disc_loss))
        generate_images(network, epoch + 1, ref_index, name)
        times_so_far.append(time.time() - start)
        print("Time for epoch {} was {:.0f}s".format(epoch + 1, times_so_far[epoch]))
        print("Losses were {:.4f} for generator and {:.4f} for discriminator".format(
            np.mean(gen_losses), np.mean(disc_losses))
        )
        if np.mean(gen_losses) > np.mean(disc_losses):
            print("Discriminator winning!")
        else:
            print("Generator winning!")
        print("ETA currently stands at {:.1f}min from now".format(
            np.mean(times_so_far) * (epochs - epoch - 1) / 60))

    generate_images(network, epochs, ref_index, name)


if __name__ == "__main__":
    main()
