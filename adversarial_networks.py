import glob
import time

import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, models, backend as k
import matplotlib.pyplot as plt
import bubble_prediction
import create_network
import loss_functions
import dat_to_training


def generate_images(model, epoch, input_images_index, name):
    images = np.zeros((16, 4, 64, 64, 1))
    for i in range(len(input_images_index)):
        for j in range(0, 4):
            images[i, j, :, :, 0] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_False_0_0.001_12/data_{}.npy".format(input_images_index[i] + j)
                , 64)[:, :, 1]
    predictions = model(images, training=False)

    fig = plt.figure(figsize=(5, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='Blues')
        plt.axis('off')
    plt.savefig("{}_predictions_at_epoch_{:04d}.png".format(name, epoch))
    plt.close('all')


def evaluate_performance(network_name, frames, size, timestep, resolution,
                         test_range=300, simulation=12, start_point=5, variant=0, flipped=False):
    network = models.load_model("models/{}".format(network_name))

    # Setting up the network's input
    starting_frames = np.zeros((1, frames, size, size, 1))
    for frame in range(frames):
        starting_frames[0, frame, :, :, 0] = dat_to_training.process_bmp(
            "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
                str(flipped), variant, resolution, simulation, start_point + frame * timestep
            ), image_size=size
        )[:, :, 1]

    # Running the predictions
    final_frames = np.zeros((test_range, size, size))
    current_frames = starting_frames
    for loop in range(test_range):
        next_frame = network(current_frames)
        for frame in range(frames - 1):
            current_frames[:, frame] = current_frames[:, frame + 1]
        current_frames[:, frames-1] = next_frame
        final_frames[loop] = next_frame[0, :, :, 0]

    # Getting the correct data
    max_data = len(glob.glob("Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/*".format(
        str(flipped), variant, resolution, simulation
    )))

    correct_frames = []
    for i in range(start_point + frames * timestep, max_data, timestep):
        correct_frames.append(dat_to_training.process_bmp(
            "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
                str(flipped), variant, resolution, simulation, i
            ), image_size=size
        )[:, :, 1])
    num_correct_frames = len(correct_frames)

    # Making a composite of both
    composite_frames = np.zeros((max(num_correct_frames, test_range), size, size, 3))
    for composite_frame in range(max(num_correct_frames, test_range)):
        if composite_frame < num_correct_frames and composite_frame < test_range:
            composite_frames[composite_frame, :, :, 0] = correct_frames[composite_frame]
            composite_frames[composite_frame, :, :, 2] = final_frames[composite_frame]
        elif composite_frame >= num_correct_frames:
            composite_frames[composite_frame, :, :, 0] = correct_frames[-1]
            composite_frames[composite_frame, :, :, 2] = final_frames[composite_frame]
        elif composite_frame >= test_range:
            composite_frames[composite_frame, :, :, 0] = correct_frames[composite_frame]
            composite_frames[composite_frame, :, :, 2] = final_frames[-1]

    # Saving as an image
    image_converts = composite_frames * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    imageio.mimsave("model_performance/{}_composite.gif".format(network_name), images)

    # Performing y-position stuff




def main():
    image_size = 64
    image_frames = 4
    timestep = 5
    future_runs = 5
    resolution = 0.001

    scenario = 2
    if scenario < 2:
        training_data = dat_to_training.generate_data(image_frames, image_size, timestep, future_runs, [0], False, resolution, [12])

        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            decay_rate=0.7
        )
        if scenario == 0:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)

            network = create_network.create_u_network(layers.LeakyReLU(), image_frames, image_size, encode_size=10,
                                                      kernel_size=5, channels=1)
            discriminator = create_network.create_discriminator(2, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer, 50, "u-net",
                          future_runs, image_frames)
            network.save("models/u_network")
        elif scenario == 1:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            network = create_network.create_basic_network(layers.LeakyReLU(), image_frames, image_size, channels=1)
            discriminator = create_network.create_discriminator(2, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer, 50, "basic",
                          future_runs, image_frames)
            network.save("models/basic_network")

    evaluate_performance("u_network", image_frames, image_size, timestep, resolution)


@tf.function
def train_step(input_images, expected_output, network, discriminator, net_op, disc_op, future_runs, frames):
    with tf.GradientTape() as net_tape, tf.GradientTape() as disc_tape:
        current_output = []
        future_input = input_images
        predictions = network(input_images, training=True)
        current_output.append(predictions)

        for future_step in range(1, future_runs):
            next_input = []
            for i in range(frames-1):
                next_input.append(future_input[:, i+1])
            next_input.append(tf.cast(predictions, tf.float64))
            future_input = tf.stack(next_input, axis=1)
            predictions = network(future_input, training=True)
        current_output.append(predictions)

        overall_predictions = tf.stack(current_output, axis=1)
        real_output = discriminator(expected_output, training=True)
        actual_output = discriminator(overall_predictions, training=True)
        network_disc_loss = loss_functions.generator_loss(actual_output)
        network_mse = k.mean(losses.mean_squared_error(expected_output, overall_predictions), axis=0)
        disc_loss = loss_functions.discriminator_loss(real_output, actual_output)
        network_loss = network_disc_loss + network_mse

    net_grad = net_tape.gradient(network_loss, network.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    net_op.apply_gradients(zip(net_grad, network.trainable_variables))
    disc_op.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

    return network_loss, disc_loss


def train_network(dataset, network, discriminator, net_op, disc_op, epochs, name, future_runs, frames):
    times_so_far = []
    ref_index = [4, 42, 69, 72, 104, 254, 298, 339, 347, 420, 481, 482, 555, 663, 681, 701]
    ref_index_float = np.linspace(3, 800, 16, endpoint=True)
    for i, r in enumerate(ref_index_float):
        ref_index[i] = int(r)
    overall_loss_gen = []
    overall_loss_disc = []
    for epoch in range(epochs):
        start = time.time()
        gen_losses = []
        disc_losses = []
        print("Running epoch {}...".format(epoch + 1))
        for questions, answers in dataset:
            gen_loss, disc_loss = train_step(questions, answers,
                                             network, discriminator,
                                             net_op, disc_op,
                                             future_runs, frames)
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
        overall_loss_disc.append(np.mean(disc_losses))
        overall_loss_gen.append(np.mean(gen_losses))
    generate_images(network, epochs, ref_index, name)
    plt.close("all")
    plt.grid()
    plt.plot(overall_loss_gen, label="Generator loss")
    plt.plot(overall_loss_disc, label="Discriminator loss")
    plt.legend()
    plt.savefig("{}_losses.png".format(name), dpi=500)
    plt.clf()


if __name__ == "__main__":
    main()
