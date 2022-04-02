import glob
import time

import imageio
import numpy as np
import scipy.optimize
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, models, backend as k
import matplotlib.pyplot as plt
import bubble_prediction
import create_network
import loss_functions
import dat_to_training
import testing_weather


def generate_weather(model, epoch, name, input_frames):
    initial_frames = input_frames
    prediction = model(initial_frames)
    plt.imshow(prediction[0], cmap='Blues')
    plt.axis('off')
    plt.savefig("{}_predictions_at_epoch_{:04d}.png".format(name, epoch))
    plt.close('all')


def generate_images(model, epoch, input_images_index, name):
    images = np.zeros((16, 4, 64, 64, 1))
    for i in range(len(input_images_index)):
        for j in range(0, 4):
            images[i, j, :, :, 0] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_False_0_0.001_12/data_{}.npy".format(input_images_index[i] + j * 5)
                , 64)[:, :, 1]
    predictions = model(images, training=False)

    fig = plt.figure(figsize=(5, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='Blues')
        plt.axis('off')
    plt.savefig("{}_predictions_at_epoch_{:04d}.png".format(name, epoch))
    plt.close('all')


def calculate_com(bubble):
    image_size = np.shape(bubble)[0]
    x = np.linspace(0, 1, image_size)
    y = np.linspace(1, 0, image_size)
    x, y = np.meshgrid(x, y, indexing='xy')
    x_com = np.sum(x * bubble) / np.sum(bubble)
    y_com = np.sum(y * bubble) / np.sum(bubble)
    return x_com, y_com


def sinfunc(t, A, w, p, c):
    return A * np.sin(w * t + p) + c


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2. * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
            "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}


def evaluate_weather(network_name, frames, size):
    network = models.load_model("models/{}".format(network_name))

    data = np.load("Meterology_data/data8.npz")
    data_useful = np.zeros((np.shape(data["data"])[1], size, size, 1))
    data_useful[:, :, :, 0] = data["data"][0, :, 69:69+size, 420:420+size]
    data_useful = np.tanh(data_useful/500)
    initial_frames = np.zeros((1, frames, size, size, 1))
    initial_frames[0] = data_useful[:frames]
    rest_of_data = data_useful[frames:]
    given_frames = initial_frames
    final_data = np.zeros((np.shape(rest_of_data)[0], size, size, 3))
    print(np.shape(rest_of_data))
    final_data[:, :, :, 0] = rest_of_data[:, :, :, 0]
    for clouds in range(np.shape(rest_of_data)[0]):
        next_frame = network(given_frames)
        for frame in range(frames-1):
            given_frames[0, frame] = given_frames[0, frame + 1]
        given_frames[0, frames-1] = next_frame
        final_data[clouds, :, :, 2] = next_frame[0, :, :, 0]

    image_converts = final_data * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    imageio.mimsave("model_performance/{}_weather_composite.gif".format(network_name), images)
    image_converts = final_data[:, :, :, 2] * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    imageio.mimsave("model_performance/{}_weather_solo.gif".format(network_name), images)


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
        current_frames[:, frames - 1] = next_frame
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
    composite_size = max(num_correct_frames, test_range)
    composite_frames = np.zeros((composite_size, size, size, 3))
    for composite_frame in range(composite_size):
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
    imageio.mimsave("model_performance/{}_{}_composite.gif".format(network_name, simulation), images)

    # For future use
    correct = composite_frames[:, :, :, 0]
    prediction = composite_frames[:, :, :, 2]
    # Getting a value on the distribution of points per frame
    # plt.xlabel("Pixel values")
    # plt.hist2d(correct[:][correct[:] > 0], np.linspace(0, 1, len(correct[:][correct[:] > 0])), bins=30)
    # # plt.hist(prediction[prediction > 0.2], bins=30)
    # plt.show()

    frame_info = []
    correct_info = []
    prediction_info = []
    for frame in range(composite_size):
        corrected_data = np.reshape(correct[frame], (size * size))
        predicted_data = np.reshape(prediction[frame], (size ** 2))
        for index in range(size ** 2):
            frame_info.append(frame)
            correct_info.append(corrected_data[index])
            prediction_info.append(predicted_data[index])
    plt.xlabel("Pixel value")
    plt.ylabel("Frame number")
    plt.hist2d(prediction_info, frame_info,
               bins=(18, min(composite_size, test_range)),
               range=((0.1, 1), (0, min(composite_size, test_range))))
    plt.colorbar()
    plt.savefig("model_performance/{}_{}_value_dist.png".format(network_name, simulation), dpi=250)
    plt.close()
    plt.xlabel("Pixel value")
    plt.ylabel("Frame number")
    plt.hist2d(correct_info, frame_info,
               bins=(18, min(composite_size, num_correct_frames)),
               range=((0.1, 1), (0, min(composite_size, num_correct_frames))))
    plt.colorbar()
    plt.savefig("model_performance/{}_value_dist.png".format(simulation), dpi=250)
    plt.close()
    # Getting an estimate on 'noise'
    noise_correct = np.zeros(composite_size)
    noise_prediction = np.zeros(composite_size)
    noise_frame = np.zeros(composite_size)
    for frame in range(composite_size):
        noise_correct[frame] = len((correct[frame])[correct[frame] > 0.1])
        noise_prediction[frame] = len((prediction[frame])[prediction[frame] > 0.1])
        noise_frame[frame] = frame
    plt.figure(figsize=(7, 4))
    plt.xlabel("Frame")
    plt.ylabel("Number of non-zero points")
    plt.scatter(noise_frame, noise_prediction, label="Prediction")
    plt.scatter(noise_frame, noise_correct, label="Simulation")
    plt.legend()
    plt.savefig("model_performance/{}_{}_non-zero.png".format(network_name, simulation), dpi=250)
    plt.close()

    # Performing y-position stuff
    prediction_x_com = []
    prediction_y_com = []
    correct_x_com = []
    correct_y_com = []
    for frame in range(composite_size):
        x, y = calculate_com(prediction[frame])
        if frame < test_range:
            prediction_x_com.append(x)
            prediction_y_com.append(y)
        x, y = calculate_com(correct[frame])
        if frame < num_correct_frames:
            correct_x_com.append(x)
            correct_y_com.append(y)
    plt.figure(figsize=(7, 4))
    plt.xlabel("Frame step")
    plt.ylabel("Position of bubble centre")
    plt.plot(prediction_y_com, label="Prediction")
    plt.plot(correct_y_com, label="Simulation")
    plt.legend()
    plt.savefig("model_performance/{}_{}_y_position.png".format(network_name, simulation), dpi=250)
    plt.close()

    # Converting to a phase space
    # correct_frequency = []
    # predicted_frequency = []
    # for i in range(min(len(correct_y_com), len(prediction_y_com))-50):
    #     prediction_phase = np.asarray(prediction_y_com)[i:i+50] - 0.5
    #     correct_phase = np.asarray(correct_y_com)[i:i + 50] - 0.5
    #
    #     results = fit_sin(np.linspace(0, len(correct_phase)-1, len(correct_phase)), correct_phase)
    #     # plt.scatter(np.linspace(0, len(correct_phase)-1, len(correct_phase)), correct_phase)
    #     # x = np.linspace(0, len(correct_phase)-1, len(correct_phase)*10)
    #     # plt.plot(x, results["fitfunc"](x), label="Simulated")
    #     correct_frequency.append(results["omega"])
    #     results = fit_sin(np.linspace(0, len(prediction_phase)-1, len(prediction_phase)), prediction_phase)
    #     # plt.scatter(np.linspace(0, len(prediction_phase)-1, len(prediction_phase)), prediction_phase)
    #     # x = np.linspace(0, len(prediction_phase)-1, len(prediction_phase)*10)
    #     # plt.plot(x, results["fitfunc"](x), label="Predicted")
    #     predicted_frequency.append(results["omega"])
    # plt.plot(correct_frequency, label="Simulated omega")
    # plt.plot(predicted_frequency, label="Predicted omega")
    # plt.legend()
    # plt.savefig("model_performance/{}_{}_angular frequency.png".format(network_name, simulation), dpi=250)
    # plt.show()
    # Finding how it did across all simulations
    # max_sim_number = 16
    # bubble_sequences = []
    # for sim in range(max_sim_number):
    #     # Setting up the network's input
    #     starting_frames = np.zeros((1, frames, size, size, 1))
    #     for frame in range(frames):
    #         starting_frames[0, frame, :, :, 0] = dat_to_training.process_bmp(
    #             "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
    #                 str(flipped), variant, resolution, sim, start_point + frame * timestep
    #             ), image_size=size
    #         )[:, :, 1]
    #
    #     # Running the predictions
    #     prediction_x_com = []
    #     prediction_y_com = []
    #     final_frames = np.zeros((test_range, size, size))
    #     current_frames = starting_frames
    #     for loop in range(test_range):
    #         next_frame = network(current_frames)
    #         for frame in range(frames - 1):
    #             current_frames[:, frame] = current_frames[:, frame + 1]
    #         current_frames[:, frames - 1] = next_frame
    #         final_frames[loop] = next_frame[0, :, :, 0]
    #         x, y = calculate_com(next_frame[0, :, :, 0])
    #         prediction_x_com.append(x)
    #         prediction_y_com.append(y)
    #     plt.plot(prediction_y_com, label="Bubble_{}".format(sim))
    # plt.legend()
    # plt.show()


def main():
    image_size = 64
    image_frames = 4
    timestep = 5
    future_runs = 20
    resolution = 0.001

    scenario = 0
    if scenario < 2:
        # training_data = dat_to_training.generate_data(image_frames, image_size, timestep, future_runs, [0], False,
        #                                               resolution, [12])
        training_data = testing_weather.main()
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            decay_rate=0.9
        )
        if scenario == 0:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)

            network = create_network.create_u_network(layers.LeakyReLU(), image_frames, image_size, encode_size=10,
                                                      kernel_size=5, channels=1)
            discriminator = create_network.create_discriminator(2, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer, 50,
                          "u-net",
                          future_runs, image_frames)
            network.save("models/u_network_weather")
        elif scenario == 1:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            network = create_network.create_basic_network(layers.LeakyReLU(), image_frames, image_size, channels=1)
            discriminator = create_network.create_discriminator(2, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer, 200,
                          "basic",
                          future_runs, image_frames)
            network.save("models/basic_network")

    for sim in range(12, 13):
        # evaluate_performance("u_network", image_frames, image_size, timestep, resolution, simulation=sim)
        evaluate_weather("u_network_weather", image_frames, image_size)


@tf.function
def train_step(input_images, expected_output, network, discriminator, net_op, disc_op, future_runs, frames):
    with tf.GradientTape() as net_tape, tf.GradientTape() as disc_tape:
        current_output = []
        future_input = input_images
        predictions = network(input_images, training=True)
        current_output.append(predictions)

        for future_step in range(future_runs):
            next_input = []
            for i in range(frames - 1):
                next_input.append(future_input[:, i + 1])
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
    size = 64
    data = np.load("Meterology_data/data8.npz")
    data_useful = np.zeros((np.shape(data["data"])[1], size, size, 1))
    data_useful[:, :, :, 0] = data["data"][0, :, 69:69 + size, 420:420 + size]
    data_useful = np.tanh(data_useful / 500)
    images = np.zeros((1, frames, size, size, 1))
    images[0] = data_useful[:frames]
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
        # generate_images(network, epoch + 1, ref_index, name)
        generate_weather(network, epoch, name, images)
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
    # generate_images(network, epochs, ref_index, name)
    generate_weather(network, epochs, name, images)
    plt.close("all")
    plt.grid()
    plt.plot(overall_loss_gen, label="Generator loss")
    plt.plot(overall_loss_disc, label="Discriminator loss")
    plt.legend()
    plt.savefig("{}_losses.png".format(name), dpi=500)
    plt.clf()


if __name__ == "__main__":
    main()
