import datetime

print('Importing basics...')
import glob
import time
import imageio
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from tqdm import tqdm
from sunpy.net import attrs as a
print("Importing Tensorflow...")
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("{} Physical GPUs, {} Logical GPUs".format(len(gpus), len(logical_gpus)))
    except RuntimeError as e:
        print(e)
print("Keras...")
from tensorflow.keras import layers, optimizers, losses, models, backend as k, initializers
from tensorflow.keras import mixed_precision
print("Python files...")
import SOHO_data
import bubble_prediction
import create_network
import loss_functions
import dat_to_training
import testing_weather
import matplotlib.cm as cm


def generate_weather(model, epoch, name, input_frames):
    """Produces an image for how well the network predicts cloud data during training.
    @param model: the network to be tested.
    @param epoch: the current epoch number of the network.
    @param name: the name of the network.
    @param input_frames: the number of frames to be input to the network.
    """
    initial_frames = input_frames
    prediction = model(initial_frames)
    plt.imshow(prediction[0], cmap='Blues')
    plt.axis('off')
    plt.savefig("{}_predictions_at_epoch_{:04d}.png".format(name, epoch))
    plt.close('all')


def generate_images(model, epoch, input_images_index, name, input_frames, input_size):
    images = np.zeros((16, input_frames, input_size, input_size, 1))
    for i in range(len(input_images_index)):
        for j in range(0, input_frames):
            images[i, j, :, :, 0] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_False_0_0.001_12/data_{}.npy".format(
                    input_images_index[i] + j * 5)
                , input_size)[:, :, 1]
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
    x = np.linspace(-1, 1, image_size)
    y = np.linspace(1, -1, image_size)
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
    data_useful = np.zeros((np.shape(data["data"])[1], 1*size + 0, 1*size + 0, 1))

    data_useful[:, :, :, 0] = data["data"][0, :, 320:320 + size, 320:320 + size]
    data_useful = np.tanh(data_useful / 1500)
    initial_frames = np.zeros((1, frames, size, size, 1))
    initial_frames[0] = data_useful[:frames]
    rest_of_data = data_useful[frames:]
    given_frames = initial_frames
    final_data = np.zeros((np.shape(rest_of_data)[0], size, size, 3))
    print(np.shape(rest_of_data))
    final_data[:, :, :, 0] = rest_of_data[:, :, :, 0]
    for clouds in range(np.shape(rest_of_data)[0]):
        next_frame = network(given_frames)
        for frame in range(frames - 1):
            given_frames[0, frame] = given_frames[0, frame + 1]
        given_frames[0, frames - 1] = next_frame
        final_data[clouds, :, :, 2] = next_frame[0, :, :, 0]

    image_converts = final_data * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    imageio.mimsave("model_performance/{}_weather_composite.gif".format(network_name), images)
    plt.imshow(final_data[-1])
    plt.savefig("model_performance/{}_weather_composite_final.png".format(network_name), dpi=200)
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
    fig_matrix = 50
    y_pos = []
    y_vel = []
    offset = np.linspace(-0.1, 0.1, fig_matrix ** 2)
    plt.figure(figsize=(10, 7))
    colours = cm.jet(np.linspace(0, 1, len(offset)))
    final_bubbles = np.zeros((len(offset), size, size))
    print("New")
    for i, off in enumerate(offset):
        starting_frames = np.zeros((1, frames, size, size, 1))
        for frame in range(frames):
            starting_frames[0, frame, :, :, 0] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
                    str(flipped), variant, resolution, simulation, start_point + frame * timestep
                ), image_size=size, adjustment=off
            )[:, :, 1]
        x_test, y_test = calculate_com(starting_frames[0, 0, :, :, 0])
        # final_frames = np.zeros((test_range, size, size))
        final_frames = []
        current_frames = starting_frames
        for loop in range(test_range):
            next_frame = network(current_frames)
            for frame in range(frames - 1):
                current_frames[:, frame] = current_frames[:, frame + 1]
            current_frames[:, frames - 1] = next_frame
            if np.sum(next_frame[0, :, :, 0]) < 100:
                break
            final_frames.append(next_frame[0, :, :, 0])
            # final_frames[loop] = next_frame[0, :, :, 0]
        if len(final_frames) > 5:
            final_frames = np.asarray(final_frames)
            final_bubbles[i] = final_frames[-1]
            prediction_x_com = []
            prediction_y_com = []
            for frame in range(len(final_frames)):
                x, y = calculate_com(final_frames[frame])
                prediction_x_com.append(x)
                prediction_y_com.append(y)
            plt.plot(prediction_y_com, color=colours[i])
            y_pos.append(prediction_y_com)
            y_vel.append(np.gradient(np.asarray(prediction_y_com), 5))
            print("x = {:.04f}, y = {:.04f}, offset = {:.04f}, final_sum = {:.02f}, index = {}, length={}".format(
                x_test, y_test, off, np.sum(final_frames[-1]), i, len(final_frames)))
    x = np.linspace(0, test_range, 500)
    plt.plot(x, np.zeros(500) + 0.31415, ls=':', color="black", label="Fixed points")
    plt.plot(x, np.zeros(500), ls=':', color="black")
    plt.plot(x, np.zeros(500) - 0.31415, ls=':', color="black")
    plt.xlabel("Steps in prediction")
    plt.ylabel("Average y position")
    plt.legend()
    plt.grid()
    plt.xlim([0, test_range])
    # plt.xlim([-0.33, 0.33])
    plt.savefig("model_performance/{}_{}_offsets.png".format(network_name, simulation), dpi=200)
    # plt.show()
    plt.close()
    big_image = np.zeros((size * fig_matrix, size * fig_matrix))
    for i in range(0, fig_matrix):
        for j in range(0, fig_matrix):
            big_image[i*size:(i+1)*size, j*size:(j+1)*size] = final_bubbles[i * fig_matrix + j]
            plt.plot([0, size*fig_matrix], [j*size, j*size], color="white")
        plt.plot([i*size, i*size], [0, size*fig_matrix], color="white")
    plt.imshow(big_image)
    plt.axis("off")
    plt.savefig("model_performance/{}_{}_offsets_endings.png".format(network_name, simulation), dpi=500)
    plt.clf()
    plt.figure(figsize=(30, 21))
    for i, y in enumerate(y_pos):
        plt.plot(y, y_vel[i], color=colours[i])
    plt.grid()
    plt.savefig("model_performance/{}_{}_offsets_phase.png".format(network_name, simulation), dpi=200)
    plt.clf()
    # exit()
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
        final_frames[loop] = np.arctanh(next_frame[0, :, :, 0]) * 20
        # final_frames[loop] = next_frame[0, :, :, 0]

    # Getting the correct data
    max_data = len(glob.glob("Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/*".format(
        str(flipped), variant, resolution, simulation
    )))

    correct_frames = []
    for i in range(start_point + frames * timestep, max_data, timestep):
        correct_frames.append(np.arctanh(
            dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
                    str(flipped), variant, resolution, simulation, i
                ), image_size=size
        )[:, :, 1]) * 20)
        # correct_frames.append(
        #     dat_to_training.process_bmp(
        #         "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
        #             str(flipped), variant, resolution, simulation, i
        #         ), image_size=size
        # )[:, :, 1])
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

    image_converts = np.tanh(composite_frames/20) * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    # imageio.mimsave("model_performance/{}_{}_composite.gif".format(network_name, simulation), images)

    image_converts = np.tanh(composite_frames[:, :, :, 2]/20) * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    # imageio.mimsave("model_performance/{}_{}_solo.gif".format(network_name, simulation), images)

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
               range=((0.1, np.max(prediction_info)), (0, min(composite_size, test_range))))
    plt.colorbar()
    plt.savefig("model_performance/{}_{}_value_dist.png".format(network_name, simulation), dpi=250)
    plt.close()
    plt.xlabel("Pixel value")
    plt.ylabel("Frame number")
    plt.hist2d(correct_info, frame_info,
               bins=(18, min(composite_size, num_correct_frames)),
               range=((0.1, np.max(correct_info)), (0, min(composite_size, num_correct_frames))))
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

    # Velocity calculations
    prediction_y_com = np.asarray(prediction_y_com)
    correct_y_com = np.asarray(correct_y_com)

    prediction_y_velocity = np.gradient(prediction_y_com, 5)
    correct_y_velocity = np.gradient(correct_y_com, 5)
    colours_c = np.linspace(0, 1, len(correct_y_velocity))
    colours_p = np.linspace(0, 1, len(prediction_y_velocity))

    plt.figure(figsize=(14, 8))
    plt.xlim([0.4, 0.6])
    plt.ylim([-0.002, 0.002])
    # plt.plot(correct_y_com, correct_y_velocity, label="Simulation")
    plt.scatter(correct_y_com, correct_y_velocity, label="Simulation")
    # plt.plot(prediction_y_com, prediction_y_velocity, label="Prediction")
    plt.scatter(prediction_y_com, prediction_y_velocity, c=colours_p, label="Prediction")
    plt.legend()
    plt.grid()
    plt.savefig("model_performance/{}_{}_phase_space.png".format(network_name, simulation), dpi=250)
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


def compare_sun(network_name, frames, size):
    input_frames = np.zeros((1, frames, size, size, 1))
    sun_images = np.sort(glob.glob("sun_data/*"))
    # times = []
    # for file_ref in sun_images:
    #     time_pos = file_ref.find(".")
    #     times.append(float(file_ref[time_pos+1:])/100)
    # valid_found = False
    # start_time = times[0]
    # track_index = [0]
    # print(len(sun_images))
    # start = 1
    # for index, sun_time in enumerate(times[start:]):
    #     if not valid_found:
    #         current_time = sun_time
    #         if len(track_index) == 0:
    #             start_time = current_time
    #             track_index.append(index - start)
    #         elif len(track_index) == frames:
    #             valid_found = True
    #         else:
    #             if 10 < current_time - start_time < 14:
    #                 track_index.append(index - start)
    #                 start_time = current_time
    #             else:
    #                 track_index = []
    #
    # print(track_index)
    #
    # for ii, i in enumerate(track_index):
    #     print(sun_images[i])
    #     input_frames[0, ii, :, :, 0] = SOHO_data.get_data(sun_images[i], size)
    actual_input = [3, 4, 5, 6]
    for i, index in enumerate(actual_input):
        input_frames[0, i, :, :, 0] = SOHO_data.get_data(sun_images[index], size)
    network = models.load_model("models/{}".format(network_name))
    next_frames = np.zeros((196, size, size, 1))
    for i in range(196):
        next_frames[i, :, :, :] = network(input_frames)
        for j in range(frames-1):
            input_frames[0, j, :, :, :] = input_frames[0, j+1, :, :, :]
        input_frames[0, frames-1, :, :, :] = next_frames[i, :, :, :]
        # print(input_frames[0, :, 22, 22, :])
    image_converts = next_frames * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    imageio.mimsave("SOHO_predictions_basic.gif", images)
    final_image = np.zeros((size, size, 3))
    final_image[:, :, 1] = image_converts[-1, :, :, 0]
    final_image[:, :, 2] = SOHO_data.get_data(sun_images[55], size)
    plt.imshow(final_image)
    plt.savefig("Final_SOHO_prediction_basic.png", dpi=200)
    # images = []
    # for i in image_converts[:, :, 2]:
    #     images.append(i)
    # imageio.mimsave("SOHO_predictions_solo.gif", images)


def read_custom_data(frames, size, num_after_points, future_look, timestep, batch_size=8):
    # all_simulations = glob.glob("sams_training_data/*")
    # All the simulations that will be transformed into data
    looking_for = []
    for sim in range(0, 16):
        if sim != 12:
            looking_for.append("sams_training_data/new_xmin_Simulation_{}_points_100".format(sim))
    # Look through simulations and get metadata
    total_number_questions = 0
    for simulation in looking_for:
        # if simulation not in all_simulations:
        #     convert_dat_files([min(variants), max(variants)], resolution)
        # all_simulations = glob.glob("sams_training_data/*")

        steps = glob.glob(simulation+"/*")
        num_steps = len(steps) - 3
        maximum_question_start = num_steps - timestep*(frames + future_look)
        total_number_questions += maximum_question_start

    questions = np.zeros((total_number_questions, frames, 100, 2))
    answers = np.zeros((total_number_questions, num_after_points + 1, 100, 2))
    tracker = 0
    print("Loading in data...")
    progress = tqdm(total=total_number_questions)
    for simulation in looking_for:
        steps = glob.glob(simulation + "/*")

        num_steps = len(steps)
        maximum_question_start = num_steps - timestep*(frames + future_look)
        for step in range(3, maximum_question_start):
            progress.update(1)
            for frame in range(frames):
                questions[tracker, frame, :, :] = np.transpose(np.load(
                    simulation+"/data_{}.npy".format(step + frame * timestep)
                ), (1, 0))
            for future_frame in range(num_after_points + 1):
                answers[tracker, future_frame, :, :] = np.transpose(np.load(
                    simulation + "/data_{}.npy".format(
                        step + (frames + future_frame*(future_look//num_after_points)) * timestep)
                ), (1, 0))
            tracker += 1
    progress.close()

    print(np.max(questions))
    testing_data = tf.data.Dataset.from_tensor_slices((questions, answers))
    testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return testing_data


def step_decay(epoch):
    initial_rate = 1e-3
    learn_rate = max(initial_rate * 0.1 ** (epoch//10), 1e-6)
    return learn_rate


class LearningRateStep(optimizers.schedules.LearningRateSchedule):
    def __init__(self, i_lr):
        self.initial_learning_rate = i_lr

    def __call__(self, step):
        epoch = step//1296
        return self.initial_learning_rate * 0.1 ** (epoch//20)


def main():
    # policy = mixed_precision.Policy('mixed_float16')
    # policy = mixed_precision.Policy('float32')
    # mixed_precision.set_global_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)
    image_size = 128
    image_frames = 1
    timestep = 5
    future_runs = 10
    num_after_points = 1
    resolution = 0.001
    # read_custom_data(image_frames, image_size, num_after_points, future_runs, timestep)
    # exit()
    scenario = 1
    # tf.compat.v1.disable_eager_execution()
    print(tf.executing_eagerly())
    if scenario < 10:
        # Weather
        # dataset = np.load("Meterology_data/data8.npz")
        # data = dataset["data"][:2, :50, :, :]
        # del dataset
        # data = testing_weather.simplify_data(data, image_size, window_downscaling=1)
        # training_data = testing_weather.extract_chain_info(data, image_frames, future_runs)

        # Bubble
        training_data = dat_to_training.generate_data(image_frames, image_size, timestep, future_runs, [0], False,
                                                      resolution, [15], num_after_points)

        # Sun
        # # obs, size, ref = SOHO_data.get_metadata([2001], [7], a.Instrument.eit, 195)
        # solar_files = glob.glob("sun_data/*")
        # solar_files = np.asarray(solar_files)
        # solar_files = np.sort(solar_files)
        # obs = []
        # for file in solar_files:
        #     date = file[12:]
        #     year = int(date[:4])
        #     month = int(date[4:6])
        #     day = int(date[6:8])
        #     hour = int(date[9:11])
        #     minute = int(date[11:13])
        #     second = int(date[13:15])
        #     file_date = datetime.datetime(year, month, day, hour, minute, second)
        #     og_date = datetime.datetime(1970, 1, 1, 0, 0, 0)
        #     obs.append((file_date-og_date).total_seconds())
        #     if year == 2001 and month == 6 and day == 1 and hour == 4:
        #         print("{}:{}".format(minute, second))
        #         print((file_date-og_date).total_seconds())
        #
        # mask, time_chains = SOHO_data.get_valid_data(image_frames, future_runs, obs)
        # # print("Total data use {:.2f}Gb ({} files), {} training data items".format(
        # #     np.sum(size[mask[:] == 1]) / 1024, np.sum(mask), len(time_chains)))
        # # SOHO_data.download_data(ref, mask)
        # print(time_chains)
        # training_data = SOHO_data.generate_training_data(time_chains, image_frames, image_size)
        # lr_schedule = optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-3,
        #     decay_steps=100000,
        #     decay_rate=0.5
        # )
        lr_schedule = LearningRateStep(1e-3)
        if scenario == -1:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)

            initializer_w = initializers.VarianceScaling(scale=2.9)
            initializer_b = initializers.RandomNormal(stddev=0.04)
            network = create_network.dense_network(
                100, image_frames, 2, layers.LeakyReLU(), None, None, 200, 2, 12, initializer_w, initializer_b, None)
            discriminator = create_network.create_sam_discriminator(num_after_points + 1)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer,
                          5,
                          "sam-net",
                          future_runs, image_frames, num_after_points, image_size)
            network.save("models/sam_network")

        if scenario == 0:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)

            network = create_network.create_u_network(layers.LeakyReLU(), image_frames, image_size, encode_size=10,
                                                      kernel_size=5, channels=1, first_channels=16)
            discriminator = create_network.create_discriminator(num_after_points + 1, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer,
                          100,
                          "u-net-plus_bubble",
                          future_runs, image_frames, num_after_points, image_size)
            network.save("models/u_network_GAN_bubble_BIG")
        if scenario == 9:
            # network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            # discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            network_optimizer = optimizers.Adam(epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(epsilon=0.1)

            network = create_network.create_dumb_network(image_frames, image_size, channels=1)
            discriminator = create_network.create_discriminator(num_after_points + 1, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer,
                          1000,
                          "dumb",
                          future_runs, image_frames, num_after_points, image_size)
            network.save("models/dumb_GAN_bubble")
        elif scenario == 1:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            network = create_network.create_basic_network(layers.LeakyReLU(), image_frames, image_size, channels=1,
                                                          start_channels=16, latent_dimensions=100)
            discriminator = create_network.create_discriminator(num_after_points + 1, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer,
                          50,
                          "basic_BIG",
                          future_runs, image_frames, num_after_points, image_size)
            network.save("models/basic_network_BIG")
        elif scenario == 2:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            network = create_network.create_parallel_network(layers.LeakyReLU(), image_frames, image_size, channels=1)
            discriminator = create_network.create_discriminator(num_after_points + 1, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer,
                          50,
                          "parallel",
                          future_runs, image_frames, num_after_points, image_size)
            network.save("models/parallel_network")
        elif scenario == 3:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            network = create_network.create_resnet(layers.LeakyReLU(), image_frames, image_size, channels=1)
            discriminator = create_network.create_discriminator(num_after_points + 1, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer,
                          50,
                          "resnet",
                          future_runs, image_frames, num_after_points, image_size)
            network.save("models/resnet")
        elif scenario == 4:
            network_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            discriminator_optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
            network = create_network.create_transformer_network(
                layers.LeakyReLU(), image_frames, image_size, channels=1, layering=2
            )
            discriminator = create_network.create_discriminator(num_after_points + 1, image_size)
            print(network.summary())
            train_network(training_data, network, discriminator, network_optimizer, discriminator_optimizer,
                          50,
                          "transformer",
                          future_runs, image_frames, num_after_points, image_size)
            network.save("models/transformer_network")

    for sim in range(0, 16):
        # compare_sun("basic_network_sun", image_frames, image_size)
        evaluate_performance("basic_network_BIG", image_frames, image_size, timestep, resolution,
                             simulation=sim, test_range=400)
        # evaluate_weather("basic_network_weather", image_frames, image_size)


@tf.function
def train_step(input_images, expected_output, network, discriminator, net_op, disc_op, future_runs, frames, num_points):
    with tf.GradientTape() as net_tape, tf.GradientTape() as disc_tape:
        current_output = []
        future_input = input_images
        predictions = network(input_images, training=True)

        for future_step in range((future_runs//num_points) * num_points):
            if future_step % (future_runs//num_points) == 0:
                # current_output.append(tf.cast(predictions, tf.float64) + future_input[:, -1]) # SAMS NETWORK ONLY
                current_output.append(predictions)
            next_input = []
            for i in range(frames - 1):
                next_input.append(future_input[:, i + 1])
            # next_input.append(tf.cast(predictions, tf.float64) + future_input[:, -1]) # SAMS NETWORK ONLY
            next_input.append(predictions)
            future_input = tf.stack(next_input, axis=1)
            predictions = network(future_input, training=True)

        # current_output.append(tf.cast(predictions, tf.float64) + future_input[:, -1]) # SAMS NETWORK ONLY
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

    return network_loss, disc_loss, network_mse


def train_network(dataset, network, discriminator, net_op, disc_op, epochs, name, future_runs, frames, num_points, size):
    # size = 64
    # data = np.load("Meterology_data/data8.npz")
    # data_useful = np.zeros((np.shape(data["data"])[1], size, size, 1))
    # data_useful[:, :, :, 0] = data["data"][0, :, 69:69 + size, 420:420 + size]
    # data_useful = np.tanh(data_useful / 500)
    # images = np.zeros((1, frames, size, size, 1))
    # images[0] = data_useful[:frames]
    times_so_far = []
    ref_index = [4, 42, 69, 72, 104, 254, 298, 339, 347, 420, 481, 482, 555, 663, 681, 701]
    ref_index_float = np.linspace(3, 700, 16, endpoint=True)
    for i, r in enumerate(ref_index_float):
        ref_index[i] = int(r)
    overall_loss_gen = []
    overall_loss_disc = []
    overall_loss_mse = []
    for epoch in range(epochs):
        start = time.time()
        gen_losses = []
        disc_losses = []
        mse_losses = []
        print("Running epoch {}...".format(epoch + 1))
        for questions, answers in dataset:
            gen_loss, disc_loss, mse = train_step(questions, answers,
                                                  network, discriminator,
                                                  net_op, disc_op,
                                                  future_runs, frames, num_points)
            gen_losses.append(k.mean(gen_loss))
            disc_losses.append(k.mean(disc_loss))
            mse_losses.append(k.mean(mse))
        generate_images(network, epoch + 1, ref_index, name, frames, size)
        # generate_weather(network, epoch, name, images)
        times_so_far.append(time.time() - start)
        seconds_per_epoch = times_so_far[epoch]
        if seconds_per_epoch / 60 > 1:
            print("Time for epoch {} was {:.0f}min, {:.0f}s".format(
                epoch + 1, seconds_per_epoch//60, seconds_per_epoch - (seconds_per_epoch//60)*60)
            )
        else:
            print("Time for epoch {} was {:.0f}s".format(epoch + 1, seconds_per_epoch))

        print("Losses were {:.4f} for generator and {:.4f} for discriminator, with MSE {:.4f}".format(
            np.mean(gen_losses), np.mean(disc_losses), np.mean(mse_losses))
        )
        if np.mean(gen_losses) > np.mean(disc_losses):
            print("Discriminator winning!")
        else:
            print("Generator winning!")

        time_left = np.mean(times_so_far) * (epochs - epoch - 1) / 60
        print("ETA currently stands at {:.0f} hours, {:.0f}min from now".format(
            time_left//60, time_left - (time_left//60)*60)
        )
        overall_loss_disc.append(np.mean(disc_losses))
        overall_loss_gen.append(np.mean(gen_losses))
        overall_loss_mse.append(np.mean(mse_losses))
    generate_images(network, epochs, ref_index, name, frames, size)
    # generate_weather(network, epochs, name, images)
    plt.close("all")
    plt.grid()
    plt.plot(overall_loss_gen, label="Generator loss")
    plt.plot(overall_loss_disc, label="Discriminator loss")
    plt.plot(overall_loss_mse, label="Mean squared error")
    plt.yscale('log')
    plt.legend()
    plt.savefig("{}_losses.png".format(name), dpi=500)
    plt.clf()


if __name__ == "__main__":
    main()
