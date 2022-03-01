import dat_to_training
import create_network
import loss_functions
from tensorflow.keras import layers, losses, models, optimizers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from time import time
from numpy.random import default_rng
import imageio
import model_analysis


def plot_performance(model, image_frames, image_size, timestep, name):
    preamble = "Simulation_images/Simulation_8/img_"
    start = 20
    initial = np.zeros((1, image_frames, image_size, image_size, 3))
    expected = np.zeros((image_size, image_size, 1))
    for frame in range(0, image_frames):
        frame_to_load = "{}{}{}".format(preamble, start + frame * timestep, ".bmp")
        initial[0, frame, :, :, :] = np.asarray(Image.open(frame_to_load)) / 255
    expected_frame = "{}{}{}".format(preamble, start + image_frames * timestep, ".bmp")
    expected[:, :, 0] = np.asarray(Image.open(expected_frame))[:, :, 1] / 255
    guess = model(initial)[0]
    print(np.shape(expected))
    print(np.shape(guess))
    difference = np.abs(np.around(guess) - expected)

    axes = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        ABCD
        EEEF
        EEEG
        EEEH
        IIJJ
        IIJJ
        """
    )
    axes["B"].imshow(expected)
    axes["C"].imshow(guess)

    axes["D"].imshow(guess - expected)
    overlap = np.zeros((image_size, image_size, 3))
    overlap[:, :, 0] = guess[:, :, 0]
    overlap[:, :, 1] = expected[:, :, 0]
    axes["E"].imshow(overlap)

    previous_frame = np.zeros((image_size, image_size, 1))
    previous_frame[:, :, 0] = initial[0, image_frames - 1, :, :, 1]
    axes["A"].imshow(previous_frame)
    rounded_guess = np.around(guess)

    axes["F"].imshow(rounded_guess - expected)
    axes["G"].imshow(expected - previous_frame)
    axes["H"].imshow(rounded_guess - previous_frame)

    print(np.sum(np.abs((rounded_guess - expected))))

    positive_real = np.zeros((image_size, image_size, 1))
    negative_real = np.zeros((image_size, image_size, 1))
    real_difference = expected - previous_frame
    positive_real[real_difference > 0] = 1
    negative_real[real_difference < 0] = 1

    positive_guess = np.zeros((image_size, image_size, 1))
    negative_guess = np.zeros((image_size, image_size, 1))
    guess_difference = rounded_guess - previous_frame
    positive_guess[guess_difference > 0] = 1
    negative_guess[guess_difference < 0] = 1

    positive_correct = np.zeros((image_size, image_size, 1))
    positive_correct[(positive_guess == positive_real) & (positive_real == 0)] = 0
    positive_correct[(positive_guess == positive_real) & (positive_real == 1)] = 2
    positive_correct[(positive_guess != positive_real) & (positive_real == 0)] = 1
    positive_correct[(positive_guess != positive_real) & (positive_real == 1)] = 3

    negative_correct = np.zeros((image_size, image_size, 1))
    negative_correct[(negative_guess == negative_real) & (negative_guess == 0)] = 0
    negative_correct[(negative_guess == negative_real) & (negative_guess == 1)] = 2
    negative_correct[(negative_guess != negative_real) & (negative_guess == 0)] = 1
    negative_correct[(negative_guess != negative_real) & (negative_guess == 1)] = 3

    positive_rgb = np.zeros((image_size, image_size, 3), dtype=int)
    positive_rgb[positive_correct[:, :, 0] == 0, :] = 0
    for i in range(0, 3):
        positive_rgb[positive_correct[:, :, 0] == i + 1, i] = 255

    axes["I"].imshow(positive_rgb)

    negative_rgb = np.zeros((image_size, image_size, 3), dtype=int)
    negative_rgb[negative_correct[:, :, 0] == 0, :] = 0
    for i in range(0, 3):
        negative_rgb[negative_correct[:, :, 0] == i + 1, i] = 255

    axes["J"].imshow(negative_rgb)

    combined_rgb = negative_rgb + positive_rgb
    combined_rgb[(combined_rgb[:, :, 0] == 0) & (combined_rgb[:, :, 1] == 0) & (combined_rgb[:, :, 2] == 0), :] = 255
    # for a in axes:
    #     axes[a].set_xticklabels([])
    #     axes[a].set_yticklabels([])
    #     axes[a].tick_params(bottom=False, left=False)

    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("{}.png".format(name), dpi=500, bbox_inches='tight')

    return np.sum(difference)


def explore_learning_space(activation_function, optimizer, loss_function):
    image_frames = []
    image_sizes = []
    time_to_learn = []
    dropouts = []
    timesteps = []
    losses = []
    diffs = []
    image_frame_range = [1, 10, 1]
    image_size_range = [16, 64, 16]
    timestep_range = [1, 20, 1]
    dropout_range = np.linspace(0, 1, 20)
    for dropout in dropout_range:
        for frames in range(image_frame_range[0], image_frame_range[1], image_frame_range[2]):
            for size in range(image_size_range[0], image_size_range[1], image_size_range[2]):
                for timestep in range(timestep_range[0], timestep_range[1], timestep_range[2]):
                    dat_to_training.convert_dat_files([0, 0], image_size=size)
                    parameters = explore_parameter_space(
                        frames, size, dropout, activation_function, optimizer, loss_function
                    )
                    params = parameters["Trainable parameters"].tolist()
                    location = params.index(min(params))
                    encode_size = parameters["encode_size"].tolist()[location]
                    pool = parameters["allow_pooling"].tolist()[location]
                    upsample = parameters["allow_upsampling"].tolist()[location]
                    transpose = parameters["max_transpose_layers"].tolist()[location]
                    kernel = parameters["kernel_size"].tolist()[location]

                    model = create_network.create_neural_network(
                        activation_function, optimizer, loss_function, frames,
                        image_size=size, encode_size=encode_size, allow_pooling=pool,
                        allow_upsampling=upsample, max_transpose_layers=transpose, kernel_size=kernel,
                        dropout_rate=dropout
                    )
                    train_time_start = time()
                    training_data = dat_to_training.create_training_data(frames, timestep, image_size=size)
                    model, history = create_network.train_model(model, training_data, epochs=5)
                    train_time_end = time()
                    diff = plot_performance(model, frames, size, timestep, "Model_tests/S{}-T{}-F{}-D{:.2f}".format(
                        size, timestep, frames, dropout
                    ).replace(".", "_"))
                    time_to_learn.append(train_time_end - train_time_start)
                    losses.append(history["loss"])
                    image_sizes.append(size)
                    timesteps.append(timestep)
                    image_frames.append(frames)
                    dropouts.append(dropout)
                    diffs.append(diff)
    learning_space = pd.DataFrame({
        "Time to learn": time_to_learn,
        "loss": losses,
        "Pixel differences": diffs,
        "image_size": image_sizes,
        "timestep": timesteps,
        "image_frames": image_frames,
        "dropout_rate": dropouts
    })
    learning_space.to_csv("Total_evaluation_of_all_spaces.csv")
    return learning_space


def explore_parameter_space(image_frames, image_size, dropout_rate, activation_function, optimizer, loss_function):
    encode_sizes = []
    allowing_pooling = []
    allowing_upsampling = []
    max_transpose_layering = []
    kernel_sizes = []
    encode_range = [1, 20, 1]
    pooling_range = [True, False]
    upsample_range = [True, False]
    transpose_range = [1, 10, 1]
    kernel_range = [2, 10, 1]
    parameters = []

    for encode in range(encode_range[0], encode_range[1], encode_range[2]):
        for pool in pooling_range:
            for upsample in upsample_range:
                for transpose in range(transpose_range[0], transpose_range[1], transpose_range[2]):
                    for kernel in range(kernel_range[0], kernel_range[1], kernel_range[2]):
                        try:
                            model = create_network.create_neural_network(
                                activation_function, optimizer, loss_function, image_frames,
                                image_size=image_size, encode_size=encode, allow_pooling=pool,
                                allow_upsampling=upsample, max_transpose_layers=transpose, kernel_size=kernel,
                                dropout_rate=dropout_rate
                            )
                            parameters_line = create_network.interpret_model_summary(model)
                            number = float(parameters_line.split(":")[1].replace(",", ""))
                            parameters.append(number)
                            encode_sizes.append(encode)
                            allowing_pooling.append(pool)
                            allowing_upsampling.append(upsample)
                            max_transpose_layering.append(transpose)
                            kernel_sizes.append(kernel)
                        except Exception as e:
                            print(e)
    parameter_data = pd.DataFrame({
        "encode_size": encode_sizes,
        "allow_pooling": allowing_pooling,
        "allow_upsampling": allowing_upsampling,
        "max_transpose_layers": max_transpose_layering,
        "kernel_size": kernel_sizes,
        "Trainable parameters": parameters
    })
    parameter_data.to_csv("Parameter_tests/Trainable_parameters_for_S{}-T{}-F{}-D{:.2f}".format(
        image_size, "any", image_frames, dropout_rate
    ).replace(".", "_") + ".csv", index=False)
    return parameter_data


def generate_sample(image, image_size, rng):
    sample_image = np.zeros((image_size, image_size, 3))
    random_image = rng.random((image_size, image_size))
    sample_image[:, :, 1] = np.greater_equal(image[:, :, 1], random_image)
    sample_image = dat_to_training.generate_rail(sample_image)
    return sample_image


def long_term_prediction(
        model, start_sim, start_image, image_size, timestep, frames, number_to_simulate, resolution, round_result=False,
        extra=True, jump=True, dry_run=False):
    input_images = np.zeros((1, frames, image_size, image_size, 3))
    for frame in range(0, frames):
        try:
            input_images[0, frame, :, :, :] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
                    "False", 0, resolution, start_sim, start_image + frame * timestep
                ), image_size
            )
        except IOError as e:
            print("Error - either invalid simulation number or image out of range!")
            print(e)
            return []
    positions = []
    if dry_run:
        try:
            for i in range(frames, number_to_simulate):
                positions.append((dat_to_training.process_bmp(
                    "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
                        "False", 0, resolution, start_sim, start_image + i * timestep
                    ), image_size
                ) * 255).astype(np.uint8))
        except:
            return positions
        return positions
    if jump:
        output_image = np.zeros((image_size, image_size, 3))
        for i in range(0, number_to_simulate):
            output_image[:, :, 1] = model(input_images)[0, :, :, 0]
            dat_to_training.generate_rail(output_image)
            for frame in range(0, frames-1):
                input_images[0, frame, :, :, :] = input_images[0, frame+1, :, :, :]
            input_images[0, frames-1, :, :, :] = output_image
            positions.append((input_images[0, frames-1, :, :, :] * 255).astype(np.uint8))
        return positions
    future_frames = np.zeros((frames, frames, image_size, image_size, 3))
    for i in range(0, number_to_simulate):
        if extra:
            if i < frames:
                averaging_arrays = np.zeros((i + 1, image_size, image_size, 3))
            else:
                averaging_arrays = np.zeros((frames, image_size, image_size, 3))
            output_image = np.zeros((frames, image_size, image_size, 3))
            output_image[:, :, :, 1] = model(input_images)[0, :, :, :, 0]
            for frame in range(0, frames):
                dat_to_training.generate_rail(output_image[frame])
            future_frames[i % frames] = output_image
            for frame in range(0, min(i + 1, frames)):
                averaging_arrays[frame, :, :, :] = future_frames[frame, (i - frame) % frames, :, :, :]
            for frame in range(1, frames):
                input_images[0, frame - 1, :, :, :] = input_images[0, frame, :, :, :]
            if round_result:
                input_images[0, frames - 1, :, :, 1] = np.around(np.average(averaging_arrays, axis=0)[:, :, 1])
                input_images[0, frames - 1, :, :, 0] = np.average(averaging_arrays, axis=0)[:, :, 0]
                input_images[0, frames - 1, :, :, 2] = np.average(averaging_arrays, axis=0)[:, :, 2]
            else:
                input_images[0, frames - 1, :, :, :] = np.average(averaging_arrays, axis=0)
            positions.append((input_images[0, frames - 1, :, :, :] * 255).astype(np.uint8))
        else:
            output_image = np.zeros((frames, image_size, image_size, 3))
            output_image[:, :, :, 1] = model(input_images)[0, :, :, :, 0]
            for frame in range(0, frames):
                if round_result:
                    output_image[frame] = np.around(output_image[frame])
                dat_to_training.generate_rail(output_image[frame])
            for frame in range(1, frames):
                input_images[0, frame - 1, :, :, :] = input_images[0, frame, :, :, :]
            input_images[0, frames - 1, :, :, :] = output_image[0]
            positions.append((input_images[0, frames - 1, :, :, :] * 255).astype(np.uint8))
    return positions


def make_gif(image, name):
    images = []
    for i in image:
        images.append(i)
    imageio.mimsave("{}.gif".format(name), images)


def generate_random_value(rng, allowed_range, i=True):
    target = rng.random(1) * (allowed_range[1] - allowed_range[0]) + allowed_range[0]
    if i:
        return int(target)
    return target


def calculate_com(image, both=False):
    image_size = np.shape(image)[0]
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')
    x_com = np.sum(x*image[:, :, 1])/np.sum(image[:, :, 1])
    y_com = np.sum(y*image[:, :, 1])/np.sum(image[:, :, 1])
    if both:
        return -x_com+(float(image_size)/2), y_com-(float(image_size)/2)
    return np.sqrt((x_com-(float(image_size)/2))**2 + (y_com-(float(image_size)/2))**2)
    return  y_com


def main():
    # activation_function = "LeakyReLU"
    tf.random.set_seed(100)
    rng = default_rng(69420)
    activation_function = layers.LeakyReLU()
    optimizer = optimizers.Adam(learning_rate=0.001, epsilon=0.1)
    # optimizer = optimizers.Adam()
    # losses = [
    #     loss_functions.UBERLOSS,
    #     loss_functions.UBERLOSS_minus_dice,
    #     loss_functions.cubic_loss,
    #     loss_functions.relative_diff,
    #     loss_functions.absolute_diff,
    #     loss_functions.ssim_loss,
    #     loss_functions.mse_dice,
    #     loss_functions.bce_dice
    # ]
    kernel_sizes = [2, 3, 4, 5, 6, 7]
    encode_sizes = [1, 2, 3, 4, 5, 10]
    image_sizes = [30, 40, 50, 60, 90]
    frames = [1, 2, 4]
    parameters_extra = [
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Alpha", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Aberdeen", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 3, 0.001, [0], True, [0], 5, "Bravo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 3, 0.001, [0], True, [0], 5, "Bristol", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 3, 0.001, [0], True, [0], 5, "Charlie", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 3, 0.001, [0], True, [0], 5, "Castlebay", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 3, 0.001, [0], True, [0], 5, "Delta", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 3, 0.001, [0], True, [0], 5, "Derby", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 2, 0.001, [0], True, [0], 5, "Echo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 2, 0.001, [0], True, [0], 5, "Exeter", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 2, 0.001, [0], True, [0], 5, "Foxtrot", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 2, 0.001, [0], True, [0], 5, "Fleet", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 2, 0.001, [0], True, [0], 5, "Golf", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 2, 0.001, [0], True, [0], 5, "Glasgow", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 2, 0.001, [0], True, [0], 5, "Hotel", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 2, 0.001, [0], True, [0], 5, "Hartlepool", 20, False],
        ####
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 5, 0.001, [0], True, [0], 5, "India", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 5, 0.001, [0], True, [0], 5, "Inverness", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 5, 0.001, [0], True, [0], 5, "Juliet", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 5, 0.001, [0], True, [0], 5, "JohnOGroats", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 5, 0.001, [0], True, [0], 5, "Kilo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 5, 0.001, [0], True, [0], 5, "Kingston", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 5, 0.001, [0], True, [0], 5, "Lima", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 5, 0.001, [0], True, [0], 5, "London", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 4, 0.001, [0], True, [0], 5, "Mancy", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 5, 4, 0.001, [0], True, [0], 5, "Manchester", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 4, 0.001, [0], True, [0], 5, "November", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 5, 4, 0.001, [0], True, [0], 5, "Nottingham", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 4, 0.001, [0], True, [0], 5, "Oscar", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 5, True, True, 1, 4, 0.001, [0], True, [0], 5, "Oxford", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 4, 0.001, [0], True, [0], 5, "Papa", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 10, True, True, 1, 4, 0.001, [0], True, [0], 5, "Portsmouth", 20, False],
        #####
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Quebec", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 3, 0.001, [0], True, [0], 5, "Quedgeley", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 3, 0.001, [0], True, [0], 5, "Romeo", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 3, 0.001, [0], True, [0], 5, "Reading", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 3, 0.001, [0], True, [0], 5, "Sierra", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 3, 0.001, [0], True, [0], 5, "Salisbury", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 3, 0.001, [0], True, [0], 5, "Tango", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 3, 0.001, [0], True, [0], 5, "Taunton", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 2, 0.001, [0], True, [0], 5, "Uniform", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 2, 0.001, [0], True, [0], 5, "Uxbridge", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 2, 0.001, [0], True, [0], 5, "Victor", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 2, 0.001, [0], True, [0], 5, "Verwood", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 2, 0.001, [0], True, [0], 5, "Whiskey", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 2, 0.001, [0], True, [0], 5, "Woking", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 2, 0.001, [0], True, [0], 5, "Xray", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 2, 0.001, [0], True, [0], 5, "Xfuckonlyknows", 20, False],
        ####
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 5, 0.001, [0], True, [0], 5, "Yankee", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 5, 0.001, [0], True, [0], 5, "Yateley", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 5, 0.001, [0], True, [0], 5, "Zulu", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 5, 0.001, [0], True, [0], 5, "Zetland", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Amsterdam", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Andover", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Brussels", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 5, 0.001, [0], True, [0], 5, "0_Birmingham", 20, False],
        ###
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Copenhagen", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Cardiff", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Dublin", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 5, 4, 0.001, [0], True, [0], 5, "0_Dundee", 20, False],
        ##
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_ElAaiun", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 3, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Edinburgh", 20, False],
        #
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Freetown", 20, True],
        [loss_functions.UBERLOSS, 45, 3, 7, True, True, 1, 4, 0.001, [0], True, [0], 5, "0_Frimley", 20, False],
    ]

    parameters_extra = [
        [losses.binary_crossentropy, 45, 1, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Alpha", 5, True],
        [losses.binary_crossentropy, 45, 1, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Aberdeen", 20, False],
        [losses.binary_crossentropy, 45, 1, 10, True, True, 5, 3, 0.001, [0], True, [0], 5, "Bravo", 20, True],
        [losses.binary_crossentropy, 45, 1, 10, True, True, 5, 3, 0.001, [0], True, [0], 5, "Bristol", 20, False],
    ]
    for parameters in parameters_extra:
        loss_function = parameters[0]
        epochs = parameters[14]
        image_frames = parameters[2]
        image_size = parameters[1]
        timestep = parameters[12]
        encode_size = parameters[3]
        resolution = parameters[8]
        max_transpose_layers = parameters[6]
        kernel_size = parameters[7]
        dropout_rate = 0
        name = parameters[13]
        linearity = parameters[15]
        # loss_function = losses.mean_squared_logarithmic_error
        # loss_function = losses.cosine_similarity
        # loss_function = losses.log_cosh
        # loss_function = losses.huber
        # loss_function = loss_functions.bce_dice
        # loss_function = losses.categorical_crossentropy
        # loss_function = losses.BinaryCrossentropy()
        # loss_function = losses.sparse_categorical_crossentropy
        # loss_function = loss_functions.mse_dice
        # loss_function = loss_functions.tester_loss
        # loss_function = loss_functions.UBERLOSS
        # # loss_function = loss_functions.ssim_loss
        # epochs = 20
        # image_frames = 2
        # image_size = 60
        # timestep = 5
        # dropout_rate = 0.1
        # encode_size = 2
        # resolution = 0.001
        # # howdy
        # max_transpose_layers = 20
        # kernel_size = 4
        # dropout = 0
        try:
            # model = models.load_model(
            #     "models/Test_collection",
            #     custom_objects={
            #         "mean_squared_logarithmic_error": losses.mean_squared_logarithmic_error,
            #         "binary_crossentropy": losses.binary_crossentropy,
            #         "ssim_loss": loss_functions.ssim_loss,
            #         "UBERLOSS": loss_functions.UBERLOSS
            #     })
            # model = create_network.create_neural_network(
            #     activation_function, optimizer, loss_function, image_frames,
            #     image_size=image_size, encode_size=encode_size, allow_pooling=True,
            #     allow_upsampling=True, max_transpose_layers=max_transpose_layers, kernel_size=kernel_size,
            #     dropout_rate=dropout_rate
            # )
            model = create_network.create_inception_network(
                activation_function, optimizer, loss_function, image_frames,
                image_size=image_size, encode_size=encode_size, allow_pooling=True,
                allow_upsampling=True, max_transpose_layers=max_transpose_layers, kernel_size=kernel_size,
                dropout_rate=dropout_rate, inception=linearity, simple=False
            )
                # parameters_line = create_network.interpret_model_summary(model)
                # print(parameters_line)
                # print(model.summary())
            print(name)
            # training_data = dat_to_training.create_training_data(
            #     image_frames, timestep, image_size=image_size,
            #     excluded_sims=[12], variants=[0], resolution=resolution, flips_allowed=False)
            training_data = dat_to_training.create_training_data(
                image_frames, timestep, image_size=image_size,
                excluded_sims=[12], variants=[0], resolution=resolution, flips_allowed=False, easy_mode=False)
            print(model.summary())
            # exit()
            model, history = create_network.train_model(model, training_data, epochs=epochs)
            model.save("models/{}".format(name))
            model_analysis.cross_check(name, [12, 20])
        except Exception as e:
            print("Fail!")
            print(e)

        overall_loss = history.history["loss"]
        overall_val = history.history["val_loss"]
        bce = history.history["binary_crossentropy"]
        mse = history.history["mean_squared_logarithmic_error"]
        # ssim = history.history["ssim_loss"]
        plt.clf()
        plt.close()
        plt.plot(overall_loss, label="overall loss")
        plt.plot(overall_val, linestyle="dashed", label="overall validation loss")
        plt.plot(bce, label="binary cross entropy")
        plt.plot(mse, label="mean squared logarithmic error")
        # ssim = np.asarray(ssim)
        # ssim_adjusted = 1 / (1 + np.exp(-ssim))
        # plt.plot(ssim_adjusted, label="SSIM (adjusted)")
        plt.grid()
        plt.yscale("log")
        plt.xlabel("Epoch number")
        plt.ylabel("Values/AU")
        plt.legend()
        plt.savefig("model_performance/{}_Losses_across_epochs.png".format(name), dpi=500)

        del model
        del training_data

    output_images = np.zeros((1, image_frames, image_size, image_size, 1))
    input_images = np.zeros((1, image_frames, image_size, image_size, 3))
    expected_images = np.zeros((1, image_frames, image_size, image_size, 1))
    for frame in range(0, image_frames):
        try:
            input_images[0, frame, :, :, :] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
                    "True", 0, 0.001, 6, 20 + frame * timestep
                ), image_size)
            expected_images[0, frame, :, :, 0] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}.npy".format(
                    "True", 0, 0.001, 6, 20 + (frame + image_frames) * timestep
                ), image_size)[:, :, 1]
        except IOError as e:
            print("Error - either invalid simulation number or image out of range!")
            print(e)
    output_images = model(input_images)
    axes = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        A
        E
        I
        Z
        """
    )
    # axes = plt.figure(constrained_layout=True).subplot_mosaic(
    #     """
    #     ABCD
    #     EFGH
    #     IJKL
    #     ZXYQ
    #     """
    # )
    # Input
    axes["A"].imshow(input_images[0, 0, :, :, :])
    # axes["B"].imshow(input_images[0, 1, :, :, :])
    # axes["C"].imshow(input_images[0, 2, :, :, :])
    # axes["D"].imshow(input_images[0, 3, :, :, :])
    # Prediction
    axes["E"].imshow(output_images[0, 0, :, :, :])
    # axes["F"].imshow(output_images[0, 1, :, :, :])
    # axes["G"].imshow(output_images[0, 2, :, :, :])
    # axes["H"].imshow(output_images[0, 3, :, :, :])
    # Actual
    axes["I"].imshow(expected_images[0, 0, :, :, :])
    # axes["J"].imshow(expected_images[0, 1, :, :, :])
    # axes["K"].imshow(expected_images[0, 2, :, :, :])
    # axes["L"].imshow(expected_images[0, 3, :, :, :])
    # Difference
    axes["Z"].imshow(expected_images[0, 0, :, :, :] - output_images[0, 0, :, :, :])
    # axes["X"].imshow(expected_images[0, 1, :, :, :] - output_images[0, 1, :, :, :])
    # axes["Y"].imshow(expected_images[0, 2, :, :, :] - output_images[0, 2, :, :, :])
    # axes["Q"].imshow(expected_images[0, 3, :, :, :] - output_images[0, 3, :, :, :])
    plt.savefig("Hey_look_at_me.png", dpi=500)
    testing = long_term_prediction(
        model, 12, 20, image_size, timestep, image_frames, 200, 0.001,
        round_result=False, extra=True
    )
    prediction_proper = testing
    make_gif(testing, 'samples/without_rounding_with_extras_small')
    testing = long_term_prediction(
        model, 12, 20, image_size, timestep, image_frames, 200, 0.001,
        round_result=False, extra=False
    )
    make_gif(testing, 'samples/without_rounding_without_extras_small')
    testing = long_term_prediction(
        model, 12, 20, image_size, timestep, image_frames, 200, 0.001,
        round_result=True, extra=True
    )
    make_gif(testing, 'samples/with_rounding_with_extras_small')
    testing = long_term_prediction(
        model, 12, 20, image_size, timestep, image_frames, 200, 0.001,
        round_result=True, extra=False
    )
    make_gif(testing, 'samples/with_rounding_without_extras_small')
    testing = long_term_prediction(
        model, 12, 20, image_size, timestep, image_frames, 200, 0.001,
        round_result=True, extra=False, dry_run=True
    )
    actual_proper = testing
    make_gif(testing, 'samples/actual_data_small')
    plt.clf()
    plt.close()
    prediction_proper = np.asarray(prediction_proper)[:, :, :, 1]
    actual_proper = np.asarray(actual_proper)[:, :, :, 1]
    combined = np.zeros((200, image_size, image_size, 3), dtype=np.uint8)
    combined[:, :, :, 0] = prediction_proper
    combined[:min(np.shape(actual_proper)[0], 200), :, :, 2] = actual_proper
    make_gif(combined, 'samples/comparison')
    plt.clf()
    plt.close()
    plt.grid()
    for simulation in range(12, 13):
        sim_com = []
        data = long_term_prediction(
            model, simulation, 30, image_size, timestep, image_frames, 200, 0.001,
            round_result=False, extra=False, dry_run=True
        )
        for datapoint in data:
            sim_com.append(calculate_com(datapoint))
        plt.plot(sim_com)
        saved = sim_com
        sim_com = []
        data = long_term_prediction(
            model, simulation, 30, image_size, timestep, image_frames, 200, 0.001,
            round_result=False, extra=True, dry_run=False
        )
        for datapoint in data:
            sim_com.append(calculate_com(datapoint))
        saved_2 = sim_com
        plt.plot(np.arange(len(sim_com)), sim_com, '--')
    #plt.show()
    plt.savefig("centre_of_mass_tests.png")
    # exit()
    plt.clf()
    plt.close()
    # plt.plot((np.asarray(saved) - np.asarray(saved_2)))
    # plt.savefig("Centre_of_mass_difference.png")
    # plt.clf()
    # plt.close()
    product = dat_to_training.process_bmp("Simulation_data_extrapolated/Simulation_True_0_0.001_0/data_4.npy", image_size)
    product = product[:, :, 1]
    product = product.flatten()
    plt.hist(product[product > 0], bins=20)
    plt.savefig("Random_value_distribution.png", dpi=500)
    overall_loss = history.history["loss"]
    bce = history.history["binary_crossentropy"]
    mse = history.history["mean_squared_logarithmic_error"]
    ssim = history.history["ssim_loss"]
    plt.clf()
    plt.close()
    plt.plot(overall_loss, label="overall loss")
    plt.plot(bce, label="binary cross entropy")
    plt.plot(mse, label="mean squared logarithmic error")
    ssim = np.asarray(ssim)
    ssim_adjusted = 1 / (1 + np.exp(-ssim))
    plt.plot(ssim_adjusted, label="SSIM (adjusted)")
    plt.grid()
    plt.yscale("log")
    plt.xlabel("Epoch number")
    plt.ylabel("Values/AU")
    plt.legend()
    plt.savefig("Losses_across_epochs.png", dpi=500)
    # number_of_ensembles = 10
    # number_of_samples = 500
    # predictions = ensemble_prediction(
    #     model, 29, 400, image_size, timestep, image_frames, 50, rng, number_of_ensembles, number_of_samples
    # )
    # for a in range(0, number_of_ensembles):
    #     predictions_slice = (predictions[:, a, :, :, :] * 255).astype(np.uint8)
    #     make_gif(predictions_slice, "samples/{}".format(a))

    # print(plot_performance(model, image_frames, image_size, timestep, name="Test"))
    # test_positions = long_term_prediction(model, 8, 20, image_size, timestep, image_frames, 200, round_result=False)
    # make_gif(test_positions, 'samples/without_rounding')
    # test_positions = long_term_prediction(model, 8, 20, image_size, timestep, image_frames, 200, round_result=True)
    # make_gif(test_positions, 'samples/with_rounding')


if __name__ == "__main__":
    main()
