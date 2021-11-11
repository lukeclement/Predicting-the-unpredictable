import dat_to_training
import create_network
import loss_functions
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from time import time


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
    difference = np.abs(np.around(guess)-expected)

    axes = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        A.C
        BDE
        """
    )
    axes["B"].imshow(guess - expected)
    axes["C"].imshow(guess)
    axes["A"].imshow(expected)
    cool = np.zeros((image_size, image_size, 3))
    cool[:, :, 0] = guess[:, :, 0]
    cool[:, :, 1] = expected[:, :, 0]
    axes["D"].imshow(cool)
    big_difference = (guess - expected) / (guess + expected + 1)
    axes["E"].imshow(big_difference)
    plt.savefig("{}.png".format(name), dpi=500)

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
                        except:
                            print("Fail!")
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
                    ).replace(".", "_")+".csv", index=False)
    return parameter_data


def main():
    # activation_function = "LeakyReLU"
    tf.random.set_seed(100)
    activation_function = layers.LeakyReLU()
    optimizer = "adam"
    loss_function = loss_functions.bce_dice
    image_frames = 1
    image_size = 44
    timestep = 1
    dropout_rate = 0.2
    dat_to_training.convert_dat_files([-1, 1], image_size=image_size)
    model = create_network.create_neural_network(
        activation_function, optimizer, loss_function, image_frames,
        image_size=image_size, encode_size=5, allow_pooling=True,
        allow_upsampling=True, max_transpose_layers=3, kernel_size=2,
        dropout_rate=dropout_rate
    )
    training_data = dat_to_training.create_training_data(image_frames, timestep, image_size=image_size)
    model, history = create_network.train_model(model, training_data, epochs=2)
    print(plot_performance(model, image_frames, image_size, timestep, name="Test"))


if __name__ == "__main__":
    main()
