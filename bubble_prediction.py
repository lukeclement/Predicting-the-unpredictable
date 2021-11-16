import numpy as np
import dat_to_training
import create_network
import loss_functions
import dask.array as da
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, optimizers, models
import tensorflow as tf


def main():
    tf.random.set_seed(100)
    activation_function = layers.LeakyReLU()
    optimizer = optimizers.Adam()
    loss_function = losses.BinaryCrossentropy()
    image_frames = 4
    image_size = 64
    timestep = 5
    # dat_to_training.convert_dat_files([0, 0], image_size=image_size)
    model = create_network.create_inception_net(activation_function, optimizer, loss_function)
    # model = create_network.create_neural_network(
    #     activation_function, optimizer, loss_function, image_frames,
    #     image_size=image_size, encode_size=13, allow_pooling=True,
    #     allow_upsampling=True, max_transpose_layers=1, kernel_size=2
    # )
    training_data = dat_to_training.create_training_data(image_frames, timestep, image_size=image_size)
    print(model.summary())
    model, history = create_network.train_model(model, training_data)
    # model.save("Test_model")
    # """
    model = models.load_model("Test_model")

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
    previous_frame = np.zeros((image_size, image_size, 1))
    previous_frame[:, :, 0] = initial[0, image_frames - 1, :, :, 1]

    positive_counts, negative_counts = difference_graphing(expected, guess, previous_frame)


    # """


def difference_graphing(expected, guess, previous_frame, plotting=True):
    rounded_guess = np.around(guess)

    real_difference = expected - previous_frame
    guess_difference = rounded_guess - previous_frame

    positive_real = np.zeros((64, 64, 1))
    positive_real[real_difference > 0] = 1
    positive_guess = np.zeros((64, 64, 1))
    positive_guess[guess_difference > 0] = 1
    positive_correct, positive_counts = real_guess_differences(positive_guess, positive_real)

    negative_real = np.zeros((64, 64, 1))
    negative_real[real_difference < 0] = 1
    negative_guess = np.zeros((64, 64, 1))
    negative_guess[guess_difference < 0] = 1
    negative_correct, negative_counts = real_guess_differences(negative_guess, negative_guess)

    if plotting:

        positive_rgb = difference_to_rgb(positive_correct)
        negative_rgb = difference_to_rgb(negative_correct)

        combined_rgb = negative_rgb + positive_rgb
        combined_rgb[(combined_rgb[:, :, 0] == 0) & (combined_rgb[:, :, 1] == 0) & (combined_rgb[:, :, 2] == 0), :] = 0

        plt.imshow(guess)
        plt.show()

        plt.imshow(expected)
        plt.show()

        plt.imshow(rounded_guess - expected)
        plt.show()

        plt.imshow(rounded_guess)
        plt.show()

        plt.imshow(expected - previous_frame)
        plt.show()

        plt.imshow(rounded_guess - previous_frame)
        plt.show()

        plt.imshow(positive_rgb)
        plt.show()

        plt.imshow(negative_rgb)
        plt.show()

        plt.imshow(combined_rgb)
        plt.show()

    return positive_counts, negative_counts


def difference_to_rgb(positive_correct):
    positive_rgb = np.zeros((64, 64, 3), dtype=int)
    positive_rgb[positive_correct[:, :, 0] == 0, :] = 0
    positive_rgb[positive_correct[:, :, 0] == 1, 0] = 255
    positive_rgb[positive_correct[:, :, 0] == 2, 1] = 255
    positive_rgb[positive_correct[:, :, 0] == 3, 2] = 255
    return positive_rgb


def real_guess_differences(positive_guess, positive_real):
    positive_correct = np.zeros((64, 64, 1))
    positive_correct[(positive_guess == positive_real) & (positive_real == 0)] = 0  # Background
    positive_correct[(positive_guess == positive_real) & (positive_real == 1)] = 2  # Correct Prediction (Green)
    positive_correct[(positive_guess != positive_real) & (positive_real == 0)] = 1  # Over prediction (Red)
    positive_correct[(positive_guess != positive_real) & (positive_real == 1)] = 3  # Under prediction (Blue)
    unique, counts = np.unique(positive_correct, return_counts=True)
    positive_counts = dict(zip(unique, counts))
    return positive_correct, positive_counts


if __name__ == "__main__":
    main()
