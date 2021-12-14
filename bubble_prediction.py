import numpy as np
import dat_to_training
import create_network
import loss_functions
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, optimizers, models
import tensorflow as tf
import imageio
from array2gif import write_gif
from datetime import date

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

        plt.imshow(previous_frame)
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


def evaluate_model(data, image_frames, image_size, model, timestep, simulation, start):
    print("Simulation " + str(simulation))
    test_positions = long_term_prediction(data, model, simulation, start, image_size, timestep, image_frames, 200,
                                          round_result=False)
    make_gif(test_positions, 'gifs/without_rounding_' + str(simulation))
    # test_positions = long_term_prediction(data, model, simulation, start, image_size, timestep, image_frames, 200,
    #                                       round_result=True)
    # make_gif(test_positions, 'gifs/with_rounding_' + simulation)


def long_term_prediction(data, model, start_sim, start_image, image_size, timestep, frames, number_to_simulate, round_result=False):
    input_images = data[0][0]
    image = input_images[0, 0, :, :, :]
    dat_to_training.generate_rail(image)
    rail = [image[:, :, 2:]]
    zeros = [image[:, :, 0:1]]
    working_frames = input_images[start_image:start_image+1]
    y_pred = model(working_frames).numpy()
    y_pred = np.append(zeros, y_pred, 3)
    y_pred = np.append(y_pred, rail, 3)
    output_images = y_pred
    y_pred_shape = np.shape(y_pred)
    y_pred = np.reshape(y_pred, newshape=(y_pred_shape[0], 1, y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]))
    working_frames = working_frames[:, 1:, :, :, :]
    working_frames = np.append(working_frames, y_pred, 1)
    for i in range(0, number_to_simulate - 1):
        y_pred = model(working_frames).numpy()
        y_pred = np.append(zeros, y_pred, 3)
        y_pred = np.append(y_pred, rail, 3)
        output_images = np.append(output_images, y_pred, axis=0)
        y_pred = np.reshape(y_pred, newshape=(y_pred_shape[0], 1, y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]))
        working_frames = working_frames[:, 1:, :, :, :]
        working_frames = np.append(working_frames, y_pred, 1)
    output_images = (output_images*255).astype(np.uint8)
    return output_images


def make_gif(image, name):
    images = []
    for i in image:
        images.append(i)
    imageio.mimsave("{}.gif".format(name), images)


def main():
    tf.random.set_seed(100)
    activation_function = layers.LeakyReLU()
    optimizer = optimizers.Adam()
    loss_function = losses.MeanSquaredError()
    image_frames = 4
    loss_frames = 2
    image_size = 60
    timestep = 5
    focus = 1

    today = date.today()
    dt_string = today.strftime("%d_%m_%Y_%H_%M")
    directory = "saved_models/" + dt_string

    # dat_to_training.save_training_data(focus, image_size)
    # training_data = dat_to_training.load_training_data(image_frames,
    #                                                    timestep=timestep,
    #                                                    image_size=image_size,
    #                                                    numpy_=True,
    #                                                    validation_split=0.001)
    #
    # model = create_network.create_inception_net(activation_function, optimizer, loss_function)
    # model, history = create_network.train_model(model, training_data, epochs=1)
    # model.save(directory)
    # for i in range(20):
    #     data = dat_to_training.load_training_data(image_frames, timestep=timestep, image_size=image_size, numpy_=True, simulation_num=i)
    #     evaluate_model(data, image_frames, image_size, model, timestep, i, 400)

    data = dat_to_training.load_training_data(image_frames, timestep=1, image_size=image_size, numpy_=True, simulation_num=1)
    plt.imshow(data[0][0][747])
    plt.show()

    # """
    # print("Loading Model")
    # model = models.load_model(directory, custom_objects={"bce_dice": loss_functions.bce_dice})
    # for i in range(20):
    #     data = dat_to_training.create_training_data(image_frames, timestep=timestep, image_size=image_size, simulation_num=i, numpy_=True)
    #     evaluate_model(data, image_frames, image_size, model, timestep, str(i), 400)

    # start = 200
    # simulation = '1'
    # preamble = "Simulation_images/Simulation_" + simulation + "/img_"
    # initial = np.zeros((1, image_frames, image_size, image_size, 3))
    # expected = np.zeros((image_size, image_size, 1))
    # for frame in range(0, image_frames):
    #     frame_to_load = "{}{}{}".format(preamble, start + frame * timestep, ".bmp")
    #     initial[0, frame, :, :, :] = np.asarray(Image.open(frame_to_load)) / 255
    # expected_frame = "{}{}{}".format(preamble, start + image_frames * timestep, ".bmp")
    # expected[:, :, 0] = np.asarray(Image.open(expected_frame))[:, :, 1] / 255
    # # guess = model(initial)[0]
    # previous_frame = np.zeros((image_size, image_size, 1))
    # previous_frame[:, :, 0] = initial[0, image_frames - 1, :, :, 1]
    # plt.imshow(previous_frame)
    # plt.show()
    # positive_counts, negative_counts = difference_graphing(expected, guess, previous_frame, plotting=plotting)
    # x, y = dat_to_training.read_file("Simulation_data/r0.54eps18.6/boundaries_840.dat")
    # plt.plot(x, y)
    # plt.show()
    # """

if __name__ == "__main__":
    main()
