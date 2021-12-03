import dat_to_training
import create_network
import loss_functions
from tensorflow.keras import layers, losses, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from time import time
from numpy.random import default_rng
import imageio


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


def ensemble_prediction(
        model, start_sim, start_image, image_size, timestep, frames, number_to_simulate, rng, ensemble_size=5,
        samples=200, focus=1):
    print("Starting ensemble")
    input_images = np.zeros((ensemble_size, frames, image_size, image_size, 3))
    for frame in range(0, frames):
        try:
            insert_array = dat_to_training.process_bmp(
                "Simulation_images/Simulation_{}/img_{}.bmp".format(
                    start_sim, start_image + frame * timestep
                ), image_size, focus=focus
            )
        except IOError as e:
            print("Error - either invalid simulation number or image out of range!")
            print(e)
            return []
        for s in range(0, ensemble_size):
            input_images[s, frame, :, :, :] = insert_array
    predictions = []
    predictions = np.zeros((number_to_simulate, ensemble_size, image_size, image_size, 3))
    next_images = np.zeros((ensemble_size, image_size, image_size, 3))
    for i in range(0, number_to_simulate):
        print(i)
        next_images[:, :, :, 1] = model(input_images)[:, :, :, 0]
        possible_images = np.zeros((ensemble_size * samples, image_size, image_size, 3))
        print("Generating samples...")
        for j, image in enumerate(next_images):
            for s in range(0, samples):
                possible_images[j * samples + s, :, :, :] = generate_sample(image, image_size, rng)
        print("Selecting best...")
        unique_images, frequency = np.unique(possible_images, return_counts=True, axis=0)
        print(len(frequency))
        for s in range(0, min(ensemble_size, len(frequency))):
            current_best = np.amax(frequency)
            current_best_index, = np.where(frequency == current_best)
            next_images[s, :, :, :] = unique_images[current_best_index[0], :, :, :]
            unique_images = np.delete(unique_images, current_best_index[0], 0)
            frequency = np.delete(frequency, current_best_index[0], 0)
        for frame in range(1, frames):
            input_images[:, frame - 1, :, :, :] = input_images[:, frame, :, :, :]
        input_images[:, frames - 1, :, :, :] = next_images
        predictions[i, :, :, :, :] = next_images
    return predictions


def long_term_prediction(
        model, start_sim, start_image, image_size, timestep, frames, number_to_simulate, round_result=False,
        extra=True, jump=False, focus=1, dry_run=False):
    input_images = np.zeros((1, frames, image_size, image_size, 3))
    for frame in range(0, frames):
        try:
            input_images[0, frame, :, :, :] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_{}/data_{}.npy".format(
                    start_sim, start_image + frame * timestep
                ), image_size, focus=focus
            )
        except IOError as e:
            print("Error - either invalid simulation number or image out of range!")
            print(e)
            return []
    positions = []
    if dry_run:
        try:
            for i in range(0, number_to_simulate):
                positions.append((dat_to_training.process_bmp(
                    "Simulation_data_extrapolated/Simulation_{}/data_{}.npy".format(
                        start_sim, start_image + i * timestep
                    ), image_size, focus=focus
                ) * 255).astype(np.uint8))
        except:
            return positions
        return positions
    if jump:
        for i in range(0, number_to_simulate):
            output_image = np.zeros((frames, image_size, image_size, 3))
            output_image[:, :, :, 1] = model(input_images)[0, :, :, :, 0]
            for frame in range(0, frames):
                dat_to_training.generate_rail(output_image[frame])
            for frame in range(0, frames):
                if round_result:
                    input_images[0, frame, :, :, 1] = np.around(output_image[frame, :, :, 1])
                    input_images[0, frame, :, :, 0] = output_image[frame, :, :, 0]
                    input_images[0, frame, :, :, 2] = output_image[frame, :, :, 2]
                else:
                    input_images[0, frame, :, :, :] = output_image[frame, :, :, :]
                positions.append((input_images[0, frame, :, :, :] * 255).astype(np.uint8))
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


def main():
    # activation_function = "LeakyReLU"
    tf.random.set_seed(100)
    rng = default_rng(69420)
    activation_function = layers.LeakyReLU()
    optimizer = "adam"
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
    loss_function = loss_functions.UBERLOSS
    # loss_function = loss_functions.ssim_loss
    # Parameter ranges
    image_frame_range = [1, 6]
    image_size_range = [20, 70]
    timestep_range = [1, 21]
    dropout_range = [0, 0.5]
    encode_range = [1, 21]
    max_transpose_range = [1, 10]
    kernel_range = [2, 10]
    focus_range = [1, 2]
    epochs = 10

    allowed_sizes = [12, 15, 18, 20, 27, 30, 36, 45, 54, 60, 90, 108, 135, 180, 270]
    allowed_sizes = [30, 36, 45, 54, 60, 90, 108, 135, 180, 270]
    allowed_sizes = [45, 54, 60, 90, 108, 135, 180, 270]

    image_frames = 4
    image_size = allowed_sizes[0]
    timestep = 5
    dropout_rate = 0.1
    encode_size = 3
    # howdy
    max_transpose_layers = 7
    kernel_size = 3
    multiply = 3
    kernel_size_data = 7
    focus = 1
    dropout = 0
    dat_to_training.convert_dat_files([0, 0], resolution=0.01)

    trainable_parameters = []
    try:
        # first = True
        # while image_frames * (image_size ** 2) > 4 * (45 ** 2) or first:
        #     first = False
        #     image_size = allowed_sizes[generate_random_value(rng, [0, len(allowed_sizes)])]
        #     image_frames = generate_random_value(rng, image_frame_range)
        # timestep = generate_random_value(rng, timestep_range)
        # dropout = 0
        # encode_size = generate_random_value(rng, encode_range)
        # max_transpose_layers = generate_random_value(rng, max_transpose_range)
        # kernel_size = generate_random_value(rng, kernel_range)
        # focus = generate_random_value(rng, focus_range)
        print("frame_{};size_{};time_{};drop_{};encode_{};maxTrans_{};kernel_{};focus_{};".format(
            image_frames, image_size, timestep, dropout, encode_size, max_transpose_layers, kernel_size, focus
        ))
        model = create_network.create_neural_network(
            activation_function, optimizer, loss_function, image_frames,
            image_size=image_size, encode_size=encode_size, allow_pooling=True,
            allow_upsampling=True, max_transpose_layers=max_transpose_layers, kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )
        parameters_line = create_network.interpret_model_summary(model)
        print(parameters_line)
        print(model.summary())
        training_data = dat_to_training.create_training_data(image_frames, timestep, image_size=image_size,
                                                             focus=focus)
        model, history = create_network.train_model(model, training_data, epochs=epochs)
        model.save(
            "models/Proper-frame_{};size_{};time_{};drop_{};encode_{};maxTrans_{};kernel_{};focus_{};".format(
                image_frames, image_size, timestep, dropout, encode_size, max_transpose_layers, kernel_size, focus
            ))
    except Exception as e:
        print("Fail!")
        print(e)

    output_images = np.zeros((1, image_frames, image_size, image_size, 1))
    input_images = np.zeros((1, image_frames, image_size, image_size, 3))
    expected_images = np.zeros((1, image_frames, image_size, image_size, 1))
    for frame in range(0, image_frames):
        try:
            input_images[0, frame, :, :, :] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_{}/data_{}.npy".format(
                    6, 20 + frame * timestep
                ), image_size, focus)
            expected_images[0, frame, :, :, 0] = dat_to_training.process_bmp(
                "Simulation_data_extrapolated/Simulation_{}/data_{}.npy".format(
                    6, 20 + (frame + image_frames) * timestep
                ), image_size, focus)[:, :, 1]
        except IOError as e:
            print("Error - either invalid simulation number or image out of range!")
            print(e)
    output_images = model(input_images)
    axes = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        ABCD
        EFGH
        IJKL
        """
    )
    # Input
    axes["A"].imshow(input_images[0, 0, :, :, :])
    axes["B"].imshow(input_images[0, 1, :, :, :])
    axes["C"].imshow(input_images[0, 2, :, :, :])
    axes["D"].imshow(input_images[0, 3, :, :, :])
    # Prediction
    axes["E"].imshow(output_images[0, 0, :, :, :])
    axes["F"].imshow(output_images[0, 1, :, :, :])
    axes["G"].imshow(output_images[0, 2, :, :, :])
    axes["H"].imshow(output_images[0, 3, :, :, :])
    # Actual
    axes["I"].imshow(expected_images[0, 0, :, :, :])
    axes["J"].imshow(expected_images[0, 1, :, :, :])
    axes["K"].imshow(expected_images[0, 2, :, :, :])
    axes["L"].imshow(expected_images[0, 3, :, :, :])
    plt.savefig("Hey_look_at_me.png", dpi=500)
    testing = long_term_prediction(
        model, 10, 20, image_size, timestep, image_frames, 200,
        round_result=False, extra=True, focus=focus
    )
    make_gif(testing, 'samples/without_rounding_with_extras_small')
    testing = long_term_prediction(
        model, 10, 20, image_size, timestep, image_frames, 200,
        round_result=False, extra=False, focus=focus
    )
    make_gif(testing, 'samples/without_rounding_without_extras_small')
    testing = long_term_prediction(
        model, 10, 20, image_size, timestep, image_frames, 200,
        round_result=True, extra=True, focus=focus
    )
    make_gif(testing, 'samples/with_rounding_with_extras_small')
    testing = long_term_prediction(
        model, 10, 20, image_size, timestep, image_frames, 200,
        round_result=True, extra=False, focus=focus
    )
    make_gif(testing, 'samples/with_rounding_without_extras_small')
    testing = long_term_prediction(
        model, 10, 20, image_size, timestep, image_frames, 200,
        round_result=True, extra=False, focus=focus, dry_run=True
    )
    make_gif(testing, 'samples/actual_data_small')

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
