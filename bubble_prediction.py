import dat_to_training
import create_network
import loss_functions
from tensorflow.keras import layers, optimizers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio


def long_term_prediction(
        model, start_sim, start_image, image_size, timestep, frames, number_to_simulate, resolution, dry_run=False):
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
        except OSError:
            return positions
        return positions
    output_image = np.zeros((image_size, image_size, 3))
    for i in range(0, number_to_simulate):
        output_image[:, :, 1] = model(input_images)[0, :, :, 0]
        dat_to_training.generate_rail(output_image)
        for frame in range(0, frames-1):
            input_images[0, frame, :, :, :] = input_images[0, frame+1, :, :, :]
        input_images[0, frames-1, :, :, :] = output_image
        positions.append((input_images[0, frames-1, :, :, :] * 255).astype(np.uint8))
    return positions


def make_gif(image, name):
    images = []
    for i in image:
        images.append(i)
    imageio.mimsave("{}.gif".format(name), images)


def calculate_com(image, both=False):
    image_size = np.shape(image)[0]
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')
    x_com = np.sum(x*image[:, :, 1])/np.sum(image[:, :, 1])
    y_com = np.sum(y*image[:, :, 1])/np.sum(image[:, :, 1])
    if both:
        return -x_com+(float(image_size)/2), y_com-(float(image_size)/2)
    return np.sqrt((x_com-(float(image_size)/2))**2 + (y_com-(float(image_size)/2))**2)


def main():
    tf.random.set_seed(100)
    activation_function = layers.LeakyReLU()
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule, epsilon=0.1)
    # optimizer = optimizers.Adam()

    parameters_extra = [
        [loss_functions.UBERLOSS, 1, 64, 5, True, True, 5, 3, 0.001, [0], True, [0], 5, "Andover", 20, True],
    ]
    scenario = 6
    for parameters in parameters_extra:
        loss_function = parameters[0]
        epochs = parameters[14]
        image_frames = parameters[1]
        image_size = parameters[2]
        timestep = parameters[12]
        encode_size = parameters[3]
        resolution = parameters[8]
        max_transpose_layers = parameters[6]
        kernel_size = parameters[7]
        dropout_rate = 0
        name = parameters[13]
        linearity = parameters[15]
        if scenario == 0:
            model = create_network.create_inception_network(
                activation_function, optimizer, loss_function, image_frames,
                image_size=image_size, encode_size=encode_size, allow_pooling=True,
                allow_upsampling=True, max_transpose_layers=max_transpose_layers, kernel_size=kernel_size,
                dropout_rate=dropout_rate, inception=linearity, simple=False
            )
        elif scenario == 1:
            model = create_network.create_basic_network(
                activation_function, optimizer, loss_function, image_frames, image_size
            )
        elif scenario == 2:
            model = create_network.create_transformer_network(
                activation_function, optimizer, loss_function, image_frames, image_size
            )
        elif scenario == 3:
            model = create_network.create_inception_transformer_network(
                activation_function, optimizer, loss_function, image_frames, image_size=image_size
            )
        elif scenario == 4:
            model = create_network.create_resnet(
                activation_function, optimizer, loss_function, image_frames, image_size=image_size, inception=linearity
            )
        elif scenario == 5:
            model = create_network.create_parallel_network(
                activation_function, optimizer, loss_function, image_frames,
                image_size=image_size, encode_size=encode_size, allow_pooling=True,
                allow_upsampling=True, max_transpose_layers=max_transpose_layers, kernel_size=kernel_size,
                dropout_rate=dropout_rate, inception=linearity
            )
        elif scenario == 6:
            # vae = create_network.VAE(
            #     create_network.create_encoder(6, image_size, image_frames),
            #     create_network.create_decoder(6),
            #     (image_size, image_size, 1)
            # )
            # vae.compile(optimizer=optimizer)
            # training_data = dat_to_training.create_training_data(
            #     image_frames, timestep, image_size=image_size,
            #     excluded_sims=[12], variants=[0], resolution=resolution, flips_allowed=False, easy_mode=False, var=True)
            # print(training_data[0])
            # vae.fit(training_data[0], epochs=1)
            # vae.save("models/{}".format("VAR"))
            #
            # samples = 30
            # figure = np.zeros((image_size*samples, image_size*samples))
            # grid_x = np.linspace(-1, 1, samples)
            # grid_y = np.linspace(-1, 1, samples)[::-1]
            #
            # for i, yi in enumerate(grid_y):
            #     for j, xi in enumerate(grid_x):
            #         z_sample = np.array([[xi, yi]])
            #         x_decoded = vae.decoder.predict(z_sample)
            #         digit = x_decoded[0].reshape(image_size, image_size)
            #         figure[
            #             i * image_size: (i + 1) * image_size,
            #             j * image_size: (j + 1) * image_size,
            #         ] = digit
            #
            # plt.figure(figsize=(30, 30))
            # start_range = image_size // 2
            # end_range = samples * image_size + start_range
            # pixel_range = np.arange(start_range, end_range, image_size)
            # sample_range_x = np.round(grid_x, 1)
            # sample_range_y = np.round(grid_y, 1)
            # plt.xticks(pixel_range, sample_range_x)
            # plt.yticks(pixel_range, sample_range_y)
            # plt.xlabel("z[0]")
            # plt.ylabel("z[1]")
            # plt.imshow(figure, cmap="Greys_r")
            # plt.savefig("VAE_test.png", dpi=1000)
            # plt.show()
            sample = np.array([[1, 2, -3, 4, 5, -6]])
            sample = np.tanh(sample)
            decoded = vae.decoder.predict(sample)
            plt.imshow(decoded)
            plt.show()
            exit()
        else:
            model = create_network.create_basic_network(
                activation_function, optimizer, loss_function, image_frames, image_size
            )
        print(name)
        print(model.summary())
        # exit()
        training_data = dat_to_training.create_training_data(
            image_frames, timestep, image_size=image_size,
            excluded_sims=[12], variants=[0], resolution=resolution, flips_allowed=False, easy_mode=False)
        try:
            model, history = create_network.train_model(model, training_data, epochs=epochs)
            model.save("models/{}".format(name))
            overall_loss = history.history["loss"]
            overall_val = history.history["val_loss"]
            bce = history.history["binary_crossentropy"]
            mse = history.history["mean_squared_logarithmic_error"]
            plt.clf()
            plt.close()
            plt.plot(overall_loss, label="overall loss")
            plt.plot(overall_val, linestyle="dashed", label="overall validation loss")
            plt.plot(bce, label="binary cross entropy")
            plt.plot(mse, label="mean squared logarithmic error")
            plt.grid()
            plt.yscale("log")
            plt.xlabel("Epoch number")
            plt.ylabel("Values/AU")
            plt.legend()
            plt.savefig("model_performance/{}_Losses_across_epochs.png".format(name), dpi=500)
        except Exception as e:
            print(e)

        del model
        del training_data


if __name__ == "__main__":
    main()
