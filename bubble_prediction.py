import numpy as np
import dat_to_training
import create_network
import loss_functions
import dask.array as da
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, optimizers
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
    model, history = create_network.train_model(model, training_data)
    model.save("Test_model")
    preamble = "Simulation_images/Simulation_8/img_"
    start = 20
    initial = np.zeros((1, image_frames, image_size, image_size, 3))
    expected = np.zeros((image_size, image_size, 1))
    for frame in range(0, image_frames):
        frame_to_load = "{}{}{}".format(preamble, start+frame*timestep, ".bmp")
        initial[0, frame, :, :, :] = np.asarray(Image.open(frame_to_load))/255
    expected_frame = "{}{}{}".format(preamble, start+image_frames*timestep, ".bmp")
    expected[:, :, 0] = np.asarray(Image.open(expected_frame))[:, :, 1]/255
    guess = model(initial)[0]
    print(np.shape(expected))
    print(np.shape(guess))

    plt.imshow(guess - expected)
    plt.colorbar()
    plt.show()

    plt.imshow(guess)
    plt.colorbar()
    plt.show()

    plt.imshow(expected)
    plt.colorbar()
    plt.show()

    plt.imshow((guess+1) / (expected+1))
    plt.colorbar()
    plt.show()

    rounded_guess = np.around(guess)

    plt.imshow(rounded_guess - expected)
    plt.colorbar()
    plt.show()

    plt.imshow(rounded_guess)
    plt.colorbar()
    plt.show()

    plt.imshow(expected)
    plt.colorbar()
    plt.show()

    plt.imshow((rounded_guess + 1) / (expected + 1))
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()