import dat_to_training
import create_network
import loss_functions
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def plot_performance(model, image_frames, image_size, timestep):
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
    plt.savefig("Test_on_simulation_8.png", dpi=500)

    return np.sum(difference)


def main():
    # activation_function = "LeakyReLU"
    activation_function = layers.LeakyReLU()
    optimizer = "adam"
    loss_function = loss_functions.bce_dice
    image_frames = 4
    image_size = 64
    timestep = 5
    dat_to_training.convert_dat_files([0, 0], image_size=image_size)
    model = create_network.create_neural_network(
        activation_function, optimizer, loss_function, image_frames,
        image_size=image_size, encode_size=5, allow_pooling=True,
        allow_upsampling=True, max_transpose_layers=3, kernel_size=2,
        dropout_rate=0.2
    )
    training_data = dat_to_training.create_training_data(image_frames, timestep, image_size=image_size)
    model, history = create_network.train_model(model, training_data, epochs=2)
    print(plot_performance(model, image_frames, image_size, timestep))


if __name__ == "__main__":
    main()
