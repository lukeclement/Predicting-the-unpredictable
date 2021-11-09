import dat_to_training
import create_network
import loss_functions
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # activation_function = "LeakyReLU"
    activation_function = layers.LeakyReLU()
    optimizer = "adam"
    loss_function = loss_functions.bce_dice
    image_frames = 4
    image_size = 64
    timestep = 5
    # dat_to_training.convert_dat_files([0, 0], image_size=image_size)
    model = create_network.create_neural_network(
        activation_function, optimizer, loss_function, image_frames,
        image_size=image_size, encode_size=13, allow_pooling=True,
        allow_upsampling=True, max_transpose_layers=1, kernel_size=2
    )
    training_data = dat_to_training.create_training_data(image_frames, timestep, image_size=image_size)
    model, history = create_network.train_model(model, training_data, epochs=2)
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


if __name__ == "__main__":
    main()
