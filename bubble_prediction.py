import dat_to_training
import create_network
import loss_functions


def main():
    activation_function = "LeakyReLU"
    optimizer = "adam"
    loss_function = loss_functions.bce_dice
    image_frames = 4
    image_size = 64
    timestep = 5
    dat_to_training.convert_dat_files([0, 0], image_size=image_size)
    model = create_network.create_neural_network(
        activation_function, optimizer, loss_function, image_frames, image_size=image_size
    )
    dat_to_training.create_training_data(image_frames, timestep)


if __name__ == "__main__":
    main()