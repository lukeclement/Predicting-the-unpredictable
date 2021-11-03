from tensorflow.keras import layers, models, losses


def create_neural_network(activation, optimizer, loss, input_frames, image_size=64, channels=3):
    """Generates a Convolutional Neural Network (CNN), based on a sequential architecture. Does not train the CNN.
    This function can be adjusted to change the overall architecture of the CNN.
    This function also prints the model summary,
    allowing for an estimation on training time to be established, among other things.
    Inputs:
        activation: A string of the activation function used on the neurons.
        optimizer: A string of the optimisation function used on the model.
        loss: A function that represents the loss on the network.
        input_frames: An integer representing the number of reference images to be passed to the network.
        image_size: (default 64) An integer of the size of an axis of the images to be passed to the network.
    Output:
        An untrained keras model
    """
