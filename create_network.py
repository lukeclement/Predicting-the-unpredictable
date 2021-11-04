import numpy as np
import loss_functions
from tensorflow.keras import layers, models


def interpret_model_summary(model):
    line_list = []
    model.summary(line_length=70, print_fn=lambda x: line_list.append(x))
    # print(line_list)
    for line in line_list:
        if "Trainable params:" in line:
            return line


def create_neural_network(activation, optimizer, loss, input_frames, image_size=64, channels=3, encode_size=2,
                          allow_upsampling=True, allow_pooling=True, kernel_size=3, max_transpose_layers=3):
    """Generates a Convolutional Neural Network (CNN), based on a sequential architecture. Does not train the CNN.
    This function can be adjusted to change the overall architecture of the CNN.
    This function also prints the model summary,
    allowing for an estimation on training time to be established, among other things.
    Inputs:
        activation:     A string of the activation function used on the neurons.
        optimizer:      A string of the optimisation function used on the model.
        loss:           A function that represents the loss on the network.
        input_frames:   An integer representing the number of reference images to be passed to the network.
        image_size:     (default 64) An integer of the size of an axis of the images to be passed to the network.
        channels:       (default 3) An integer for the number of channels used in the image. RGB images have 3 channels.
        encode_size:    (default 2) An integer for the size of an axis for the encoded image.
                            *Note* This is the final size achieved by ConvXD and MaxPoolingXD layers.
        allow_upsampling: (default True) A boolean to say whether to include upsampling layers.
        allow_pooling:  (default True) A boolean to say whether to include max pooling layers.
        kernel_size:    (default 3) An integer for the default size of kernels used in convolutional layers.
        max_transpose_layers: (default 3) An integer for the maximum number of transpose layers after the
                            final upsampling layer.
    Output:
        An untrained keras model
    """
    # TODO: dropout functions
    model = models.Sequential()
    current_axis_size = image_size
    target_axis_size = encode_size
    current_frames = input_frames
    model.add(layers.Conv3D(64, 1, activation=activation, input_shape=(input_frames, image_size, image_size, channels)))
    # Encoding the image
    while current_axis_size > target_axis_size:
        if current_frames > 1:
            # Initial 3D convolutional layers to reduce the frames into a single image
            model.add(layers.Conv3D(64, 2, activation=activation))
            current_frames -= 1
            current_axis_size -= 1
        elif current_frames == 1:
            # Reshaping the image to be the correct dimensions
            current_frames -= 1
            model.add(layers.Reshape((current_axis_size, current_axis_size, 64)))
            model.add(layers.Conv2D(64, kernel_size, activation=activation))
            current_axis_size -= (kernel_size - 1)
        else:
            # Bringing the image down to encoding size
            if np.floor(current_axis_size / 2) > target_axis_size and allow_pooling:
                model.add(layers.MaxPooling2D(2))
                current_axis_size = np.floor(current_axis_size / 2)
            if current_axis_size - (kernel_size - 1) < target_axis_size:
                model.add(layers.Conv2D(128, 2, activation=activation))
                current_axis_size -= 1
            else:
                model.add(layers.Conv2D(64, kernel_size, activation=activation))
                current_axis_size -= (kernel_size - 1)
    # Now decoding the image using transpose operations
    model.add(layers.Conv2DTranspose(128, kernel_size, activation=activation))
    current_axis_size += (kernel_size - 1)
    # Some variables to keep track of in the while loop
    max_leaps = max_transpose_layers
    leap_correction = 0
    calculated = False
    while current_axis_size < image_size:
        if current_axis_size * 2 < image_size and allow_upsampling:
            # Upsampling
            model.add(layers.UpSampling2D(2))
            current_axis_size = current_axis_size * 2
        if (image_size - current_axis_size) > (kernel_size - 1) * max_leaps \
                and not calculated \
                and (not allow_upsampling or current_axis_size * 2 > image_size):
            # Calculating the ideal kernel size for the fewest layers needed
            leaps_needed = np.floor((image_size - current_axis_size) / (kernel_size - 1))
            leap_correction = int(np.floor((kernel_size - 1) * (leaps_needed / max_leaps - 1)))
            calculated = True
        # Transpose operations
        if current_axis_size + kernel_size - 1 > image_size:
            # Close to full size
            model.add(layers.Conv2DTranspose(32, 2, activation=activation))
            current_axis_size += 1
        elif current_axis_size + kernel_size - 1 + leap_correction > image_size:
            # Within a few jumps of full size
            model.add(layers.Conv2DTranspose(32, kernel_size, activation=activation))
            current_axis_size += kernel_size - 1
        else:
            # Full size far away but too close for upsampling
            model.add(layers.Conv2DTranspose(64, kernel_size + leap_correction, activation=activation))
            current_axis_size += kernel_size - 1 + leap_correction
    # Final adjustments
    model.add(layers.Conv2DTranspose(1, 1, activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=[
        loss_functions.iou_coef, loss_functions.dice_coef, loss_functions.mass_preservation, loss_functions.com_coef
    ])
    return model


def train_model(model, training_images, validation_split=0.1, epochs=2):
    """Trains the model. This can take a while!
    Inputs:
        model:              A sequential model object.
        training_images:    A 2d array of training data, structured as [input images, expected output images].
                                *Note* This technically means the shape of training_images is not easily defined!
        validation_split:   (default 0.1) The split of training vs validation data.
        epochs:             (default 2) The number of epochs to perform.
    Outputs:
        model:              The fitted model.
        history:            The history of changes to important variables, like loss.
    """
    questions = training_images[0]
    answers = training_images[1]
    history = model.fit(questions, answers, epochs=epochs, validation_split=validation_split, shuffle=True)
    return model, history
