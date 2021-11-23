import numpy as np
from tensorflow.keras import layers, models, initializers, losses
import gc
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback

import loss_functions


def interpret_model_summary(model):
    line_list = []
    model.summary(line_length=70, print_fn=lambda x: line_list.append(x))
    # print(line_list)
    for line in line_list:
        if "Trainable params:" in line:
            return line


def create_neural_network(activation, optimizer, loss, input_frames, image_size=64, channels=3, encode_size=2,
                          allow_upsampling=True, allow_pooling=True, kernel_size=3, max_transpose_layers=3,
                          dropout_rate=0.2):
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
        dropout_rate:   (default 0.2) A float for the rate that weights are dropped out in the dropout layers.
    Output:
        An untrained keras model
    """
    initializer = initializers.HeNormal()
    model = models.Sequential()
    current_axis_size = image_size
    target_axis_size = encode_size
    current_frames = input_frames
    model.add(layers.Conv3D(
        3, 1, activation=activation, kernel_initializer=initializer,
        input_shape=(input_frames, image_size, image_size, channels)
    ))
    # Encoding the image
    while current_axis_size > target_axis_size:
        if np.floor(current_axis_size / 2) > target_axis_size and allow_pooling and np.floor(current_frames / 2) > 1:
            model.add(layers.MaxPooling3D((1, 2, 2)))
            current_axis_size = np.floor(current_axis_size / 2)
        if current_axis_size - (kernel_size - 1) < target_axis_size:
            if current_frames > 1:
                model.add(layers.Conv3D(32, (2, 2, 2), activation=activation, kernel_initializer=initializer))
                current_axis_size -= 1
                current_frames -= 1
            else:
                model.add(layers.Conv3D(32, (1, 2, 2), activation=activation, kernel_initializer=initializer))
                current_axis_size -= 1
        else:
            model.add(layers.Conv3D(32, (1, kernel_size, kernel_size), activation=activation, kernel_initializer=initializer))
            current_axis_size -= (kernel_size - 1)
    while current_frames > 1:
        model.add(layers.Conv3D(32, (2, 1, 1), activation=activation, kernel_initializer=initializer))
        current_frames -= 1
    # Now decoding the image using transpose operations
    # model.add(layers.Dropout(rate=dropout_rate))
    # model.add(layers.Conv2DTranspose(64, kernel_size, activation=activation))
    # current_axis_size += (kernel_size - 1)
    # Some variables to keep track of in the while loop
    max_leaps = max_transpose_layers
    leap_correction = 0
    calculated = False
    first_run = True
    while current_axis_size < image_size:
        if current_frames == input_frames:
            if current_axis_size * 2 < image_size and allow_upsampling and not first_run:
                # Upsampling
                model.add(layers.UpSampling3D((1, 2, 2)))
                current_axis_size = current_axis_size * 2
            first_run = False
            if (image_size - current_axis_size) > (kernel_size - 1) * max_leaps \
                    and not calculated \
                    and (not allow_upsampling or current_axis_size * 2 > image_size or first_run):
                # Calculating the ideal kernel size for the fewest layers needed
                leaps_needed = np.floor((image_size - current_axis_size) / (kernel_size - 1))
                leap_correction = int(np.floor((kernel_size - 1) * (leaps_needed / max_leaps - 1)))
                calculated = True
            # Transpose operations
            if current_axis_size + kernel_size - 1 > image_size:
                # Close to full size
                model.add(layers.Conv3DTranspose(32, (1, 2, 2), activation=activation, kernel_initializer=initializer))
                current_axis_size += 1
            elif current_axis_size + kernel_size - 1 + leap_correction > image_size:
                # Within a few jumps of full size
                model.add(layers.Conv3DTranspose(32, (1, kernel_size, kernel_size),
                                                 activation=activation, kernel_initializer=initializer))
                current_axis_size += kernel_size - 1
            else:
                # Full size far away but too close for upsampling
                model.add(layers.Conv3DTranspose(
                    32, (1, kernel_size + leap_correction, kernel_size + leap_correction),
                    activation=activation, kernel_initializer=initializer
                ))
                current_axis_size += kernel_size - 1 + leap_correction
        else:
            model.add(layers.Conv3DTranspose(32, (2, kernel_size, kernel_size),
                                             activation=activation, kernel_initializer=initializer))
            current_axis_size += kernel_size - 1
            current_frames += 1

    # Final adjustments
    model.add(layers.Conv3DTranspose(1, 1, activation='sigmoid', kernel_initializer=initializer))
    print(model.summary(line_length=100))
    # model.compile(optimizer=optimizer, loss=loss, run_eagerly=False, metrics=[loss_functions.bce_dice, losses.BinaryCrossentropy()])
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False)
    return model


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


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
    print("Got to training!")
    # x = training_images[0]
    # y = training_images[1]
    # gc.collect()
    # history = model.fit(x, y, validation_split=validation_split, epochs=epochs, shuffle=True)

    # questions = training_images[0]
    # validation = training_images[1]
    # history = model.fit(questions, validation_data=validation, epochs=epochs, shuffle=True)

    data = training_images[0]
    validation = training_images[1]
    history = model.fit(data, epochs=epochs, shuffle=True, validation_data=validation)

    return model, history
