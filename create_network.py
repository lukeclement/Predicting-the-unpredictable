import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, initializers, losses, optimizers, activations, Input, metrics
import gc
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

import bubble_prediction
import loss_functions


def interpret_model_summary(model):
    line_list = []
    model.summary(line_length=70, print_fn=lambda x: line_list.append(x))
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
        if current_axis_size - (kernel_size - 1) < target_axis_size:
            if current_frames > 1:
                model.add(layers.Conv3D(32, (2, 2, 2), activation=activation, kernel_initializer=initializer))
                current_axis_size -= 1
                current_frames -= 1
            else:
                model.add(layers.Conv3D(32, (1, 2, 2), activation=activation, kernel_initializer=initializer))
                current_axis_size -= 1
        else:
            model.add(
                layers.Conv3D(32, (1, kernel_size, kernel_size), activation=activation, kernel_initializer=initializer))
            current_axis_size -= (kernel_size - 1)
        if np.floor(current_axis_size / 2) > target_axis_size and allow_pooling:
            model.add(layers.MaxPooling3D((1, 2, 2)))
            current_axis_size = np.floor(current_axis_size / 2)
    while current_frames > 1:
        model.add(layers.Conv3D(64, (2, 1, 1), activation=activation, kernel_initializer=initializer))
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
    # print(model.summary(line_length=100))
    # model.compile(optimizer=optimizer, loss=loss, run_eagerly=False, metrics=[loss_functions.bce_dice, losses.BinaryCrossentropy()])
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=False, metrics=[
        losses.binary_crossentropy, losses.mean_squared_logarithmic_error, loss_functions.ssim_loss
    ])
    return model


def inception_cell(model, activation, initializer, channels, axis=3):
    shape = model.output_shape
    li = list(shape)
    li.pop(0)
    shape = tuple(li)
    input_tower = layers.Input(shape=shape)

    tower_1 = layers.Conv2D(channels, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)

    tower_2 = layers.Conv2D(channels, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)
    tower_2 = layers.Conv2D(channels, (3, 3), padding='same', activation=activation, kernel_initializer=initializer)(
        tower_2)

    tower_3 = layers.Conv2D(channels, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)
    tower_3 = layers.Conv2D(channels, (5, 5), padding='same', activation=activation, kernel_initializer=initializer)(
        tower_3)

    tower_4 = layers.MaxPooling2D(3, strides=1, padding='same')(input_tower)
    tower_4 = layers.Conv2D(channels, 1, padding='same', activation=activation, kernel_initializer=initializer)(
        tower_4)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)

    model.add(Model(input_tower, merged))
    return model


def inception_cell_revive(x, channels, axis=3):

    tower_1 = layers.Conv2D(channels, (1, 1), padding='same')(
        x)

    tower_2 = layers.Conv2D(channels, (1, 1), padding='same')(
        x)
    tower_2 = layers.Conv2D(channels, (3, 3), padding='same')(
        tower_2)

    tower_3 = layers.Conv2D(channels, (1, 1), padding='same')(
        x)
    tower_3 = layers.Conv2D(channels, (5, 5), padding='same')(
        tower_3)

    tower_4 = layers.MaxPooling2D(3, strides=1, padding='same')(x)
    tower_4 = layers.Conv2D(channels, 1, padding='same')(
        tower_4)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)
    return merged


def inception_cell_revive3d(x, channels, axis=3):

    tower_1 = layers.Conv3D(channels, (1, 1), padding='same')(
        x)

    tower_2 = layers.Conv3D(channels, (1, 1), padding='same')(
        x)
    tower_2 = layers.Conv3D(channels, (3, 3), padding='same')(
        tower_2)

    tower_3 = layers.Conv3D(channels, (1, 1), padding='same')(
        x)
    tower_3 = layers.Conv3D(channels, (5, 5), padding='same')(
        tower_3)

    tower_4 = layers.MaxPooling3D(3, strides=1, padding='same')(x)
    tower_4 = layers.Conv3D(channels, 1, padding='same')(
        tower_4)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)
    return merged


def convolutional_transformer(x, channels, activation):
    key = layers.Conv2D(channels, 1, padding='same')(x)
    query = layers.Conv2D(channels, 1, padding='same')(x)
    value = layers.Conv2D(channels, 1, padding='same')(x)
    attention_map = layers.Multiply()([key, query])
    attention_map = layers.Softmax()(attention_map)
    transformed = layers.Multiply()([attention_map, value])
    transformed = layers.Conv2D(channels, 1, padding='same', activation=activation)(transformed)
    return transformed


def inception_cell_transformer(x, channels, activation, axis=3):

    tower_1 = convolutional_transformer(x, channels, activation)
    # tower_1 = layers.Conv2D(channels, 1, padding='same', activation=activation)(x)

    tower_2 = layers.Conv2D(channels, 1, padding='same', activation=activation)(x)
    tower_2 = layers.Conv2D(channels, 3, padding='same', activation=activation)(tower_2)
    tower_2 = convolutional_transformer(tower_2, channels, activation)

    tower_3 = layers.Conv2D(channels, 1, padding='same', activation=activation)(x)
    tower_3 = layers.Conv2D(channels, 5, padding='same', activation=activation)(tower_3)
    tower_3 = convolutional_transformer(tower_3, channels, activation)

    tower_4 = layers.MaxPooling2D(3, strides=1, padding='same')(x)
    tower_4 = layers.Conv2D(channels, 1, padding='same', activation=activation)(tower_4)
    tower_4 = convolutional_transformer(tower_4, channels, activation)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)

    return merged


def create_inception_transformer_network(activation, optimizer, loss, input_frames, image_size=64, channels=3, num_layers=6):

    input_layer = layers.Input(shape=(input_frames, image_size, image_size, channels))
    x = input_layer
    while x.shape[1] > 1:
        x = layers.Conv3D(32, (2, 1, 1), activation=activation)(x)
    x = layers.Reshape((x.shape[2], x.shape[2], 32))(x)
    for _ in range(num_layers):
        x = inception_cell_transformer(x, 32, activation)
    x = layers.Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(input_layer, x)
    model.compile(optimizer=optimizer, loss=loss, metrics=[
        losses.binary_crossentropy, losses.mean_squared_logarithmic_error
    ])
    return model


def make_transformer_encoder(inputs, head_size, head_number, full_forward, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=head_number, dropout=dropout, attention_axes=1)(inputs, inputs, inputs)
    total = x + inputs
    total = layers.LayerNormalization(epsilon=1e-6)(total)
    x = layers.Dense(full_forward, activation=activations.swish)(total)
    x = layers.Dense(inputs.shape[-1], activation=activations.swish)(x)
    return layers.LayerNormalization(epsilon=1e-6)(total + x)
    # return total + x


def make_transformer_decoder(inputs, encodes, head_size, head_number, full_forward, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=head_number, dropout=dropout, attention_axes=1)(inputs, inputs, inputs)
    total = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    # total = x + inputs
    # VKQ VK->encodes, Q->total
    # QVK
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=head_number, dropout=dropout, attention_axes=1)(total, encodes, encodes)
    # total = x + total
    total = layers.LayerNormalization(epsilon=1e-6)(x + total)
    x = layers.Dense(full_forward, activation=activations.swish)(total)
    x = layers.Dense(inputs.shape[-1], activation=activations.swish)(x)
    return layers.LayerNormalization(epsilon=1e-6)(x + total)


def create_basic_transformer_network(activation, input_frames, data_points, layering=3):
    input_layer = layers.Input(shape=(input_frames, data_points))
    inputs = input_layer
    x = inputs
    y = inputs
    for _ in range(layering):
        x = make_transformer_encoder(x, 64, 8, 10)
        y = make_transformer_decoder(y, x, 64, 8, 10)

    model = Model(input_layer, y)
    # model.compile(optimizer=optimizer, loss=loss, metrics=[
    #     losses.binary_crossentropy, losses.mean_squared_logarithmic_error
    # ])
    return model


def create_transformer_network(activation, input_frames, image_size, channels=3, layering=1):
    input_layer = layers.Input(shape=(input_frames, image_size, image_size, channels))
    # x = layers.Reshape((input_frames*image_size*image_size*channels, 1))(input_layer)
    # y = layers.Reshape((input_frames*image_size*image_size*channels, 1))(input_layer)
    inputs = input_layer
    x = inputs
    y = inputs
    for _ in range(layering):
        x = make_transformer_encoder(x, 64, 8, 10)
        y = make_transformer_decoder(y, x, 64, 8, 10)

    while y.shape[1] > 2:
        y = layers.Conv3D(32, (2, 1, 1), activation=activations.swish)(y)
    if y.shape[1] == 2:
        y = layers.Conv3D(1, (2, 1, 1), activation=activations.swish)(y)
        y = layers.Reshape((image_size, image_size, 1))(y)
        # inputs = layers.Reshape((image_size, image_size, 1))(inputs)
    y = layers.Reshape((image_size, image_size, 1))(y)
    model = Model(input_layer, y)
    # model.compile(optimizer=optimizer, loss=loss, metrics=[
    #     losses.binary_crossentropy, losses.mean_squared_logarithmic_error
    # ])
    return model


def create_basic_network(activation, input_frames, image_size, channels=3, latent_dimensions=100, start_channels=32):
    input_layer = layers.Input(shape=(input_frames, image_size, image_size, channels))
    layer_depth = -1
    frames = input_frames
    size = image_size
    x = input_layer
    frames_ran = False
    while max(frames, 1) * size * size > latent_dimensions:
        layer_depth += 1
        if frames > 1:
            x = layers.Conv3D(start_channels*2**layer_depth, 5, strides=2, padding='same')(x)
            x = activation(x)
            frames = frames // 2
            size = size // 2
            frames_ran = True
        else:
            if frames == 1:
                if frames_ran:
                    x = layers.Reshape((size, size, int(start_channels*2**(layer_depth-1))))(x)
                else:
                    x = layers.Reshape((size, size, channels))(x)
                frames = 0
            x = layers.Conv2D(start_channels*2**layer_depth, 5, strides=2, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = activation(x)
            size = size // 2

    # x = layers.Flatten()(x)
    # x = layers.Dense(8 * 8 * 32 * 2**layer_depth)(x)
    # x = layers.BatchNormalization()(x)
    # x = activation(x)
    # x = layers.Reshape((8, 8, 32 * 2**layer_depth))(x)
    while size != image_size:
        layer_depth -= 1
        x = layers.Conv2DTranspose(start_channels*2**layer_depth, 5, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = activation(x)
        size = size * 2

    x = layers.Conv2DTranspose(1, 5, activation='sigmoid', padding='same', use_bias=False)(x)

    model = Model(input_layer, x)
    return model


# DEFAULT IS RESNET 50
def create_resnet(activation, input_frames,
                  image_size=64, channels=3, inception=True, structure=None, names=None):
    if names is None:
        names = ["alpha", "beta", "delta", "gamma"]
    if structure is None:
        structure = [3, 4, 6, 3]
    input_layer = layers.Input(shape=(input_frames, image_size, image_size, channels), name="Input layer")
    x = layers.Conv3D(64, (input_frames, 1, 1), activation='tanh')(input_layer)
    x = layers.Reshape((image_size, image_size, 64))(x)
    x = layers.Conv2D(64, 7, strides=2, activation=activation, padding='same')(x)
    x = layers.MaxPool2D(3, strides=2, padding='same')(x)
    for index, value in enumerate(structure):
        for i in range(value):
            x_temp = x
            if i == 0 and index != 0:
                stride_length = 2
            else:
                stride_length = 1
            x = layers.Conv2D(64 * (2 ** index), 1, strides=stride_length,
                              name="{}-{}".format(names[index], i))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Conv2D(64 * (2 ** index), 3, padding='same')(x)
            if inception:
                x = inception_cell_revive(x, channels=64 * (2 ** index))
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Conv2D(64 * (2 ** index) * 4, 1, activation=activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)

            if i == 0:
                x_temp = layers.Conv2D(64 * (2 ** index) * 4, 1, strides=stride_length)(x_temp)
                x_temp = layers.BatchNormalization()(x_temp)
                x_temp = layers.Activation(activation)(x_temp)
            x = layers.Add()([x, x_temp])

    for index, value in enumerate(structure):
        for i in range(value):
            x_temp = x
            if i == 0 and index != 0:
                stride_length = 2
            else:
                stride_length = 1
            x = layers.Conv2DTranspose(64 * (2 ** (index-4)) * 4, 1, strides=stride_length,
                              name="{}-{}-T".format(names[index], i))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Conv2DTranspose(64 * (2 ** (index-4)), 3, padding='same')(x)
            if inception:
                x = inception_cell_revive(x, channels=64 * (2 ** index))
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            x = layers.Conv2DTranspose(64 * (2 ** (index-4)), 1, activation=activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)

            if i == 0:
                x_temp = layers.Conv2DTranspose(64 * (2 ** (index-4)), 1, strides=stride_length)(x_temp)
                x_temp = layers.BatchNormalization()(x_temp)
                x_temp = layers.Activation(activation)(x_temp)
            x = layers.Add()([x, x_temp])
        if index == 0:
            x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(1, 3, strides=2, activation=activation, padding='same')(x)
    # x = layers.Reshape((image_size, image_size, 64))(x)
    # x = layers.Conv3D(64, (input_frames, 1, 1), activation='tanh')(x)
    # x = layers.Conv2DTranspose(64 * (2 ** len(structure)), 2)(x)
    # # if inception:
    # #     x = inception_cell_revive(x, channels=64 * (2 ** (len(structure)+1)))
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation(activation)(x)
    # x = layers.UpSampling2D(3)(x)
    # x = layers.Conv2DTranspose(64 * (2 ** len(structure)-1), 4)(x)
    # # if inception:
    # #     x = inception_cell_revive(x, channels=64 * (2 ** (len(structure))))
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation(activation)(x)
    # x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2DTranspose(64 * (2 ** (len(structure)-2)), 2)(x)
    # # if inception:
    # #     x = inception_cell_revive(x, channels=64 * (2 ** (len(structure)-1)))
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation(activation)(x)
    # x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2DTranspose(64 * (2 ** (len(structure)-3)), 4)(x)
    # # if inception:
    # #     x = inception_cell_revive(x, channels=64 * (2 ** (len(structure)-2)))
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation(activation)(x)
    # x = layers.UpSampling2D(2)(x)
    # x = layers.Conv2D(64, 3)(x)
    # # if inception:
    # #     x = inception_cell_revive(x, channels=64)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation(activation)(x)
    # x = layers.Conv2D(1, 1)(x)
    # x = layers.Activation(activations.sigmoid)(x)
    model = Model(input_layer, x)
    # model.compile(optimizer=optimizer, loss=loss, metrics=[
    #     losses.binary_crossentropy, losses.mean_squared_logarithmic_error
    # ])
    return model


def create_inception_network(activation, optimizer, loss, input_frames, image_size=64, channels=3, encode_size=2,
                             allow_upsampling=True, allow_pooling=True, kernel_size=3, max_transpose_layers=3,
                             dropout_rate=0.2, inception=True, simple=False):
    initializer = initializers.HeNormal()
    model = models.Sequential()
    current_axis_size = image_size
    target_axis_size = encode_size
    current_frames = input_frames
    model.add(layers.Conv3D(32, 1, activation=activation, kernel_initializer=initializer,
                            input_shape=(input_frames, image_size, image_size, channels)
                            ))
    model.add(layers.Conv3D(32, (input_frames, 1, 1), activation=activation, kernel_initializer=initializer))
    current_frames -= input_frames - 1
    model.add(layers.Reshape((current_axis_size, current_axis_size, 32)))
    laps = 1
    while current_axis_size > target_axis_size:
        # Encoding image
        if current_axis_size - kernel_size + 1 > target_axis_size:
            model.add(layers.Conv2D(32 * 2**laps, kernel_size, activation=activation, kernel_initializer=initializer))
            current_axis_size -= kernel_size - 1
        else:
            model.add(layers.Conv2D(32 * 2**laps, 2, activation=activation, kernel_initializer=initializer))
            current_axis_size -= 1
        if np.floor(current_axis_size / 2) > target_axis_size:
            model.add(layers.MaxPool2D(2))
            current_axis_size = int(np.floor(current_axis_size / 2))
        if inception:
            model = inception_cell(model, activation=activation, axis=3, initializer=initializer,
                                   channels=32 * 2**(laps-2))
        if laps < 2:
            laps += 1
    if not simple:
        max_leaps = max_transpose_layers
        leap_correction = 0
        calculated = False
        first_run = True
        boopage = True
        while current_axis_size < image_size:
            if laps < 0:
                boopage = False
                laps = 0
            if current_axis_size * 2 < image_size and allow_upsampling and not first_run:
                # Upsampling
                model.add(layers.UpSampling2D(2))
                current_axis_size = current_axis_size * 2
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
                model.add(layers.Conv2DTranspose(32 * 2**laps, 2, activation=activation, kernel_initializer=initializer))
                current_axis_size += 1
            elif current_axis_size + kernel_size - 1 + leap_correction > image_size or boopage:
                # Within a few jumps of full size
                model.add(layers.Conv2DTranspose(
                    32 * 2**laps, kernel_size, activation=activation, kernel_initializer=initializer
                ))
                current_axis_size += kernel_size - 1
            else:
                # Full size far away but too close for upsampling
                model.add(layers.Conv2DTranspose(
                    32 * 2**laps, kernel_size + leap_correction, activation=activation, kernel_initializer=initializer
                ))
                current_axis_size += kernel_size - 1 + leap_correction
            laps -= 1
            first_run = False
        model.add(layers.Conv2DTranspose(1, 1, activation='sigmoid', kernel_initializer=initializer))
    else:
        model.add(layers.Flatten())
        current_axis_size = (current_axis_size**2) * 32 * 2**laps
        model.add(layers.Dense(np.sqrt(current_axis_size), activation=activation, kernel_initializer=initializer))
        model.add(layers.Dense(4, activation='sigmoid', kernel_initializer=initializer))
    model.compile(optimizer=optimizer, loss=loss, metrics=[
        losses.binary_crossentropy, losses.mean_squared_logarithmic_error
    ])
    return model


def create_parallel_network(activation, input_frames, image_size, channels=3, encode_size=2,
                             allow_upsampling=True, allow_pooling=True, kernel_size=3, max_transpose_layers=3,
                             dropout_rate=0.2, inception=True):
    input_layer = layers.Input(shape=(input_frames, image_size, image_size, channels))
    streams = []
    final_axis = 1
    for i in range(input_frames):
        x = layers.Lambda(lambda l: l[:, i, :, :, :])(input_layer)
        current_axis = image_size
        while current_axis/2 >= encode_size:
            x = layers.Conv2D(32, kernel_size, padding='same', activation=activation)(x)
            # x = layers.Conv2D(32, kernel_size, padding='same', activation=activation)(x)
            if inception:
                x = inception_cell_revive(x, channels)
            x = layers.MaxPooling2D(2)(x)
            current_axis = int(np.floor(current_axis/2))
            final_axis = current_axis
        streams.append(x)
    x = layers.concatenate(streams, axis=3)
    x = layers.Conv2D(32 * input_frames, 1, activation=activation)(x)
    while final_axis < image_size:
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2DTranspose(64, kernel_size, padding='same', activation=activation)(x)
        # x = layers.Conv2DTranspose(64, kernel_size, padding='same', activation=activation)(x)
        final_axis *= 2
    x = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    model = Model(input_layer, x)
    # model.compile(optimizer=optimizer, loss=loss, metrics=[
    #     losses.binary_crossentropy, losses.mean_squared_logarithmic_error
    # ])
    return model


"""
SAMS CODE
"""


def residual_cell(x, activation, initialiser_w, initialiser_b, layer_size=2, size=10):
    x_skip = x
    for i in range(layer_size):
        #x = layers.Dense(size, activation=activation)(x)
        x = layers.Dense(size, activation=activation, kernel_initializer=initialiser_w, bias_initializer=initialiser_b)(x)
    input_size = x_skip.get_shape()
    #x = layers.Dense(input_size, activation=activation)(x)
    #x = ScaleLayer()(x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)
    return x


def dense_network(input_number, frames, points, activation, optimiser, input_nodes, nodes, layer_num, cell_count, initialiser_w, initialiser_b, loss_func):
    # tf.compat.v1.keras.backend.clear_session()
    x_input = layers.Input(shape=(frames, input_number, points), name="message_input")
    x = layers.Flatten()(x_input)
    # x = layers.Dense(nodes, activation=activation)(x)

    residual_cells = [layer_num, nodes]
    x = layers.Dense(nodes, activation=activations.linear, kernel_initializer=initialiser_w, bias_initializer=initialiser_b)(x)
    #x = layers.Dense(nodes, activation=activations.linear)(x)
    for i in range(cell_count):
        # x = layers.Dropout(0.05)(x)
        x = residual_cell(x, activation, initialiser_w, initialiser_b, layer_size=residual_cells[0], size=residual_cells[1])

    #x = layers.Dense(200, activation=activations.linear)(x)
    x = layers.Dense(200, activation=activations.linear, kernel_initializer=initialiser_w, bias_initializer=initialiser_b)(x)
    x = layers.Reshape((100, 2))(x)
    model = Model(x_input, x)
    # model.compile(optimizer=optimiser)
    # print(model.summary())
    # model = CustomModel(model)
    # model.compile(optimizer=optimiser, loss=loss_func , run_eagerly=False)
    return model


"""
/SAMS CODE
"""


def create_sam_discriminator(input_frames):
    input_layer = Input(shape=(input_frames, 100, 2))
    # x = layers.Conv2D(32, 3, strides=2, padding='same')(input_layer)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dropout(0.3)(x)
    #
    # x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dropout(0.3)(x)
    #
    # x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(input_layer)
    x = layers.Dense(1000, activation='sigmoid')(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(input_layer, x, name='discriminator')
    return model


def create_dumb_network(input_frames, image_size=64, channels=3):
    input_layer = layers.Input(shape=(input_frames, image_size, image_size, channels))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(10)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=activations.swish)(x)
    x = layers.Dense(100)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=activations.swish)(x)
    x = layers.Dense(100)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=activations.swish)(x)
    x = layers.Dense(4096)(x)
    x = layers.Reshape((64, 64, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=activations.sigmoid)(x)
    model = Model(input_layer, x, name='dumb')
    return model


def create_u_network(activation, input_frames,
                     image_size=64, channels=3, encode_size=2, kernel_size=3, inception=False, first_channels=32):
    input_layer = layers.Input(shape=(input_frames, image_size, image_size, channels))
    saving_layers = []
    x = layers.Conv3D(first_channels*2, (input_frames, 1, 1), use_bias=False)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=activations.swish)(x)
    x = layers.Reshape((image_size, image_size, first_channels*2))(x)
    current_axis = image_size
    loops = 0
    while current_axis > encode_size:
        loops += 1
        x = layers.Conv2D(first_channels*2**loops, kernel_size, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation=activations.swish)(x)
        # x = layers.Conv2D(32, kernel_size, padding='same', activation=activation)(x)
        if inception:
            x = inception_cell_revive(x, channels)
        # x = layers.MaxPooling2D(2)(x)
        current_axis = int(np.floor(current_axis / 2))
        saving_layers.append(x)
    x = layers.Conv2D(first_channels*2**loops, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=activations.swish)(x)

    for loop in range(loops):
        x = layers.concatenate([saving_layers[loops - loop - 1], x], axis=3)
        # x = layers.Add()([saving_layers[loops - loop - 1], x])
        # x = layers.UpSampling2D(2)(x)
        x = layers.Conv2DTranspose(first_channels*2**(loops - loop - 1), kernel_size, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation=activations.swish)(x)
    x = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    # x = layers.Activation('sigmoid', dtype='float32')(x)
    # x = layers.Conv2D(1, 1, padding='same', activation='relu')(x)
    model = Model(input_layer, x, name='u-net')
    # model.compile(optimizer=optimizer, loss=loss, metrics=[
    #     losses.binary_crossentropy, losses.mean_squared_logarithmic_error
    # ])
    return model


def sampling(args, latent_dim=6):
    z_mean, z_log_sigma = args
    epsilon = k.random_normal(shape=(k.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + k.exp(z_log_sigma) * epsilon


def create_autoencoder(optimizer, loss, latent_dim=2, image_size=64, image_frames=1, channels=3):
    encoder_inputs = Input(shape=(image_frames, image_size, image_size, 1))
    x = layers.Conv3D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    # x = layers.Reshape((image_size//2, image_size//2, 32))(x)
    # x = inception_cell_revive(x, 1)
    x = layers.Conv3D(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = inception_cell_revive(x, 1)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')
    x = layers.Dense(8 * 8 * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((1, 8, 8, 128))(x)
    x = layers.Conv3DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    # x = inception_cell_revive(x, 1)
    x = layers.Conv3DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    # x = inception_cell_revive(x, 1)
    x = layers.Conv3DTranspose(32, 3, activation="relu", strides=(1, 2, 2), padding="same")(x)
    # x = inception_cell_revive(x, 1)
    x = layers.Conv3DTranspose(1, 3, activation="sigmoid", padding="same")(x)

    decoder = Model(latent_inputs, x, name="decoder")

    autoencoder = Model(encoder_inputs, decoder(encoder(encoder_inputs)[2]), name='VAE')

    reconstruction_loss = losses.binary_crossentropy(encoder_inputs, decoder(encoder(encoder_inputs)[2]))
    reconstruction_loss *= image_size * image_size * image_frames
    reconstruction_loss = k.mean(reconstruction_loss)
    kl_loss = 1 + z_log_var - k.square(z_mean) - k.exp(z_log_var)
    kl_loss = k.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = k.mean(reconstruction_loss + kl_loss)
    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer=optimizer)
    return encoder, decoder, autoencoder


def create_generator():
    input_layer = Input(shape=(100, ))
    x = layers.Dense(4 * 4 * 4 * 256, use_bias=False)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((4, 4, 4, 256))(x)

    x = layers.Conv3DTranspose(128, 3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv3DTranspose(64, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv3DTranspose(32, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv3DTranspose(1, 3, strides=2, padding='same', use_bias=False, activation='sigmoid')(x)

    model = Model(input_layer, x, name='generator')
    return model


def create_discriminator(input_frames, input_size):
    input_layer = Input(shape=(input_frames, input_size, input_size, 1))
    x = layers.Conv3D(32, 3, strides=2, padding='same')(input_layer)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(64, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv3D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    x = layers.Activation('sigmoid', dtype='float32')(x)

    model = Model(input_layer, x, name='discriminator')
    return model


def create_special_discriminator(input_size):
    input_layer = Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(32, 3, strides=2, padding='same')(input_layer)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(input_layer, x, name='discriminator')
    return model




# class ClearMemory(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         gc.collect()
#         k.clear_session()


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
