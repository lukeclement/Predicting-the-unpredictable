import numpy as np
import loss_functions
import dat_to_training
from keras.engine import data_adapter
from tensorflow.keras import layers, models, Model, initializers, activations, losses, metrics
import tensorflow as tf


def metric_differences(positive_guess, positive_real):
    positive_correct = tf.zeros((64, 64, 1), tf.float16)
    positive_correct[(positive_guess == positive_real) & (positive_real == 0)] = 0  # Background
    positive_correct[(positive_guess == positive_real) & (positive_real == 1)] = 2  # Correct Prediction (Green)
    positive_correct[(positive_guess != positive_real) & (positive_real == 0)] = 1  # Over prediction (Red)
    positive_correct[(positive_guess != positive_real) & (positive_real == 1)] = 3  # Under prediction (Blue)
    unique, counts = tf.unique_with_counts(positive_correct)
    positive_counts = dict(zip(unique, counts))
    return positive_correct, positive_counts


def pixel_prediction(input_data):
    def update_state(y_true, y_pred):
        rounded_guess = tf.math.round(y_pred)

        real_difference = y_true - input_data
        guess_difference = rounded_guess - input_data

        positive_real = tf.zeros((64, 64, 1), tf.float16)
        positive_real[real_difference > 0] = 1
        positive_guess = tf.zeros((64, 64, 1), tf.float16)
        positive_guess[guess_difference > 0] = 1
        positive_correct, positive_counts = metric_differences(positive_guess, positive_real)

        negative_real = np.zeros((64, 64, 1))
        negative_real[real_difference < 0] = 1
        negative_guess = np.zeros((64, 64, 1))
        negative_guess[guess_difference < 0] = 1
        negative_correct, negative_counts = metric_differences(negative_guess, negative_guess)
        return {**positive_counts, **negative_counts}
    return update_state




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
    model.add(layers.Conv3D(3, 1, activation=activation, input_shape=(input_frames, image_size, image_size, channels)))
    # Encoding the image
    while current_axis_size > target_axis_size:
        if current_frames > 1:
            # Initial 3D convolutional layers to reduce the frames into a single image
            # model.add(layers.Conv3D(64, 2, activation=activation))
            # current_frames -= 1
            # current_axis_size -= 1
            model.add(layers.Conv3D(32, input_frames, activation=activation))
            current_frames -= input_frames-1
            current_axis_size -= input_frames-1
        elif current_frames == 1:
            # Reshaping the image to be the correct dimensions
            current_frames -= 1
            model.add(layers.Reshape((current_axis_size, current_axis_size, 32)))
            model.add(layers.Conv2D(32, kernel_size, activation=activation))
            current_axis_size -= (kernel_size - 1)
        else:
            # Bringing the image down to encoding size
            if np.floor(current_axis_size / 2) > target_axis_size and allow_pooling:
                model.add(layers.MaxPooling2D(2))
                current_axis_size = np.floor(current_axis_size / 2)
            if current_axis_size - (kernel_size - 1) < target_axis_size:
                model.add(layers.Conv2D(64, 2, activation=activation))
                current_axis_size -= 1
            else:
                model.add(layers.Conv2D(64, kernel_size, activation=activation))
                current_axis_size -= (kernel_size - 1)
    # Now decoding the image using transpose operations
    # model.add(layers.Conv2DTranspose(64, kernel_size, activation=activation))
    # current_axis_size += (kernel_size - 1)
    # Some variables to keep track of in the while loop
    max_leaps = max_transpose_layers
    leap_correction = 0
    calculated = False
    while current_axis_size < image_size:
        if current_axis_size * 2 < image_size and allow_upsampling and not first_run:
            # Upsampling
            model.add(layers.UpSampling2D(2))
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
            model.add(layers.Conv2DTranspose(32, 2, activation=activation))
            current_axis_size += 1
        elif current_axis_size + kernel_size - 1 + leap_correction > image_size:
            # Within a few jumps of full size
            model.add(layers.Conv2DTranspose(64, kernel_size, activation=activation))
            current_axis_size += kernel_size - 1
        else:
            # Full size far away but too close for upsampling
            model.add(layers.Conv2DTranspose(64, kernel_size + leap_correction, activation=activation))
            current_axis_size += kernel_size - 1 + leap_correction
    # Final adjustments
    model.add(layers.Conv2DTranspose(1, 1, activation='sigmoid'))
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss)
    return model


class CustomModel(Model):
    custom_steps = 2
    loss_tracker = metrics.Mean(name="loss")
    bce_metric = metrics.BinaryCrossentropy(name="BCE")

    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.input_model = model

    def call(self, inputs, training=None, mask=None):
        y_pred = self.input_model(inputs)
        return tf.reshape(y_pred, shape=(tf.shape(y_pred)[0], 1, 64, 64, 1))

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        xold = x
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = losses.binary_crossentropy(y[:, 0, :, :, :], y_pred)
            x = x[:, 1:, :, :, :]
            x = tf.concat([x, y_pred], 1)
            if self.custom_steps > 1:
                for i in range(0, self.custom_steps-1):
                    y_pred = self(x, training=True)
                    y_true = y[:, i+1, :, :, :]
                    loss = tf.math.add(loss, losses.binary_crossentropy(y_true, y_pred))
                    x = x[:, 1:, :, :, :]
                    x = tf.concat([x, y_pred], 1)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        # Collect metrics to return
        y_pred = self(xold, training=False)
        self.loss_tracker.update_state(loss)
        self.bce_metric.update_state(y[:, 0, :, :, :], y_pred)
        loss_result = self.loss_tracker.result()
        bce_result = self.bce_metric.result()
        return {"loss": loss_result, "BCE": bce_result}

    def test_step(self, data):
        self.loss_tracker.reset_state()
        self.bce_metric.reset_state()
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        xold = x
        y_pred = self(x, training=False)
        loss = losses.binary_crossentropy(y[:, 0, :, :, :], y_pred)
        x = x[:, 1:, :, :]
        x = tf.concat([x, y_pred], 1)
        if self.custom_steps > 1:
            for i in range(0, self.custom_steps - 1):
                y_pred = self(x, training=False)
                y_true = y[:, i + 1, :, :, :]
                loss = tf.math.add(loss, losses.binary_crossentropy(y_true, y_pred))
                x = x[:, 1:, :, :, :]
                x = tf.concat([x, y_pred], 1)

        y_pred = self(xold, training=False)
        self.loss_tracker.update_state(loss)
        self.bce_metric.update_state(y[:, 0, :, :, :], y_pred)
        loss_result = self.loss_tracker.result()
        bce_result = self.bce_metric.result()
        self.loss_tracker.reset_state()
        self.bce_metric.reset_state()
        return {"loss": loss_result, "BCE": bce_result}


def inception_cell(model, activation, axis, initializer):
    shape = model.output_shape
    li = list(shape)
    li.pop(0)
    shape = tuple(li)
    input_tower = layers.Input(shape=shape)

    tower_1 = layers.Conv2D(32, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(input_tower)

    tower_2 = layers.Conv2D(32, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(input_tower)
    tower_2 = layers.Conv2D(32, (3, 3), padding='same', activation=activation, kernel_initializer=initializer)(tower_2)

    tower_3 = layers.Conv2D(32, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(input_tower)
    tower_3 = layers.Conv2D(32, (5, 5), padding='same', activation=activation, kernel_initializer=initializer)(tower_3)

    # tower_4 = layers.MaxPooling2D((3, 3), strides=1)(input_tower)
    tower_4 = layers.Conv2D(32, (3, 3), padding='same', activation=activation, kernel_initializer=initializer)(input_tower)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)
    model.add(Model(input_tower, merged))

    return model


def create_inception_net(activation, optimizer, loss, frames=4, size=64, channels=3):
    """Creates the CNN.
    Inputs:
        activation: The activation function used on the neurons (string)
        optimizer:  The optimisation function used on the model (string)
        loss:       The loss function to be used in training    (function)
        size:       The size of the image, defaults to 128      (int)
    Output:
        The model, ready to be fitted!
    """
    initializer = initializers.HeNormal()

    model = models.Sequential()

    # input_tensor = layers.Input(shape=(frames, size, size))
    # constant = layers.Input(shape=(frames, size, size), tensor=rail)

    # merged_input = layers.concatenate([input_tensor, constant], axis=0)

    # model.add(merged_input)

    model.add(layers.Conv3D(32, (4, 7, 7), kernel_initializer=initializer, activation=activation, input_shape=(frames, size, size, 1)))
    model.add(layers.Dropout(0.05))
    model.add(layers.Reshape((58, 58, 32)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.BatchNormalization())
    model = inception_cell(model, activation=activation, axis=1, initializer=initializer)
    model = inception_cell(model, activation=activation, axis=2, initializer=initializer)

    model.add(layers.Conv2D(32, (1, 1), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(32, (5, 5), activation=activation, kernel_initializer=initializer, strides=(5, 5)))
    model = inception_cell(model, activation=activation, axis=1, initializer=initializer)
    model = inception_cell(model, activation=activation, axis=2, initializer=initializer)
    model.add(layers.Conv2DTranspose(32, (6, 6), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(32, (6, 6), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    model = CustomModel(model)
    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer, loss=loss)

    # model.compile(optimizer=optimizer, loss=loss, metrics=pixel_prediction(i))
    # model.compile(optimizer=optimizer, loss=loss, metrics=[mass_preservation])

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
    # X = training_images[0]
    # Y = training_images[1]
    # history = model.fit(X, Y, validation_split=validation_split, epochs=epochs, shuffle=True)

    history = model.fit(training_images, epochs=epochs, shuffle=True)

    return model, history
