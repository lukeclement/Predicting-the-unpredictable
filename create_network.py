import numpy as np
from keras.engine import data_adapter
from tensorflow.keras import layers, models, Model, initializers, activations, losses, metrics, backend
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


class CustomModel(Model):
    custom_steps = 2

    mse = metrics.MeanSquaredError(name="MSE")
    loss_tracker = metrics.Mean(name="loss")
    loss_tracker_live = metrics.Mean(name="loss_live")
    bce_metric = metrics.BinaryCrossentropy(name="BCE")
    bce_metric_live = metrics.BinaryCrossentropy(name="BCE_live")

    def custom_loss(self, y_true, y_pred):
        # se = (y_pred - y_true) * (y_pred - y_true)
        # mse = backend.mean(se)
        loss = losses.binary_crossentropy(y_true, y_pred)
        return loss

    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.input_model = model

    def call(self, inputs, training=None, mask=None):
        return self.input_model(inputs)

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        xold = x
        rail = x[:, 0, :, :, 2:]
        zeros = x[:, 0, :, :, 0:1]
        lf = self.custom_loss
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_true = y[:, 0, :, :, :]
            loss = lf(y_true, y_pred)
            if self.custom_steps > 1:
                for i in range(0, self.custom_steps - 1):
                    y_pred = tf.concat([zeros, y_pred], 3)
                    y_pred = tf.concat([y_pred, rail], 3)
                    y_pred_shape = tf.shape(y_pred)
                    y_pred = tf.reshape(y_pred, shape=(y_pred_shape[0], 1, y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]))
                    x = x[:, 1:, :, :, :]
                    x = tf.concat([x, y_pred], 1)
                    y_pred = self(x, training=True)
                    y_true = y[:, i + 1, :, :, :]
                    loss = loss + lf(y_true, y_pred)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        y_pred = self(xold, training=False)
        self.loss_tracker_live.reset_state()
        self.bce_metric_live.reset_state()
        self.loss_tracker.update_state(loss)
        self.loss_tracker_live.update_state(loss)
        self.bce_metric.update_state(y, y_pred)
        self.mse.update_state(y, y_pred)
        self.bce_metric_live.update_state(y, y_pred)

        loss_result = self.loss_tracker.result()
        bce_result = self.bce_metric.result()
        mse_result = self.mse.result()
        loss_result_live = self.loss_tracker_live.result()
        bce_result_live = self.bce_metric_live.result()
        return {"Average loss": loss_result, "Average BCE": bce_result, "Live loss": loss_result_live,
                "Live BCE": bce_result_live, "MSE": mse_result}

    def test_step(self, data):
        self.loss_tracker.reset_state()
        self.bce_metric.reset_state()
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        xold = x
        rail = x[:, 0, :, :, 2:]
        zeros = x[:, 0, :, :, 0:1]
        lf = self.custom_loss
        y_pred = self(x, training=False)
        y_true = y[:, 0, :, :, :]
        loss = lf(y_true, y_pred)
        if self.custom_steps > 1:
            for i in range(0, self.custom_steps - 1):
                y_pred = tf.concat([zeros, y_pred], 3)
                y_pred = tf.concat([y_pred, rail], 3)
                y_pred_shape = tf.shape(y_pred)
                y_pred = tf.reshape(y_pred, shape=(y_pred_shape[0], 1, y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]))
                x = x[:, 1:, :, :, :]
                x = tf.concat([x, y_pred], 1)
                y_pred = self(x, training=False)
                y_true = y[:, i + 1, :, :, :]
                loss = loss + lf(y_true, y_pred)

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

    tower_1 = layers.Conv2D(32, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)

    tower_2 = layers.Conv2D(32, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)
    tower_2 = layers.Conv2D(32, (3, 3), padding='same', activation=activation, kernel_initializer=initializer)(tower_2)

    tower_3 = layers.Conv2D(32, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)
    tower_3 = layers.Conv2D(32, (5, 5), padding='same', activation=activation, kernel_initializer=initializer)(tower_3)

    # tower_4 = layers.MaxPooling2D((3, 3), strides=1)(input_tower)
    tower_4 = layers.Conv2D(32, (3, 3), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)

    model.add(Model(input_tower, merged))

    return model


def create_inception_net(activation, optimizer, frames=4, size=60, channels=3):
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

    model.add(layers.Conv3D(32, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))

    model.add(layers.Conv2D(32, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer)

    model.add(layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer)

    model.add(layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer)

    model.add(layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(32, (3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer)

    model.add(layers.Conv2DTranspose(32, (3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer)

    model.add(layers.Conv2DTranspose(32, (3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer)

    model.add(layers.Conv2DTranspose(32, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(32, (7, 7), activation=activations.sigmoid, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activation, kernel_initializer=initializer))

    print(model.summary())

    model = CustomModel(model)
    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer)

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
    X = training_images[0]
    Y = training_images[1]
    history = model.fit(X[0], X[1], validation_split=validation_split, epochs=epochs, shuffle=True)

    # history = model.fit(training_images[0], epochs=epochs, shuffle=True)

    return model, history
