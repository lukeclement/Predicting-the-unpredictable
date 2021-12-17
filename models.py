import numpy as np
from keras.engine import data_adapter
from tensorflow.keras import layers, models, Model, initializers, activations, losses, metrics, backend, optimizers
import tensorflow as tf


class CustomModel_BCE(Model):
    mse = metrics.MeanSquaredError(name="MSE")
    loss_tracker = metrics.Mean(name="loss")
    loss_tracker_live = metrics.Mean(name="loss_live")
    bce_metric = metrics.BinaryCrossentropy(name="BCE")
    bce_metric_live = metrics.BinaryCrossentropy(name="BCE_live")

    loss_choice = 1
    summary_v = ""

    def custom_loss(self, y_true, y_pred):
        if self.loss_choice == 1:
            loss = losses.binary_crossentropy(y_true, y_pred)
        else:
            loss = losses.mean_squared_error(y_true, y_pred)
        return loss

    def __init__(self, model):
        super(CustomModel_BCE, self).__init__()
        self.input_model = model

    def call(self, inputs, training=None, mask=None):
        return self.input_model(inputs)

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        xold = x
        lf = self.custom_loss
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = lf(y, y_pred)

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
        lf = self.custom_loss
        y_pred = self(x, training=False)
        y_true = y
        loss = lf(y_true, y_pred)
        y_pred = self(xold, training=False)
        self.loss_tracker.update_state(loss)
        self.bce_metric.update_state(y, y_pred)
        loss_result = self.loss_tracker.result()
        bce_result = self.bce_metric.result()
        self.loss_tracker.reset_state()
        self.bce_metric.reset_state()
        return {"loss": loss_result, "BCE": bce_result}


def inception_cell(model, activation, axis, initializer, kernal_size):
    shape = model.output_shape
    li = list(shape)
    li.pop(0)
    shape = tuple(li)
    input_tower = layers.Input(shape=shape)

    tower_1 = layers.Conv2D(kernal_size, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)

    tower_2 = layers.Conv2D(kernal_size, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)
    tower_2 = layers.Conv2D(kernal_size, (3, 3), padding='same', activation=activation, kernel_initializer=initializer)(
        tower_2)

    tower_3 = layers.Conv2D(kernal_size, (1, 1), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)
    tower_3 = layers.Conv2D(kernal_size, (5, 5), padding='same', activation=activation, kernel_initializer=initializer)(
        tower_3)

    # tower_4 = layers.MaxPooling2D((3, 3), strides=1)(input_tower)
    tower_4 = layers.Conv2D(kernal_size, (3, 3), padding='same', activation=activation, kernel_initializer=initializer)(
        input_tower)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)

    model.add(Model(input_tower, merged))
    return model


def model_1(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    loss_function = losses
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(kernal_size, (4, 4), activation=activation, kernel_initializer=initializer))

    model.add(layers.UpSampling2D((3, 3)))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(), metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_2(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    loss_function = losses
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(kernal_size, (4, 4), activation=activation, kernel_initializer=initializer))

    model.add(layers.UpSampling2D((3, 3)))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_3(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    loss_function = losses
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(kernal_size, (4, 4), activation=activation, kernel_initializer=initializer))

    model.add(layers.UpSampling2D((3, 3)))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_4(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    loss_function = losses
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(kernal_size, (4, 4), activation=activation, kernel_initializer=initializer))

    model.add(layers.UpSampling2D((3, 3)))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_5(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    loss_function = losses
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (4, 4), activation=activation, kernel_initializer=initializer))

    model.add(layers.UpSampling2D((3, 3)))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_6(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    loss_function = losses
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (4, 4), activation=activation, kernel_initializer=initializer))

    model.add(layers.UpSampling2D((3, 3)))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_7(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    loss_function = losses
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (4, 4), activation=activation, kernel_initializer=initializer))

    model.add(layers.UpSampling2D((3, 3)))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_8(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    loss_function = losses
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 7, 7), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((54, 54, 32)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=3, initializer=initializer, kernal_size=kernal_size)
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2DTranspose(kernal_size, (4, 4), activation=activation, kernel_initializer=initializer))

    model.add(layers.UpSampling2D((3, 3)))
    model.add(layers.Conv2DTranspose(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_9(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 3, 3), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((58, 58, kernal_size)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=1, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=2, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_10(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 3, 3), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((58, 58, kernal_size)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=1, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=2, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=1, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=2, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)
    return model


def model_11(activation, optimizer, type=1, frames=4, size=60, kernal_size=32):
    initializer = initializers.HeNormal()
    model = models.Sequential()

    model.add(layers.Conv3D(kernal_size, (4, 3, 3), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, size, 3)))
    model.add(layers.Reshape((58, 58, kernal_size)))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=1, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=2, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=1, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=2, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(kernal_size, (3, 3), strides=(3, 3), activation=activation, kernel_initializer=initializer))

    model = inception_cell(model, activation=activation, axis=1, initializer=initializer, kernal_size=kernal_size)
    model = inception_cell(model, activation=activation, axis=2, initializer=initializer, kernal_size=kernal_size)

    model.add(layers.Conv2D(kernal_size, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv2D(kernal_size, (5, 5), activation=activation, kernel_initializer=initializer))

    model.add(layers.Conv2D(1, (3, 3), activation=activations.sigmoid, kernel_initializer=initializer))

    summary = model.summary()
    if type == 1:
        model = CustomModel_BCE(model)
        model.summary_v = summary
        print(model.summary_v)
        model.compile(optimizer=optimizer)
    else:
        model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy(),
                      metrics=[metrics.MeanSquaredError(name="MSE")])
        print(summary)

    return model


def model_array(activation_function, optimizer):
    model_arr = []
    model_arr.append(model_1(activation_function, optimizer, type=2))
    model_arr.append(model_2(activation_function, optimizer, type=2))
    model_arr.append(model_3(activation_function, optimizer, type=2))
    model_arr.append(model_4(activation_function, optimizer, type=2))
    model_arr.append(model_5(activation_function, optimizer, type=2))
    model_arr.append(model_6(activation_function, optimizer, type=2))
    model_arr.append(model_7(activation_function, optimizer, type=2))
    model_arr.append(model_8(activation_function, optimizer, type=2))
    model_arr.append(model_9(activation_function, optimizer, type=2))
    model_arr.append(model_10(activation_function, optimizer, type=2))
    model_arr.append(model_11(activation_function, optimizer, type=2))

    return model_arr


def train_model(training_data, activation_function, optimizer, epochs=1):
    model = model_1(activation_function, optimizer, type=1)
    X = training_data[0]
    history = model.fit(X[0], X[1], validation_split=0.01, epochs=epochs, shuffle=True)
    history = model.fit(X[0], X[1], validation_split=0.01, epochs=epochs, shuffle=True)
    model.loss_choice = 2
    model.compile()
    history = model.fit(X[0], X[1], validation_split=0.01, epochs=epochs, shuffle=True)
    return model, history

def train_model_1(training_data, model, epochs=1):
    X = training_data[0]
    labels = X[1]
    labels = labels[:, 0, :, :]
    history = model.fit(X[0], labels, validation_split=0.01, epochs=epochs, shuffle=True)
    return model, history


def main():
    activation_function = layers.LeakyReLU()
    optimizer = optimizers.Adam()
    # model = model_1(activation_function, optimizer)
    # model = model_2(activation_function, optimizer)
    # model = model_3(activation_function, optimizer)
    # model = model_4(activation_function, optimizer)
    model = model_11(activation_function, optimizer)


if __name__ == "__main__":
    main()
