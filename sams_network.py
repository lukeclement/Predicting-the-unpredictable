import time

import tensorflow as tf
from tensorflow.keras import layers, Model, losses, metrics, optimizers, activations, backend
from keras.engine import data_adapter
import glob
import os
from tqdm import tqdm
import numpy as np


def residual_cell(x, activation, layer_size=2, size=10):
    x_skip = x
    for i in range(layer_size):
        x = layers.Dense(size, activation=activation)(x)
    input_size = x_skip.get_shape()[1]
    x = layers.Dense(input_size, activation=activation)(x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)
    return x


def message_model(frames, input_number, output_number, activation, residual_cells=None):
    if residual_cells is None:
        residual_cells = [[2, 10], [2, 10], [2, 10]]
    x_input = layers.Input(shape=(frames, input_number), name="message_input")
    x = layers.Flatten()(x_input)
    for cell_struct in residual_cells:
        x = residual_cell(x, activation, layer_size=cell_struct[0], size=cell_struct[1])
    x = layers.Dense(output_number, activation=activation)(x)
    model = Model(x_input, x)
    return model


def interpretation_model(input_number, output_number, activation, residual_cells=None):
    if residual_cells is None:
        residual_cells = [[2, 20], [2, 20], [2, 20]]
    x_input = layers.Input(shape=input_number, name="interpretation_input")
    x = layers.Dense(30, activation=activation)(x_input)
    for cell_struct in residual_cells:
        x = residual_cell(x, activation, layer_size=cell_struct[0], size=cell_struct[1])
    x = layers.Dense(output_number, activation=activations.linear)(x)
    model = Model(x_input, x)
    return model


@tf.function
def train_step(x, y, messenger, interpreter, m_op, i_op):
    # x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    loss_total = 0
    with tf.GradientTape() as message_tape, tf.GradientTape() as inter_tape:
        index = 0
        for single in x:
            col_index = 0
            messages = messenger(single, training=True)
            for col in single:
                message_sum = tf.concat([messages[:col_index], messages[col_index:]], 0)
                message_sum = backend.sum(message_sum, axis=0)
                message_sum = tf.concat([message_sum, tf.cast(col[-1], tf.float32)], 0)
                message_sum = tf.expand_dims(message_sum, 0)
                y_pred = interpreter(message_sum, training=True)[0]
                y_true = tf.cast(y[index, col_index], tf.float32)
                print(y_pred)
                print(y_true)
                loss_total += losses.mean_squared_error(y_true, y_pred)
                col_index += 1
            index += 1
    message_grad = message_tape.gradient(loss_total, messenger.trainable_variables)
    interpreter_grad = inter_tape.gradient(loss_total, interpreter.trainable_variables)
    m_op.apply_gradients(zip(message_grad, messenger.trainable_variables))
    i_op.apply_gradients(zip(interpreter_grad, interpreter.trainable_variables))

    return loss_total


def train_network(messenger, interpreter, m_op, i_op, epochs, dataset):
    times_so_far = []
    for epoch in range(epochs):
        start = time.time()
        print("Running epoch {}...".format(epoch + 1))
        for questions, answers in dataset:
            loss = train_step(questions, answers, messenger, interpreter, m_op, i_op)
        times_so_far.append(time.time() - start)
        seconds_per_epoch = times_so_far[epoch]
        if seconds_per_epoch / 60 > 1:
            print("Time for epoch {} was {:.0f}min, {:.0f}s".format(
                epoch + 1, seconds_per_epoch // 60, seconds_per_epoch - (seconds_per_epoch // 60) * 60)
            )
        else:
            print("Time for epoch {} was {:.0f}s".format(epoch + 1, seconds_per_epoch))


def save_border_data(point_num, simulation):
    data_names = glob.glob("Simulation_data_extrapolated/Simulation_False_0_0.001_{}/*".format(str(simulation)))
    folder_length = len(data_names)
    try:
        os.mkdir("training_data/Simulation_{}_points_{}/".format(simulation, point_num))
    except OSError:
        print("Folder already exists!")
    pbar = tqdm(total=folder_length - 3)
    for i in range(3, folder_length):
        pbar.update()
        final_data = border_data(point_num, simulation, i)
        np.save("training_data/Simulation_{}_points_{}/data_{}".format(simulation, point_num, i), final_data)
    pbar.close()


def redefine_index(arrays, index):
    arr_length = len(arrays[0])
    new_arrays = [np.zeros(arr_length), np.zeros(arr_length)]
    for i in range(arr_length - 1):
        prime_index = index + i
        new_arrays[0][i] = arrays[0][prime_index]
        new_arrays[1][i] = arrays[1][prime_index]
        if index + i == arr_length - 1:
            index = -i
    return new_arrays


def circ_array(arrays):
    arr_length = len(arrays[0])
    new_array = np.zeros(arr_length)
    length = 0
    for i in range(arr_length - 1):
        x_1 = arrays[0][i]
        y_1 = arrays[1][i]
        x_2 = arrays[0][i + 1]
        y_2 = arrays[1][i + 1]
        length += ((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2) ** 0.5
        new_array[i + 1] = length
    return new_array


def circumference(file):
    length = 0
    for i in range(len(file[0]) - 1):
        x_1 = file[0][i]
        y_1 = file[1][i]
        x_2 = file[0][i + 1]
        y_2 = file[1][i + 1]
        length += ((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2) ** 0.5
    return length


def find_zero(x_array, y_array):
    x_copy = np.array(x_array)
    best_idx = 0
    for i in range(20):
        idx = (np.abs(x_copy - 0)).argmin()
        if y_array[idx] > y_array[best_idx]:
            best_idx = idx
        x_copy[idx] = 999
    return best_idx


def border_data(point_num, simulation_num, file_num):
    # data_names = glob.glob("Simulation_data_extrapolated/Simulation_False_0_0.0001_0/*")
    file = np.load("Simulation_data_extrapolated/Simulation_False_0_0.001_{}/data_{}.npy".format(
        simulation_num, file_num
    ))
    final_array = [np.zeros(point_num), np.zeros(point_num)]
    idx = find_zero(file[1], file[0])
    circ = circumference(file)
    length = circ / point_num
    data = redefine_index(file, idx)
    # data[1][0] = 0
    circ_data = circ_array(data)
    for i in range(point_num):
        idx = (np.abs(circ_data - length * i)).argmin()
        final_array[0][i] = data[0][idx]
        final_array[1][i] = data[1][idx]

    if final_array[1][0]-final_array[1][1] > 0:
        final_array = np.flip(final_array, axis=1)
        final_array = np.insert(final_array, 0, final_array[:,-1], axis=1)
        final_array = np.delete(final_array, -1, axis=1)

    return final_array


def load_data(points, frames, time_step, scaling):
    training_data = []
    labels = []
    pbar = tqdm(total=1)
    for simulation in range(1):
        pbar.update()
        data_names = glob.glob("training_data/Simulation_{}_points_{}/*".format(simulation, points))
        folder_length = len(data_names)
        print(folder_length)
        data_array = []
        for file_num in range(3, folder_length + 1):
            data = np.load("training_data/Simulation_{}_points_{}/data_{}.npy".format(simulation, points, file_num))
            data_array.append(data)
        xy_data = []
        for data in data_array:
            xy_array = []
            for i in range(points):
                xy = []
                for j in range(2):
                    xy.append(data[j][i])
                xy_array.append(xy)
            xy_data.append(xy_array)
        vel_data = velocity_calculation(xy_data, scaling, time_step)
        for i in range(0, len(xy_data) - frames*time_step - time_step):
            single = []
            for j in range(0, frames):
                row = np.array(xy_data[i + j*time_step + time_step])
                row = np.append(row, vel_data[i+j*time_step], axis=1)
                single.append(row)
                # single.append(vel_data[i + j])
            training_data.append(single)
            labels.append(vel_data[i + frames*time_step])
    pbar.close()
    return [np.array(training_data), np.array(labels)]


def velocity_calculation(data, scaling, time_step):
    data = np.array(data)
    velocity = [data[time_step] - data[0]]
    for i in range(1, len(data)-time_step):
        velocity.append((data[i+time_step]-data[i]))
    return np.array(velocity)*scaling


def main():
    print("Working")
    # for i in range(0, 16):
    #     save_border_data(100, i)
    activation = activations.tanh
    optimizer_m = optimizers.Adam(learning_rate=0.001)
    optimizer_i = optimizers.Adam(learning_rate=0.001)
    frames = 2
    points = 100
    kernel_size = 32
    scaling = 100
    messenger = message_model(frames, 4, 4, activation)
    interpreter = interpretation_model(4+4, 2, activation)
    print(messenger.summary())
    print(interpreter.summary())
    data = load_data(points, frames, 5, scaling)

    data[0] = np.transpose(data[0], axes=(0, 2, 1, 3))
    print(np.shape(data[0]))
    print(np.shape(data[1]))

    datas = tf.data.Dataset.from_tensor_slices((data[0][:], data[1][:])).shuffle(buffer_size=100000).batch(32)
    train_network(messenger, interpreter, optimizer_m, optimizer_i, 5, datas)


if __name__ == "__main__":
    main()
