import os
import gc
import pandas as pd
import matplotlib.cm as cm
import scipy.optimize
import data_creation
import subprocess
import numpy as np
import glob
from tqdm import tqdm
from tensorflow.keras import layers, initializers, activations, losses, metrics, optimizers, Model, callbacks, backend
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from datetime import datetime
import pickle


def residual_cell(x, activation, initialiser_w, initialiser_b, layer_size=2, size=10):
    x_skip = x
    for i in range(layer_size):
        x = layers.Dense(size, activation=activation, kernel_initializer=initialiser_w, bias_initializer=initialiser_b)(
            x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)
    return x


def dense_network(input_number, frames, points, activation, optimiser, input_nodes, nodes, layer_num, cell_count,
                  initialiser_w, initialiser_b, loss_func):
    tf.compat.v1.keras.backend.clear_session()
    x_input = layers.Input(shape=(input_number, frames, points), name="message_input")
    x = layers.Flatten()(x_input)

    residual_cells = [layer_num, nodes]
    x = layers.Dense(nodes, activation=activations.linear, kernel_initializer=initialiser_w,
                     bias_initializer=initialiser_b)(x)
    for i in range(cell_count):
        x = residual_cell(x, activation, initialiser_w, initialiser_b, layer_size=residual_cells[0],
                          size=residual_cells[1])

    x = layers.Dense(200, activation=activations.linear, kernel_initializer=initialiser_w,
                     bias_initializer=initialiser_b)(x)
    x = layers.Reshape((100, 2))(x)
    model = Model(x_input, x)
    model.compile(optimizer=optimiser, loss=loss_func, run_eagerly=False)
    return model


def step_decay_1(epoch):
    int_rate = 0.002
    return np.exp(np.log(0.1) / 100) ** epoch * int_rate


def step_decay_2(epoch):
    int_rate = 0.1
    return np.exp(epoch * np.log(0.1) / 500) * (1 + 0.3 * np.sin((2 * 3.14 * epoch) / 50)) * int_rate


def step_decay_3(epoch):
    int_rate = 0.001
    return max(int_rate * 0.1 ** int(epoch / 100), 1e-6)


def step_decay(epoch):
    int_rate = 0.001
    l_rate = max(int_rate * 0.1 ** int(epoch / 20), 1e-6)
    if epoch < 10:
        l_rate = 0.001
    return l_rate


def lr_scheduler():
    return callbacks.LearningRateScheduler(step_decay)


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0.2, guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2. * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f}
    # return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


def plot_data():
    for i in range(15):
        data_mae = creating_data_graph(1, 100, [i], 500, -1)
        x = data_mae[0][-4, :, 0, 0]
        x = np.append(x[-1], x)
        y = data_mae[0][-4, :, 0, 1]
        y = np.append(y[-1], y)
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        plt.axis('off')
        plt.plot(y, x, color='k')
        plt.savefig("last_frame_testing/{}".format(i), transparent=True)
        plt.show()


def unstable_orbit_fitting(model_mae, min_c, max_c, min_c_1, max_c_1):
    data_mae = creating_data_graph(1, 100, [3], 3, 50)
    # prediction[0, :, 0, 0] = prediction[0, :, 0, 0] + 0.017516327542968738
    central = -0.0002
    difference = 0.025
    cl = 0
    # text_file = open("start_average_upper.txt", "r")
    # lines = text_file.readlines()
    # text_file.close()
    # array = np.array(lines[:], dtype=np.float)
    # text_file_lower = open("start_average_lower.txt", "w")
    # text_file_upper = open("start_average_upper.txt", "w")
    standard_average_array = []
    fig, axs = plt.subplots(figsize=[7, 5], dpi=300)
    # axins = inset_axes(axs, 1, 0.5, bbox_to_anchor=[0.4, 0.8], bbox_transform=axs.figure.transFigure)
    gif = False
    upper = 0.0104
    lower = 0.0071
    diffs = 0.00002
    # count = int((upper-lower)/diffs)+1
    # print("count = ", count)
    xy_array = []
    # sims = np.arange(lower, upper, diffs)
    sims = [0, 0.001, 0.0021, 0.0030, 0.0125]
    count = len(sims)
    # upper = 0.01260
    # lower = -0.00149
    upper = 0.02
    lower = -0.02
    diffs = 0.00001
    count = int((upper - lower) / diffs) + 1
    print("count = ", count)
    colors = cm.winter(np.linspace(0, 1, count))
    pbar = tqdm(total=count)
    # for sim in np.arange(lower, upper, diffs):
    for sim in sims:
        pbar.update(1)
        prediction = np.copy(data_mae[0][0:1])
        average = np.average(prediction[0, :, 0, 0])
        # diff = -average+sim
        prediction[0, :, 0, 0] = prediction[0, :, 0, 0] + sim
        start_average = np.average(prediction[0, :, 0, 0])
        # print("average:", start_average, ", sim: ", sim)
        average_array = [start_average]
        prediction = data_normalisation(prediction, min_c, max_c)
        f = 0
        image_array = []
        xy = []
        while True:
            if f > 175:
                break

            pred = model_mae(prediction).numpy()
            corr_x = data_unnormalisation(prediction[0, :, -1, 0], min_c[:, 0], max_c[:, 0])
            corr_y = data_unnormalisation(prediction[0, :, -1, 1], min_c[:, 1], max_c[:, 1])
            diff_x = data_unnormalisation(pred[0, :, 0], min_c_1[:, 0], max_c_1[:, 0])
            diff_y = data_unnormalisation(pred[0, :, 1], min_c_1[:, 1], max_c_1[:, 1])
            data_adjusted = position_transform([corr_x, corr_y], [diff_x, diff_y])
            y_average = np.average(data_adjusted[1])
            if f % 1 == 0 and 1 < f < 1000 and abs(np.average(data_adjusted[0])) < 0.001:
                try:
                    dat = data_creation.make_lines(data_adjusted[0], data_adjusted[1], 0.001)
                except:
                    # print("fail", f)
                    break
                final_array = [np.zeros(100), np.zeros(100)]
                circ = data_creation.circumfrance(dat)
                length = circ / 100
                circ_data = data_creation.circ_array(dat)
                for i in range(100):
                    idx = (np.abs(circ_data - length * i)).argmin()
                    final_array[0][i] = dat[0][idx]
                    final_array[1][i] = dat[1][idx]
                if final_array[0][0] - final_array[0][1] > 0:
                    final_array = np.flip(final_array, axis=1)
                    final_array = np.insert(final_array, 0, final_array[:, -1], axis=1)
                    final_array = np.delete(final_array, -1, axis=1)
            else:
                final_array = data_adjusted
            y_average = np.average(final_array[0])
            xy.append(final_array)
            if 0.41 < y_average or y_average < -0.41:
                break
            if f > 1:
                if abs(average_array[-1] - y_average) > 0.03:
                    break
            if len(average_array) > 60:
                if abs(np.max(average_array[-50:])) < 0.018:
                    break
                # if np.std(average_array[-10:]) < 0.0012:
                #     break
            average_array.append(y_average)
            prediction[0, :, :-1] = prediction[0, :, 1:]
            for i in range(len(prediction[0])):
                prediction[0, i, -1, 0] = data_normalisation(final_array[0][i], min_c[i, 0], max_c[i, 0])
                prediction[0, i, -1, 1] = data_normalisation(final_array[1][i], min_c[i, 1], max_c[i, 1])
            f += 1
            if f > 100:
                if gif:
                    fig = plt.Figure(figsize=[3, 3], dpi=100)
                    canvas = FigureCanvas(fig)
                    ax = fig.gca()
                    ax.scatter(data_adjusted[1], data_adjusted[0], s=0.5)
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.axhline(0)
                    ax.axis('off')
                    canvas.draw()
                    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    image_array.append(image)
        if gif:
            make_gif(image_array, "upper_gifs/pred_adjust_0_{}".format(np.round(start_average, 5)))
        else:
            # pass
            axs.scatter(len(average_array[:]), average_array[-1], color='k', s=4.5, marker='x')
            axs.plot(average_array[:], label=np.round(start_average, 5), color=colors[cl])
            # np.save("generated_samples/COM_{}".format(start_average), np.array(average_array[:]))
            # axins.plot(average_array[:10], label=np.round(start_average, 5), color=colors[cl], linewidth=0.4)
        # plt.show()
        # plt.scatter(xy[-10][1], xy[-10][0])
        # plt.title(str(sim))
        # plt.ylim([-1,1])
        # plt.xlim([-1, 1])
        # plt.show()
        # print(start_average)
        # standard_average_array.append(str(start_average))
        xy_array.append(xy)
        cl += 1

    # plt.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.05),
    #       ncol=2, fancybox=True, shadow=True)
    pbar.close()
    # for i in range(len(xy_array)-1):
    #     t_diff = []
    #     for j in range(np.minimum(len(xy_array[i]), len(xy_array[i+1]))):
    #         x1 = np.array(xy_array[i][j][0])
    #         x2 = np.array(xy_array[i+1][j][0])
    #         x_diff = np.power(np.abs(x1 - x2), 2)
    #         y1 = np.array(xy_array[i][j][1])
    #         y2 = np.array(xy_array[i + 1][j][1])
    #         y_diff = np.power(np.abs(y1 - y2), 2)
    #         diff = np.sum(np.sqrt(x_diff+y_diff))
    #         if len(t_diff) > 1:
    #             if (diff-t_diff[-1])/t_diff[-1] > 0.2:
    #                 break
    #         t_diff.append(diff)
    #     plt.plot(t_diff, color=colors[i])
    # plt.yscale('log')
    # x = np.average(np.array(xy_array[-1])[75:150, 0])
    # y = np.average(np.array(xy_array[-1])[0:150, 0], axis=1)
    # x = np.arange(len(y))
    # vals = fit_sin(x, y)
    t = np.arange(0, 175)
    ys = -0.027983396808249045 * np.sin(0.21087895045432684 * t + 0.05372832864299764) + 0.004779817208552361
    # print(vals)
    axs.set_ylabel("$\hat{y}$")
    axs.plot(t + 1, ys, color='k', linestyle='--')
    plt.show()
    for i in range(len(xy_array)):
        x = np.array(xy_array[i])[-4, 0]
        y = np.array(xy_array[i])[-4, 1]
        plt.plot(x, y)
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        plt.show()

    # axins.set_xlim(0, 0.1)
    # axins.set_ylim(0.006, 0.008)
    # axins.yaxis.tick_right()
    # axins.xaxis.set_visible(False)
    # mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # axs.indicate_inset_zoom(axins, edgecolor="black")
    # plt.draw()
    # axs.set_ylabel("$\hat{y}$")
    # axs.legend()
    # plt.savefig("generated_samples_plot.png")
    # plt.show()
    # text_file_upper.close()
    # text_file_lower.close()

    # x = np.arange(len(average_array[10:250]))
    # vals = fit_sin(x, average_array[10:250])
    # print(vals)


def single_adjust(model_mae, frames, points, min_c, max_c, min_c_1, max_c_1):
    com = 0.01584
    data_mae = creating_data_graph(1, 100, [3], 3, 50)
    data_mae[0] = data_normalisation(data_mae[0], min_c, max_c)
    prediction = np.copy(data_mae[0][0:1])
    average = np.average(prediction[0, :, 0, 0])
    diff = -average + com
    prediction[0, :, 0, 0] = prediction[0, :, 0, 0] + diff
    # image_array = []
    plt.Figure(figsize=[5, 5], dpi=200)
    average_array = []
    f = 0
    succ_count = 0
    while True:
        # fig = plt.Figure(figsize=[3, 3], dpi=100)
        # canvas = FigureCanvas(fig)
        # ax = fig.gca()
        pred = model_mae(prediction).numpy()
        corr_x = data_unnormalisation(prediction[0, :, -1, 0], min_c[:, 0], max_c[:, 0])
        corr_y = data_unnormalisation(prediction[0, :, -1, 1], min_c[:, 1], max_c[:, 1])
        diff_x = data_unnormalisation(pred[0, :, 0], min_c_1[:, 0], max_c_1[:, 0])
        diff_y = data_unnormalisation(pred[0, :, 1], min_c_1[:, 1], max_c_1[:, 1])
        data_adjusted = position_transform([corr_x, corr_y], [diff_x, diff_y])

        if f % 1 == 0 and f > 1 and abs(np.average(data_adjusted[0])) < 0.0002:
            succ_count += 1
            try:
                dat = data_creation.make_lines(data_adjusted[0], data_adjusted[1], 0.001)
            except:
                print("fail", f, com)
                break
            final_array = [np.zeros(100), np.zeros(100)]
            circ = data_creation.circumfrance(dat)
            length = circ / 100
            circ_data = data_creation.circ_array(dat)
            for i in range(100):
                idx = (np.abs(circ_data - length * i)).argmin()
                final_array[0][i] = dat[0][idx]
                final_array[1][i] = dat[1][idx]
            if final_array[0][0] - final_array[0][1] > 0:
                final_array = np.flip(final_array, axis=1)
                final_array = np.insert(final_array, 0, final_array[:, -1], axis=1)
                final_array = np.delete(final_array, -1, axis=1)
        else:
            final_array = data_adjusted
        y_average = np.average(final_array[0])
        # if abs(y_average) > 0.04:
        #     break
        if len(average_array) > 60:
            if np.min(average_array[-50:]) > -0.022:
                break
        average_array.append(y_average)
        prediction[0, :, :-1] = prediction[0, :, 1:]
        for i in range(len(prediction[0])):
            prediction[0, i, -1, 0] = data_normalisation(final_array[0][i], min_c[i, 0], max_c[i, 0])
            prediction[0, i, -1, 1] = data_normalisation(final_array[1][i], min_c[i, 1], max_c[i, 1])
        # ax.scatter(data_adjusted[1], data_adjusted[0], s=0.5)
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        #
        # ax.axhline(0)
        # ax.axis('off')
        # canvas.draw()
        # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # image_array.append(image)
        f += 1
    plt.plot(average_array, label=np.round(com, 5), color='b')
    plt.show()


def prediction_adjust(model_mae, frames, points, min_c, max_c, min_c_1, max_c_1):
    data_mae = creating_data_graph(frames, points, [15], 3, 50)
    data_mae[0] = data_normalisation(data_mae[0], min_c, max_c)
    diff = 0.002
    colors = cm.winter(np.linspace(0, 1, 15))

    upper = 0.01757
    lower = 0.0175

    central = 0.01756632754296874
    difference = 0.001
    while True:
        best_len = 0
        cl = 0
        new_central = central
        for sim in np.arange(central - difference, central + difference, difference / 5):
            prediction = np.copy(data_mae[0][0:1])
            prediction[0, :, 0, 0] = prediction[0, :, 0, 0] + sim
            # image_array = []
            plt.Figure(figsize=[5, 5], dpi=200)
            average_array = []
            f = 0
            succ_count = 0
            while True:
                # fig = plt.Figure(figsize=[3, 3], dpi=100)
                # canvas = FigureCanvas(fig)
                # ax = fig.gca()
                pred = model_mae(prediction).numpy()
                corr_x = data_unnormalisation(prediction[0, :, -1, 0], min_c[:, 0], max_c[:, 0])
                corr_y = data_unnormalisation(prediction[0, :, -1, 1], min_c[:, 1], max_c[:, 1])
                diff_x = data_unnormalisation(pred[0, :, 0], min_c_1[:, 0], max_c_1[:, 0])
                diff_y = data_unnormalisation(pred[0, :, 1], min_c_1[:, 1], max_c_1[:, 1])
                data_adjusted = position_transform([corr_x, corr_y], [diff_x, diff_y])

                if f % 1 == 0 and f > 1 and abs(np.average(data_adjusted[0])) < 0.0002:
                    succ_count += 1
                    try:
                        dat = data_creation.make_lines(data_adjusted[0], data_adjusted[1], 0.001)
                    except:
                        print("fail", f, sim)
                        break
                    final_array = [np.zeros(100), np.zeros(100)]
                    circ = data_creation.circumfrance(dat)
                    length = circ / 100
                    circ_data = data_creation.circ_array(dat)
                    for i in range(100):
                        idx = (np.abs(circ_data - length * i)).argmin()
                        final_array[0][i] = dat[0][idx]
                        final_array[1][i] = dat[1][idx]
                    if final_array[0][0] - final_array[0][1] > 0:
                        final_array = np.flip(final_array, axis=1)
                        final_array = np.insert(final_array, 0, final_array[:, -1], axis=1)
                        final_array = np.delete(final_array, -1, axis=1)
                else:
                    final_array = data_adjusted
                y_average = np.average(final_array[0])
                if abs(y_average) > 0.04:
                    break
                if len(average_array) > 60:
                    if np.min(average_array[-50:]) > -0.022:
                        break
                average_array.append(y_average)
                prediction[0, :, :-1] = prediction[0, :, 1:]
                for i in range(len(prediction[0])):
                    prediction[0, i, -1, 0] = data_normalisation(final_array[0][i], min_c[i, 0], max_c[i, 0])
                    prediction[0, i, -1, 1] = data_normalisation(final_array[1][i], min_c[i, 1], max_c[i, 1])
                # ax.scatter(data_adjusted[1], data_adjusted[0], s=0.5)
                # ax.set_xlim([-1, 1])
                # ax.set_ylim([-1, 1])
                #
                # ax.axhline(0)
                # ax.axis('off')
                # canvas.draw()
                # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # image_array.append(image)
                f += 1
            if len(average_array) > best_len:
                best_len = len(average_array)
                new_central = sim
            plt.plot(average_array, label=np.round(sim, 5), color=colors[cl])
            cl += 1
            print(cl, sim, len(average_array), succ_count)
        plt.ylim([-0.04, 0.04])
        # plt.legend(loc=2)
        plt.show()
        # make_gif(image_array, "pred_adjust/pred_adjust_0_{}".format(sim*diff))
        central = new_central
        print("Best Central:", new_central)
        difference = difference / 4


def position_transform(data, change):
    data[0] = change[0] + data[0]
    data[1] = change[1] + data[1]
    return data


def prediction_gif(model_mae, initial_data, min_c, max_c, min_c_1, max_c_1, type=True, name="pred_gif"):
    prediction = np.array([initial_data[0][0]])
    # prediction = np.copy(initial_data[0][0:1])
    # prediction[0, :, 0, 0] = prediction[0, :, 0, 0] + 0.017516327542968738
    # plt.scatter(*zip(*prediction[0, :, 0:2]))
    # plt.Figure(figsize=[5, 5], dpi=300)
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # plt.show()
    # plt.clf()
    # prediction = np.reshape(prediction, newshape=(1, 1, np.shape(prediction)[1], 4))
    image_array = []
    x_predictions = []
    y_predictions = []
    x_actual = []
    y_actual = []
    mse = []
    for f in range(int(len(initial_data[0]) / 5) - 3):
        # for f in range(1550):
        #     if f%10==0:
        #         print(f)
        fig, axs = plt.subplots(3, 1, figsize=[5, 7], dpi=200, gridspec_kw={'height_ratios': [3, 1, 1]})
        canvas = FigureCanvas(fig)
        # ax = fig.gca()
        pred = model_mae(prediction).numpy()
        corr_x = data_unnormalisation(prediction[0, :, -1, 0], min_c[:, 0], max_c[:, 0])
        corr_y = data_unnormalisation(prediction[0, :, -1, 1], min_c[:, 1], max_c[:, 1])
        diff_x = data_unnormalisation(pred[0, :, 0], min_c_1[:, 0], max_c_1[:, 0])
        diff_y = data_unnormalisation(pred[0, :, 1], min_c_1[:, 1], max_c_1[:, 1])
        data_adjusted = position_transform([corr_x, corr_y], [diff_x, diff_y])
        if f % 1 == 0 and f > 30 and abs(np.average(data_adjusted[0])) < 0.002:
            try:
                dat = data_creation.make_lines(data_adjusted[0], data_adjusted[1], 0.001)
            except:
                print("fail", f)
                break
            final_array = [np.zeros(100), np.zeros(100)]
            idx = 0
            circ = data_creation.circumfrance(dat)
            length = circ / 100
            # data = data_creation.redefine_index(dat, idx)
            # data[1][0] = 0
            circ_data = data_creation.circ_array(dat)
            for i in range(100):
                idx = (np.abs(circ_data - length * i)).argmin()
                final_array[0][i] = dat[0][idx]
                final_array[1][i] = dat[1][idx]
            if final_array[0][0] - final_array[0][1] > 0:
                final_array = np.flip(final_array, axis=1)
                final_array = np.insert(final_array, 0, final_array[:, -1], axis=1)
                final_array = np.delete(final_array, -1, axis=1)
        else:
            final_array = data_adjusted

        prediction[0, :, :-1] = prediction[0, :, 1:]
        for i in range(len(prediction[0])):
            prediction[0, i, -1, 0] = data_normalisation(final_array[0][i], min_c[i, 0], max_c[i, 0])
            prediction[0, i, -1, 1] = data_normalisation(final_array[1][i], min_c[i, 1], max_c[i, 1])

        if type:
            x = data_unnormalisation(initial_data[0][5 + f * 5, :, -1, 0], min_c[:, 0], max_c[:, 0])
            y = data_unnormalisation(initial_data[0][5 + f * 5, :, -1, 1], min_c[:, 1], max_c[:, 1])
            x_predictions.append(final_array[0])
            y_predictions.append(final_array[1])
            x_actual.append(x)
            y_actual.append(y)
            axs[0].scatter(y, x, s=0.5)
            axs[0].scatter(final_array[1], final_array[0], s=0.5)
            axs[0].set_xlim([-1, 1])
            axs[0].set_ylim([-1, 1])
            mse = losses.mean_absolute_error(x_actual, x_predictions).numpy()
        else:
            x = initial_data[1][5 + f * 5, :, 0]
            y = initial_data[1][5 + f * 5, :, 1]
            axs[0].plot(y, x, linewidth=0.5)
            axs[0].scatter(pred[0, :, 1], pred[0, :, 0], s=0.5)
            axs[0].set_xlim([-1, 1])
            axs[0].set_ylim([-1, 1])
        axs[1].plot(np.average(x_predictions, axis=1))
        axs[1].plot(np.average(x_actual, axis=1))
        axs[2].plot(mse)
        axs[2].set_yscale('log')
        # ax.axvline(0)
        axs[0].axhline(0)
        # ax.axis('off')
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_array.append(image)
    make_gif(image_array, name)
    return [x_predictions, y_predictions], [x_actual, y_actual]


def prediction_losses(model_mae, frames, points, min_c_mae, max_c_mae, min_c_mae_1, max_c_mae_1, type=True):
    plt.figure(dpi=200, figsize=[5, 5])
    loss_totals = []
    colors = cm.winter(np.linspace(0, 1, 16))
    for sim in range(0, 16):
        data_mae = creating_data_graph(frames, points, [sim], 3, -1)
        copy_data1 = np.copy(data_mae[0])
        data_mae[0] = data_normalisation(data_mae[0], min_c_mae, max_c_mae)
        data_mae[1] = data_normalisation(data_mae[1], min_c_mae_1, max_c_mae_1)
        # data[1] = data_normalisation(data[1], min_c, max_c)
        prediction = np.array([data_mae[0][0]])
        x_predictions = []
        y_predictions = []
        x_actual = []
        y_actual = []
        count = 0
        for f in range(int(len(data_mae[0]) / 5) - 3):
            pred = model_mae(prediction).numpy()
            corr_x = data_unnormalisation(prediction[0, :, -1, 0], min_c_mae[:, 0], max_c_mae[:, 0])
            corr_y = data_unnormalisation(prediction[0, :, -1, 1], min_c_mae[:, 1], max_c_mae[:, 1])
            diff_x = data_unnormalisation(pred[0, :, 0], min_c_mae_1[:, 0], max_c_mae_1[:, 0])
            diff_y = data_unnormalisation(pred[0, :, 1], min_c_mae_1[:, 1], max_c_mae_1[:, 1])
            data_adjusted = position_transform([corr_x, corr_y], [diff_x, diff_y])
            # plt.scatter(data_adjusted[0], data_adjusted[1], s=0.3)
            if f % 1 == 0 and f > 1 and abs(np.average(data_adjusted[0])) > 0.031:
                count += 1
                print(count, f)
                try:
                    dat = data_creation.make_lines(data_adjusted[0], data_adjusted[1], 0.001)
                except:
                    print("fail", f / ((len(data_mae[0]) / 5) - 3))
                    break
                final_array = [np.zeros(points), np.zeros(points)]
                idx = 0
                circ = data_creation.circumfrance(dat)
                length = circ / points
                # data = data_creation.redefine_index(dat, idx)
                # data[1][0] = 0
                circ_data = data_creation.circ_array(dat)
                for i in range(points):
                    idx = (np.abs(circ_data - length * i)).argmin()
                    final_array[0][i] = dat[0][idx]
                    final_array[1][i] = dat[1][idx]
                if final_array[0][0] - final_array[0][1] > 0:
                    final_array = np.flip(final_array, axis=1)
                    final_array = np.insert(final_array, 0, final_array[:, -1], axis=1)
                    final_array = np.delete(final_array, -1, axis=1)
            else:
                final_array = data_adjusted
            # plt.scatter(final_array[0], final_array[1], s=0.3)
            # plt.scatter(final_array[0][0], final_array[1][0], color='g')
            # plt.scatter(data_adjusted[0][0], data_adjusted[1][0], color='r')
            # plt.show()
            prediction[0, :, :-1] = prediction[0, :, 1:]
            for i in range(len(prediction[0])):
                prediction[0, i, -1, 0] = data_normalisation(final_array[0][i], min_c_mae[i, 0], max_c_mae[i, 0])
                prediction[0, i, -1, 1] = data_normalisation(final_array[1][i], min_c_mae[i, 1], max_c_mae[i, 1])

            if type:
                x = copy_data1[5 + f * 5, :, -1, 0]
                y = copy_data1[5 + f * 5, :, -1, 1]
                x_predictions.append(data_adjusted[0])
                y_predictions.append(data_adjusted[1])
                x_actual.append(x)
                y_actual.append(y)
            else:
                x = copy_data1[1][5 + f * 5, :, 0]
                y = copy_data1[1][5 + f * 5, :, 1]
        xy_predictions, xy_actual = [x_predictions, y_predictions], [x_actual, y_actual]
        loss_arr = []
        for i in range(len(xy_predictions[0])):
            xa = xy_actual[0][i]
            xp = xy_predictions[0][i]
            mse = losses.mean_absolute_error(xa, xp).numpy()
            loss_arr.append(mse)
        plt.plot(loss_arr[:], label=sim, color=colors[sim])
        loss_totals.append(loss_arr)
    arr = pd.DataFrame(loss_totals).mean(axis=0).to_numpy()
    plt.plot(arr, color='k')
    plt.yscale('log')
    plt.ylim((5 * pow(10, -5), 0.1))
    plt.legend()
    plt.show()


def make_gif(images, name):
    imageio.mimsave("{}.gif".format(name), images)


def velocity_calculation(data, time_step):
    data = np.array(data)
    velocity = [data[time_step] - data[0]]
    for i in range(1, len(data) - time_step):
        vel = (data[i + time_step] - data[i])
        velocity.append(vel)
    return np.array(velocity)


def load_data(points, frames, time_step, simulation_array, initial, final):
    training_data = []
    labels = []
    pbar = tqdm(total=len(simulation_array))
    for simulations in simulation_array:
        simulation = simulations
        pbar.update()
        data_names = glob.glob("training_data/new_new2_xmin_Simulation_{}_points_{}/*".format(simulation, points))
        folder_length = len(data_names)
        data_array = []
        if final == -1:
            end_point = folder_length - 1
        else:
            end_point = final
        for file_num in range(initial, end_point - 5):
            data = np.load(
                "training_data/new_new2_xmin_Simulation_{}_points_{}/data_{}.npy".format(simulation, points, file_num))
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
        vel_data = velocity_calculation(xy_data, time_step)
        for i in range(0, len(xy_data) - frames * time_step - time_step):
            # for i in range(0, len(xy_data)-100):
            single = []
            for j in range(0, frames):
                row = np.array(xy_data[i + j * time_step + time_step])
                # row = np.append(row, vel_data[i + j * time_step], axis=1)
                single.append(row)
            vel = vel_data[i + frames * time_step]
            training_data.append(single)
            labels.append(vel)
    pbar.close()
    return [np.array(training_data), np.array(labels)]


def creating_data_graph(frames, points, simulations_array, initial, final):
    data = load_data(points, frames, 5, simulations_array, initial, final)
    data = [data[0][:], data[1][:]]
    data[0] = np.transpose(data[0], axes=(0, 2, 1, 3))
    return data


def data_normalisation_constants(data, axis):
    min_array = []
    max_array = []
    for i in range(len(data[0])):
        min_placeholder = np.min(data[:, i], axis=axis)
        max_placeholder = np.max(data[:, i], axis=axis)
        if axis == 1:
            min_array.append(np.min(min_placeholder, axis=0))
            max_array.append(np.max(max_placeholder, axis=0))
        else:
            min_array.append(min_placeholder)
            max_array.append(max_placeholder)

    return np.array(min_array), np.array(max_array)


def data_normalisation(data, min_array, max_array):
    u = 1
    l = -1
    try:
        for i in range(len(data[0])):
            data[:, i] = ((data[:, i] - min_array[i]) / (max_array[i] - min_array[i])) * (u - l) + l
    except:
        data = ((data - min_array) / (max_array - min_array)) * (u - l) + l
    return data


def data_unnormalisation(data, min_, max_):
    u = 1
    l = -1
    data = (data - l) * (max_ - min_) / (u - l) + min_
    return data


def main():
    backend.set_floatx('float64')
    print("Running Training")
    activation = activations.swish
    optimizer = optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    initialiser_w = initializers.VarianceScaling(scale=2.9)
    initialiser_b = initializers.RandomNormal(stddev=0.04)
    frames = 1
    points = 100

    simulations_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    data_mae = creating_data_graph(frames, points, simulations_array, 3, -1)
    min_c_mae, max_c_mae = data_normalisation_constants(data_mae[0], axis=1)
    min_c_mae_1, max_c_mae_1 = data_normalisation_constants(data_mae[1], axis=0)
    data_mae[0] = data_normalisation(data_mae[0], min_c_mae, max_c_mae)
    data_mae[1] = data_normalisation(data_mae[1], min_c_mae_1, max_c_mae_1)

    loss_mae = losses.mean_absolute_error
    model_mae = dense_network(100, frames, 2, activation, optimizer, 200, 200, 2, 12, initialiser_w, initialiser_b,
                              loss_mae)

    min_c = min_c_mae
    max_c = max_c_mae
    min_c_1 = min_c_mae_1
    max_c_1 = max_c_mae_1
    print(model_mae.summary())
    model_mae.load_weights("mae_loss0.0010667422569220822.h5")

    # plot_data()
    max_runs = 300
    # initial_bubble_base = 11
    num_interval = 11
    offsets = np.linspace(-0.01, 0.01, num_interval)
    for b in range(0, 16):
        initial_bubble_base = b
        initial_bubble = creating_data_graph(frames, points, [initial_bubble_base], 3, -1)
        initial_bubble = initial_bubble[0]
        bubble_variants = []
        final_states = []
        bar = tqdm(total=num_interval)
        plt.figure(figsize=(10, 7))
        for offset in offsets:
            prediction = np.zeros((1, points, 1, 2))
            prediction[:, :, :, :] = initial_bubble[:1]

            bubble_positions = []
            prediction[:, :, :, 0] += offset
            prediction = data_normalisation(prediction, min_c, max_c)
            for future in range(max_runs):
                pred = model_mae(prediction).numpy()
                corr_x = data_unnormalisation(prediction[0, :, -1, 0], min_c[:, 0], max_c[:, 0])
                corr_y = data_unnormalisation(prediction[0, :, -1, 1], min_c[:, 1], max_c[:, 1])
                diff_x = data_unnormalisation(pred[0, :, 0], min_c_1[:, 0], max_c_1[:, 0])
                diff_y = data_unnormalisation(pred[0, :, 1], min_c_1[:, 1], max_c_1[:, 1])
                data_adjusted = position_transform([corr_x, corr_y], [diff_x, diff_y])
                if abs(np.average(data_adjusted[0])) < 0.001 and False:
                    print("Adjustment made at step {}".format(future))
                    try:
                        dat = data_creation.make_lines(data_adjusted[0], data_adjusted[1], 0.001)
                    except:
                        print("Failed to do that")
                        final_states.append(4)
                        break
                    final_array = [np.zeros(100), np.zeros(100)]
                    circ = data_creation.circumfrance(dat)
                    length = circ / 100
                    circ_data = data_creation.circ_array(dat)
                    for i in range(100):
                        idx = (np.abs(circ_data - length * i)).argmin()
                        final_array[0][i] = dat[0][idx]
                        final_array[1][i] = dat[1][idx]
                    if final_array[0][0] - final_array[0][1] > 0:
                        final_array = np.flip(final_array, axis=1)
                        final_array = np.insert(final_array, 0, final_array[:, -1], axis=1)
                        final_array = np.delete(final_array, -1, axis=1)
                else:
                    final_array = data_adjusted
                circumference = 0
                for i in range(1, points):
                    first_x = final_array[1][i-1]
                    first_y = final_array[0][i-1]
                    now_x = final_array[1][i]
                    now_y = final_array[0][i]
                    circumference += np.sqrt((first_x - now_x)**2 + (first_y - now_y)**2)
                if circumference >= 4.2:
                    final_states.append(4)
                    break
                if np.mean(final_array[0]) > 0.31:
                    final_states.append(1)
                    bubble_positions.append(final_array)
                    break
                if np.mean(final_array[0]) < -0.31:
                    final_states.append(2)
                    bubble_positions.append(final_array)
                    break
                bubble_positions.append(final_array)
                prediction[0, :, :-1] = prediction[0, :, 1:]
                for i in range(len(prediction[0])):
                    prediction[0, i, -1, 0] = data_normalisation(final_array[0][i], min_c[i, 0], max_c[i, 0])
                    prediction[0, i, -1, 1] = data_normalisation(final_array[1][i], min_c[i, 1], max_c[i, 1])
            if len(bubble_positions) == max_runs:
                final_states.append(3)
            bubble_variants.append(bubble_positions)
            if offset == 0:
                actual = initial_bubble[5::5]
                predicted = np.asarray(bubble_positions)
                print(np.shape(actual))
                print(np.shape(predicted))
                final_point_actual = np.shape(actual)[0] - 1
                final_point_predicted = np.shape(predicted)[0] - 1
                final_point = np.minimum(final_point_actual, final_point_predicted)
                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                plt.fill_between([-1, 1], 0.33, -0.33, color="gray", alpha=0.2, label="Rail")
                plt.scatter(actual[final_point, :, 0, 1], actual[final_point, :, 0, 0], label="Simulation")
                plt.scatter(predicted[final_point, 1, :], predicted[final_point, 0, :], label="Prediction")
                plt.legend()
                plt.xlabel("x position")
                plt.ylabel("y position")
                plt.savefig("{}_final_diff.png".format(initial_bubble_base), dpi=200)
                plt.clf()
                print("Ping")

            bar.update(1)
        bar.close()
        colours = cm.cool(np.linspace(0, 1, len(bubble_variants)))
        index = 0
        for bubble_series in bubble_variants:
            avg_y = []
            cir = []
            for bubble in bubble_series:
                avg_y.append(np.mean(bubble[0]))
                circumference = 0
                for i in range(1, points):
                    first_x = bubble[1][i-1]
                    first_y = bubble[0][i-1]
                    now_x = bubble[1][i]
                    now_y = bubble[0][i]
                    circumference += np.sqrt((first_x - now_x)**2 + (first_y - now_y)**2)
                cir.append(circumference)
            plt.plot(cir, color=colours[index])
            index += 1
        avg_y_correct = []

        for bubble in initial_bubble[5::5]:
            circumference = 0
            for i in range(1, points):
                first_x = bubble[i-1][0][1]
                first_y = bubble[i-1][0][0]
                now_x = bubble[i][0][1]
                now_y = bubble[i][0][0]
                circumference += np.sqrt((first_x - now_x)**2 + (first_y - now_y)**2)
            avg_y_correct.append(circumference)

        plt.plot(avg_y_correct, color='red', label="Actual simulation")
        plt.grid()
        plt.xlabel("Steps in prediction")
        plt.ylabel("Bubble circumference")
        plt.plot([0, max_runs], [4.2, 4.2], label="Prediction cutoff", ls='--', color='black')
        plt.legend()
        plt.savefig("{}_circumference.png".format(initial_bubble_base), dpi=200)
        plt.clf()
        colours = cm.cool(np.linspace(0, 1, len(bubble_variants)))
        index = 0
        avg_ys = []
        for bubble_series in bubble_variants:
            avg_y = []
            cir = []
            for bubble in bubble_series:
                avg_y.append(np.mean(bubble[0]))
            plt.plot(avg_y, color=colours[index])
            avg_ys.append(avg_y)
            index += 1
        avg_y_correct = []

        for bubble in initial_bubble[5::5]:
            avg_y_correct.append(np.mean(bubble[:, 0, 0]))
        # plt.ylim([-0.4, 0.4])
        r = 0.031415
        x = np.linspace(0, max_runs, 500)
        # omega = np.pi / 15
        position_alpha = 51.9
        position_beta = 289.5
        omega = np.pi * 2 / ((position_beta - position_alpha)/8)
        correction = -omega * position_alpha
        y = r * np.cos((x + 0.2) * omega + correction)
        plt.plot(avg_y_correct, color='red', label="Actual simulation")
        plt.plot(x, y, ls="--", color="black", label="Unstable periodic orbit")
        plt.plot(x, np.zeros(500) + 0.31415, ls=':', color="black", label="Fixed points")
        plt.plot(x, np.zeros(500), ls=':', color="black")
        plt.plot(x, np.zeros(500) - 0.31415, ls=':', color="black")
        plt.xlabel("Steps in prediction")
        plt.ylabel("Average y position")
        plt.grid()
        # plt.plot([0, max_runs], [0.3, 0.3], label="Prediction cutoff", ls='--', color='black')
        # plt.plot([0, max_runs], [-0.3, -0.3], ls='--', color='black')
        plt.legend()
        plt.xlim([0, max_runs])
        # plt.show()
        plt.savefig("{}_y_position.png".format(initial_bubble_base), dpi=200)
        plt.clf()
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        ii = 0
        for index, bubble_series in enumerate(bubble_variants):
            if final_states[index] == 4:
                plt.scatter(bubble_series[-1][0], bubble_series[-1][1], color=colours[ii])
            ii += 1
        plt.scatter(initial_bubble[-1, :, 0, 0], initial_bubble[-1, :, 0, 1], color='red')
        plt.savefig("{}_final_position.png".format(initial_bubble_base), dpi=200)
        plt.clf()

        y_vels = []
        for i, avg in enumerate(avg_ys):
            p = np.asarray(avg)
            v = np.gradient(p, 2)
            y_vels.append(v)
            plt.plot(p, v, color=colours[i])
        p_correct = np.asarray(avg_y_correct)
        v_correct = np.gradient(p_correct, 2)
        r = 0.031
        theta = np.linspace(0, np.pi * 2, 500)
        x = r * np.cos(theta)
        y = r * np.sin(theta) / 10
        plt.plot(p_correct, v_correct, color='red', label="Actual simulation")
        plt.plot(x, y, color="black", ls="--", label="Unstable periodic orbit")
        plt.scatter([-0.31415, 0, 0.31415], [0, 0, 0], label="Fixed points", marker="o", color="black")
        close = 47
        start = 0
        end = 100
        # plt.scatter([avg_ys[10][close]], [y_vels[10][close]], label="Close", marker="x", color="red", zorder=1000)
        # plt.scatter([avg_ys[20][close]], [y_vels[20][close]], marker="x", color="red", zorder=1000)
        # plt.scatter([avg_ys[30][close]], [y_vels[30][close]], marker="x", color="red", zorder=1000)
        # plt.scatter([avg_ys[40][close]], [y_vels[40][close]], marker="x", color="red", zorder=1000)
        #
        # plt.scatter([avg_ys[10][start]], [y_vels[10][start]], label="Start", marker="x", color="blue", zorder=1000)
        # plt.scatter([avg_ys[20][start]], [y_vels[20][start]], marker="x", color="blue", zorder=1000)
        # plt.scatter([avg_ys[30][start]], [y_vels[30][start]], marker="x", color="blue", zorder=1000)
        # plt.scatter([avg_ys[40][start]], [y_vels[40][start]], marker="x", color="blue", zorder=1000)
        #
        # plt.scatter([avg_ys[10][end]], [y_vels[10][end]], label="End", marker="x", color="black", zorder=1000)
        # plt.scatter([avg_ys[20][end]], [y_vels[20][end]], marker="x", color="black", zorder=1000)
        # plt.scatter([avg_ys[30][end]], [y_vels[30][end]], marker="x", color="black", zorder=1000)
        # plt.scatter([avg_ys[40][end]], [y_vels[40][end]], marker="x", color="black", zorder=1000)
        # print(avg_ys[10][47])
        plt.ylabel("Average y velocity")
        plt.xlabel("Average y position")
        plt.grid()
        plt.legend()
        plt.savefig("{}_phase_space.png".format(initial_bubble_base), dpi=200)
        # plt.show()
        plt.clf()

        top_y = []
        top_v = []
        bottom_y = []
        bottom_v = []
        break_y = []
        break_v = []
        for i, value in enumerate(final_states):
            average_y_positions = avg_ys[i]
            average_y_velocities = y_vels[i]
            termination_reason = value
            if termination_reason == 1:
                top_y += list(average_y_positions)
                top_v += list(average_y_velocities)
            elif termination_reason == 2:
                bottom_y += list(average_y_positions)
                bottom_v += list(average_y_velocities)
            elif termination_reason == 4:
                break_y += list(average_y_positions)
                break_v += list(average_y_velocities)
        top_y = np.asarray(top_y)
        bottom_y = np.asarray(bottom_y)
        break_y = np.asarray(break_y)
        top_v = np.asarray(top_v)
        bottom_v = np.asarray(bottom_v)
        break_v = np.asarray(break_v)
        final_top, _, _ = np.histogram2d(top_v, top_y, range=((-0.02, 0.015), (-0.35, 0.35)), bins=(500, 500))
        final_bottom, _, _ = np.histogram2d(bottom_v, bottom_y, range=((-0.02, 0.015), (-0.35, 0.35)), bins=(500, 500))
        final_break, _, _ = np.histogram2d(break_v, break_y, range=((-0.02, 0.015), (-0.35, 0.35)), bins=(500, 500))
        final_array = np.zeros((500, 500, 3))
        final_array[:, :, 2] = np.minimum(final_break, np.zeros((500, 500))+1)
        final_array[:, :, 1] = np.minimum(final_top, np.zeros((500, 500))+1)
        final_array[:, :, 0] = np.minimum(final_bottom, np.zeros((500, 500))+1)
        plt.yticks(np.linspace(0, 499, 8), np.round(np.linspace(0, 1, 8)*0.035 - 0.02, 3))
        plt.xticks(np.linspace(0, 499, 11), np.round(np.linspace(0, 1, 11)*0.7 - 0.35, 3))
        plt.imshow(final_array)
        plt.ylim([0, 500])
        plt.xlim([0, 500])
        plt.scatter([999], [0], color="green", label="Terminates above rail")
        plt.scatter([999], [0], color="red", label="Terminates below rail")
        plt.scatter([999], [0], color="blue", label="Breaks up")
        plt.legend(loc="lower right")
        plt.ylabel("Average y velocity")
        plt.xlabel("Average y position")
        plt.savefig("{}_phase_domain.png".format(initial_bubble_base), dpi=200)
        plt.clf()

if __name__ == "__main__":
    main()
