import dat_to_training
# import bubble_prediction
import numpy as np
# import pickle
# import os
# from dill.source import getsource
import matplotlib.pyplot as plt
from tensorflow.keras import models
# import tensorflow as tf
import imageio
# import datetime
# import models as md


def long_term_prediction(model, start_sim, start_image, image_size, timestep, frames, number_to_simulate,
                         round_result=False):
    input_images = dat_to_training.create_gif_dataset(start_sim, start_image, frames, image_size, timestep)
    image = input_images[0, 0, :, :, :]
    dat_to_training.generate_rail(image)
    rail = [image[:, :, 2:]]
    zeros = [image[:, :, 0:1]]
    working_frames = input_images
    y_pred = model(working_frames).numpy()
    y_pred = np.append(zeros, y_pred, 3)
    y_pred = np.append(y_pred, rail, 3)
    output_images = y_pred
    y_pred_shape = np.shape(y_pred)
    y_pred = np.reshape(y_pred, newshape=(y_pred_shape[0], 1, y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]))
    working_frames = working_frames[:, 1:, :, :, :]
    working_frames = np.append(working_frames, y_pred, 1)
    for i in range(0, number_to_simulate - 1):
        y_pred = model(working_frames).numpy()
        y_pred = np.append(zeros, y_pred, 3)
        y_pred = np.append(y_pred, rail, 3)
        output_images = np.append(output_images, y_pred, axis=0)
        y_pred = np.reshape(y_pred, newshape=(y_pred_shape[0], 1, y_pred_shape[1], y_pred_shape[2], y_pred_shape[3]))
        working_frames = working_frames[:, 1:, :, :, :]
        working_frames = np.append(working_frames, y_pred, 1)
    output_images = (output_images * 255).astype(np.uint8)
    return output_images


def make_gif(image, name):
    images = []
    for i in image:
        images.append(i)
    imageio.mimsave("{}.gif".format(name), images)

def model_gif(directory, model, simulation, start, image_size, timestep, image_frames, gif_length=200):
    pred_positions = long_term_prediction(model, simulation, start, image_size, timestep, image_frames, gif_length, round_result=False)
    true_positions = np.array(dat_to_training.create_gif_dataset(simulation, start+image_frames*timestep, gif_length, image_size, timestep)*255, dtype=np.int32)
    final_gif = pred_positions[:len(true_positions[0]), :, :, :]
    final_gif[:, :, :, 0] = true_positions[0, :, :, :, 1]
    imageio.mimsave("{}.gif".format(directory + "/gifs/sim{}_start{}_timestep_{}".format(str(simulation), str(start), str(timestep))), final_gif)
    return final_gif

def func():
    directorys = ["experiment_3/03_01_2022 00_56","experiment_3/03_01_2022 05_24","experiment_3/03_01_2022 09_49", "experiment_3/03_01_2022 16_53", "experiment_3/04_01_2022 06_29","experiment_3/04_01_2022 13_14", "experiment_3/04_01_2022 15_04", "experiment_3/test_1", "experiment_3/test_2", "experiment_3/test_3"]
    directorys2 = ["experiment_3/03_01_2022 16_53", "experiment_3/test_1", "experiment_3/test_2", "experiment_3/test_3", "experiment_3/test_5"]
    directorys3 = ["experiment_3/03_01_2022 00_56", "experiment_3/03_01_2022 05_24", "experiment_3/03_01_2022 09_49", "experiment_3/03_01_2022 16_53", "experiment_3/04_01_2022 13_14", "experiment_3/04_01_2022 15_04", "experiment_3/04_01_2022 16_58", "experiment_3/04_01_2022 23_12", "experiment_3/04_01_2022 20_18", "experiment_3/05_01_2022 03_03"]
    directorys4 = ["experiment_3/04_01_2022 23_12"]
    for i in range(10):
        plt.figure(figsize=(8, 6), dpi=100)
        for start_pos in range(50 , 501, 50):
            for directory in directorys4:
                print(i, directory)
                model = models.load_model(directory)
                fin = model_gif(directory, model, i, start_pos, 60, 5, 4, 1000)
                y_true = fin[:, :, :, 0]
                y_pred = fin[:, :, :, 1]
                loss = []
                for j in range(len(y_true)):
                    se = ((y_true[j, :, :] - y_pred[j, :, :])/255) ** 2
                    mse = np.mean(se)
                    loss.append(mse)
                x = range(start_pos, len(loss)*5 + start_pos, 5)
                plt.plot(x, loss, label=directory)
        plt.legend()
        plt.savefig("experiment_3/sim"+str(i)+".png")
        plt.clf()

def main():
    func()


if __name__ == "__main__":
    main()