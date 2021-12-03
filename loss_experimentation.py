import loss_functions
import numpy as np
import dat_to_training
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k, losses
import tensorflow as tf


def main():
    x, y = dat_to_training.read_file("Simulation_data/r0.54eps18.925/boundaries_750.dat")
    x, y = dat_to_training.make_lines(x, y)
    invert = False
    variant = 0
    BASE_SIZE = 60
    h, x_edge, y_edge = np.histogram2d(
        ((-1) ** (not invert)) * x, y + variant / (BASE_SIZE / 2),
        # range=[[-1, 1], [-1, 1]], bins=(image_size*multiply + kernel_size - 1, image_size*multiply + kernel_size - 1)
        range=[[-1, 1], [-1, 1]], bins=(BASE_SIZE, BASE_SIZE)
    )
    plt.imshow(h)
    plt.show()
    # example_images = np.zeros((1, 4, 54, 54, 1))
    # example_images[0, 0, :, :, 0] = dat_to_training.process_bmp(
    #     "Simulation_images/Simulation_0/img_55.bmp", 54, 1
    # )[:, :, 1]
    # example_images[0, 1, :, :, 0] = dat_to_training.process_bmp(
    #     "Simulation_images/Simulation_0/img_60.bmp", 54, 1
    # )[:, :, 1]
    # example_images[0, 2, :, :, 0] = dat_to_training.process_bmp(
    #     "Simulation_images/Simulation_0/img_65.bmp", 54, 1
    # )[:, :, 1]
    # example_images[0, 3, :, :, 0] = dat_to_training.process_bmp(
    #     "Simulation_images/Simulation_0/img_70.bmp", 54, 1
    # )[:, :, 1]
    # print(np.shape(example_images))
    # random_data = np.random.random((1, 4, 54, 54, 1))
    # zero_data = np.zeros((1, 4, 54, 54, 1))
    # hot_corners = np.zeros((1, 4, 54, 54, 1))
    # hot_corners[0, :, 0, 0, :] = 1
    # hot_corners[0, :, 0, 53, :] = 1
    # hot_corners[0, :, 53, 0, :] = 1
    # hot_corners[0, :, 53, 53, :] = 1
    # example_images = k.constant(example_images)
    # hot_corners = k.constant(hot_corners)
    # random_data = k.constant(random_data)
    # zero_data = k.constant(zero_data)
    # print(loss_functions.UBERLOSS(example_images, random_data))
    # print(loss_functions.UBERLOSS(example_images, example_images))
    # print(loss_functions.UBERLOSS(example_images, zero_data))
    # print(loss_functions.UBERLOSS(example_images, hot_corners))
    # print("UBER")
    # print(k.mean(loss_functions.UBERLOSS(example_images, random_data)))
    # print(k.mean(loss_functions.UBERLOSS(example_images, example_images)))
    # print(k.mean(loss_functions.UBERLOSS(example_images, zero_data)))
    # print(k.mean(loss_functions.UBERLOSS(example_images, hot_corners)))
    # print("--iou--")
    # print(loss_functions.iou_coef(example_images, random_data))
    # print(loss_functions.iou_coef(example_images, example_images))
    # print(loss_functions.iou_coef(example_images, zero_data))
    # print(loss_functions.iou_coef(example_images, hot_corners))
    # print("--dice--")
    # print(loss_functions.dice_coef(example_images, random_data))
    # print(loss_functions.dice_coef(example_images, example_images))
    # print(loss_functions.dice_coef(example_images, zero_data))
    # print(loss_functions.dice_coef(example_images, hot_corners))
    # print("--ssim--")
    # print(loss_functions.ssim_loss(example_images, random_data))
    # print(loss_functions.ssim_loss(example_images, example_images))
    # print(loss_functions.ssim_loss(example_images, zero_data))
    # print(loss_functions.ssim_loss(example_images, hot_corners))
    # print("--------")
    #
    # print("bce mean")
    # print(k.mean(losses.binary_crossentropy(example_images, random_data)))
    # print(k.mean(losses.binary_crossentropy(example_images, example_images)))
    # print(k.mean(losses.binary_crossentropy(example_images, zero_data)))
    # print(k.mean(losses.binary_crossentropy(example_images, hot_corners)))
    # print("ssim")
    # print(loss_functions.ssim_loss(example_images, random_data))
    # print(loss_functions.ssim_loss(example_images, example_images))
    # print(loss_functions.ssim_loss(example_images, zero_data))
    # print(loss_functions.ssim_loss(example_images, hot_corners))
    #
    # print("mse mean")
    # y_true = example_images
    # y_pred = random_data
    # moment_zero = k.sum(k.square(y_true-y_pred))/float(k.prod(k.shape(y_true)))
    # mean = k.mean(k.square(y_true-y_pred))
    # print(k.shape(y_true))
    # print(k.sum(k.shape(y_true)))
    # print(moment_zero)
    # print(mean)


if __name__ == "__main__":
    main()