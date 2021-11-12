import tensorflow as tf
from tensorflow.keras import backend as k, losses


def iou_coef(y_true, y_pred, smooth=1):
    # print(y_true)
    intersection = k.sum(k.abs(y_true * y_pred), axis=[1, 2, 3])
    union = k.sum(y_true, [1, 2, 3]) + k.sum(y_pred, [1, 2, 3]) - intersection
    iou = k.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def bce_dice(y_true, y_pred):
    bce = losses.binary_crossentropy(y_true, y_pred)
    # bce = losses.MeanSquaredError(y_true, y_pred)
    di = k.log(dice_coef(y_true, y_pred))
    iou = k.log(iou_coef(y_true, y_pred))
    # com = k.log(com_coef(y_true, y_pred))
    # return bce - di - iou + com
    return bce - di - iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = k.sum(y_true * y_pred, axis=[1, 2, 3])
    union = k.sum(y_true, axis=[1, 2, 3]) + k.sum(y_pred, axis=[1, 2, 3])
    dice = k.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def com_coef(y_true, y_pred):
    i_1, j_1 = tf.meshgrid(tf.range(64), tf.range(64), indexing='ij')
    coords_1 = tf.stack([tf.reshape(i_1, (-1,)), tf.reshape(j_1, (-1,))], axis=-1)
    coords_1 = tf.cast(coords_1, tf.float32)
    volumes_flat_1 = tf.reshape(y_true, [-1, 64 * 64, 1])
    total_mass_1 = tf.reduce_sum(volumes_flat_1, axis=None)
    centre_of_mass_1 = tf.reduce_sum(volumes_flat_1 * coords_1, axis=None) / total_mass_1
    volumes_flat_2 = tf.reshape(y_pred, [-1, 64 * 64, 1])
    total_mass_2 = tf.reduce_sum(volumes_flat_2, axis=None)
    centre_of_mass_2 = tf.reduce_sum(volumes_flat_2 * coords_1, axis=None) / total_mass_2
    difference = centre_of_mass_1 - centre_of_mass_2
    return k.abs(difference)


def mass_preservation(y_true, y_pred, smooth=1):
    true_mass = k.sum(y_true)
    pred_mass = k.sum(y_pred)
    return k.exp(-k.sqrt(k.abs(true_mass - pred_mass)) / 2)
