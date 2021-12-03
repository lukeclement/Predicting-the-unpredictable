import tensorflow as tf
from tensorflow.keras import backend as k, losses


def UBERLOSS(y_true, y_pred):
    test = tester_loss(y_true, y_pred)
    dice = dice_coef(y_true, y_pred)
    iou = iou_coef(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    mse = losses.mean_squared_logarithmic_error(y_true, y_pred)
    bce = losses.binary_crossentropy(y_true, y_pred)
    if k.mean(bce) > 0.4:
        return (bce - k.tanh(iou*0.5) - k.tanh(dice*0.5) + k.sigmoid(ssim) + 2)/2
    else:
        return (mse - k.tanh(iou*0.5) - k.tanh(dice*0.5) + k.sigmoid(ssim) + 2)/2


def cubic_loss(y_true, y_pred):
    constant = tf.fill(tf.shape(y_true), 1.005)
    se = (y_pred - y_true) * (y_pred - y_true)
    dominator = constant - y_true
    bce = losses.binary_crossentropy(y_true, y_pred)
    loss = 0.143*k.log(40*k.mean(se / dominator)+1)+bce
    return loss


def relative_diff(y_true, y_pred):
    return k.sum(k.sqrt(k.abs(y_true-y_pred)))


def tester_loss(y_true, y_pred):
    total_guess = k.sum(y_pred, axis=[2, 3, 4])
    total_actual = k.sum(y_true, axis=[2, 3, 4])
    axis_a_sum_pred = k.sum(y_pred, axis=[3, 4])
    axis_a_sum_true = k.sum(y_true, axis=[3, 4])
    axis_b_sum_pred = k.sum(y_pred, axis=[2, 4])
    axis_b_sum_true = k.sum(y_true, axis=[2, 4])
    diff_a = k.abs(axis_a_sum_pred - axis_a_sum_true)
    diff_b = k.abs(axis_b_sum_pred - axis_b_sum_true)
    total_diff = k.sum(diff_a + diff_b, axis=1)
    mse = losses.mean_squared_logarithmic_error(y_true, y_pred)
    c = 0.0
    if k.sum(y_pred) < 4.0:
        c = 9999999.0
    return mse + k.mean(total_diff) + c


def absolute_diff(y_true, y_pred):
    frame_differences = k.sum(k.abs(y_true-y_pred), axis=[2, 3, 4])
    return k.mean(frame_differences)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = k.sum(y_true * y_pred, axis=[2, 3, 4])
    union = k.sum(y_true, axis=[2, 3, 4]) + k.sum(y_pred, axis=[2, 3, 4])
    dice = k.mean((2. * intersection + smooth) / (union + smooth), axis=1)
    dice_t = k.sum(dice, axis=0)
    return dice_t


def iou_coef(y_true, y_pred, smooth=1):
    intersection = k.sum(k.abs(y_true * y_pred), axis=[2, 3, 4])
    union = k.sum(y_true, [2, 3, 4]) + k.sum(y_pred, [2, 3, 4]) - intersection
    iou = k.mean((intersection + smooth) / (union + smooth), axis=1)
    iou_t = k.sum(iou, axis=0)
    return iou_t


def ssim_loss(y_true, y_pred):
    mid_t = k.mean(y_true, axis=1)
    mid_p = k.mean(y_pred, axis=1)
    return 1 - tf.reduce_sum(tf.image.ssim(mid_t, mid_p, 1.0))
    # return 1 - tf.image.ssim(mid_t, mid_p, 1.0)


def mse_dice(y_true, y_pred):
    mse = losses.mean_squared_logarithmic_error(y_true, y_pred)
    di = k.log(dice_coef(y_true, y_pred))
    iou = k.log(iou_coef(y_true, y_pred))
    return mse - di - iou


def bce_dice(y_true, y_pred):
    bce = losses.binary_crossentropy(y_true, y_pred)
    di = k.log(dice_coef(y_true, y_pred))
    iou = k.log(iou_coef(y_true, y_pred))
    # com = k.log(com_coef(y_true, y_pred))
    # return bce - di - iou + com
    return bce - di - iou


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
