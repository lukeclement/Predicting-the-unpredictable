import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as k, losses


def construct_r_table(image, seeking_range=5):
    data = image[:, :, 1]
    reference_point = calculate_com(image, both=True)
    r_values = []
    thi_values = []
    for x, values in enumerate(data):
        for y, value in enumerate(values):
            if value > 0 and x != np.shape(data)[0]-1 and y != np.shape(data)[0]-1 and x != 0 and y != 0:
                alpha = np.arctan2(reference_point[1] - y, reference_point[0] - x)
                radius = np.sqrt((x-reference_point[0])**2 + (y-reference_point[1])**2)
                points_x = []
                points_y = []
                for i in range(0, seeking_range):
                    for j in range(0, seeking_range):
                        for v in range(0, int(data[x+i-int(seeking_range/2), y+j-int(seeking_range/2)]*10)):
                            points_x.append(x+i-int(seeking_range/2))
                            points_y.append((y+j-int(seeking_range/2)))
                if max(points_y) - min(points_y) > max(points_x) - min(points_x):
                    p = np.polyfit(points_y, points_x, 1)
                    thi = np.pi/2 - np.arctan(p[0])
                    thi = np.floor(thi*100)/100
                else:
                    p = np.polyfit(points_x, points_y, 1)
                    thi = np.floor(np.arctan(p[0])*100)/100
                # see if this thi already exists
                if thi in thi_values:
                    # find thi in thi_values and then append correct r_value pairing
                    index = thi_values.index(thi)
                    r_values[index].append([radius, alpha])
                else:
                    # append thi_values and r_values
                    thi_values.append(thi)
                    r_values.append([[radius, alpha]])
    return r_values, thi_values


def generate_rail(input_image):
    image_size = np.shape(input_image)[0]
    for i in range(0, image_size):
        if i < image_size/2:
            rail = i / (image_size / 2)
        else:
            rail = 2 - i / (image_size / 2)
        runway = np.zeros(image_size) + rail
        input_image[i, :, 2] = runway
    return input_image


def calculate_com(image_a, both=False):
    image_size = np.shape(image_a)[0]
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')
    x_com = np.sum(x * image_a[:, :, 1]) / np.sum(image_a[:, :, 1])
    y_com = np.sum(y * image_a[:, :, 1]) / np.sum(image_a[:, :, 1])
    if both:
        return x_com, y_com
    return np.sqrt((x_com-(float(image_size)/2))**2 + (y_com-(float(image_size)/2))**2)


def process_bmp(filename, image_size):
    x, y = np.load(filename)
    h, x_edge, y_edge = np.histogram2d(
        x, y,
        range=[[-1, 1], [-1, 1]], bins=(image_size, image_size)
    )
    output_array = np.zeros((image_size, image_size, 3))
    h = np.tanh(image_size * 0.05 * h / 100)
    output_array[:, :, 1] = h
    output_array = generate_rail(output_array)
    return output_array


def custom_loss(y_true, y_pred):
    y_true = k.sum(y_true, axis=[2, 3, 4])
    y_pred = k.sum(y_pred, axis=[2, 3, 4])
    ref_table, thetas = construct_r_table(y_true)
    spread_pred = detect_object(y_pred, ref_table, thetas)
    spread_true = detect_object(y_true, ref_table, thetas)
    # s_true = np.max(spread_true)/np.sum(spread_true)
    # s_pred = np.max(spread_true)/np.sum(spread_pred)
    s_true = np.mean(spread_true[spread_true > 0])
    s_pred = np.mean(spread_pred[spread_pred > 0])
    if np.isnan(s_pred):
        s_pred = 0
    loss = 1 - s_pred/s_true
    return loss


def detect_object(image, r_table, thetas, seeking_range=5):
    data = image[:, :, 1]
    image_size = np.shape(data)[0]
    buffer = 100
    spread = np.zeros((image_size+buffer, image_size+buffer))

    for x, values in enumerate(data):
        for y, value in enumerate(values):
            if value > 0 and \
                    x < np.shape(data)[0] - int(seeking_range / 2) and y < np.shape(data)[0] - int(seeking_range / 2) \
                    and x > int(seeking_range / 2) and y > int(seeking_range / 2):
                points_x = []
                points_y = []
                for i in range(0, seeking_range):
                    for j in range(0, seeking_range):
                        for v in range(0, int(data[x + i - int(seeking_range / 2), y + j - int(seeking_range / 2)] * 10)):
                            points_x.append(x + i - int(seeking_range / 2))
                            points_y.append((y + j - int(seeking_range / 2)))
                if max(points_y) - min(points_y) > max(points_x) - min(points_x):
                    p = np.polyfit(points_y, points_x, 1)
                    thi = np.pi / 2 - np.arctan(p[0])
                    thi = np.floor(thi*100)/100
                else:
                    p = np.polyfit(points_x, points_y, 1)
                    thi = np.floor(np.arctan(p[0])*100)/100
                if thi in thetas:
                    index = thetas.index(thi)
                    for element in r_table[index]:
                        candidate_x = int(np.floor(x + element[0]*np.cos(element[1])))
                        candidate_y = int(np.floor(y + element[0]*np.sin(element[1])))
                        spread[candidate_x+int(buffer/2), candidate_y+int(buffer/2)] += 1
    return spread


def UBERLOSS(y_true, y_pred):
    # test = tester_loss(y_true, y_pred)
    dice = dice_coef(y_true, y_pred)
    iou = iou_coef(y_true, y_pred)
    # ssim = ssim_loss(y_true, y_pred)
    mse = losses.mean_squared_logarithmic_error(y_true, y_pred)
    bce = losses.binary_crossentropy(y_true, y_pred)
    if k.mean(bce) > 0.1:
        return (bce - k.tanh(iou*0.5) - k.tanh(dice*0.5) + 2)/2
        # return (bce + k.sigmoid(ssim))/2
    else:
        return (mse - k.tanh(iou*0.5) - k.tanh(dice*0.5) + 2)/2
        # return (mse + k.sigmoid(ssim))/2


def UBERLOSS_minus_dice(y_true, y_pred):
    # test = tester_loss(y_true, y_pred)
    dice = dice_coef(y_true, y_pred)
    iou = iou_coef(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    mse = losses.mean_squared_logarithmic_error(y_true, y_pred)
    bce = losses.binary_crossentropy(y_true, y_pred)
    if k.mean(bce) > 0.1:
        # return (bce)/2
        return (bce + k.sigmoid(ssim))/2
    else:
        # return (mse)/2
        return (mse + k.sigmoid(ssim))/2


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
    frame_differences = k.sum(k.abs(y_true-y_pred), axis=[2, 3])
    return k.mean(frame_differences)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = k.sum(y_true * y_pred, axis=[2, 3])
    union = k.sum(y_true, axis=[2, 3]) + k.sum(y_pred, axis=[2, 3])
    dice = k.mean((2. * intersection + smooth) / (union + smooth), axis=1)
    dice_t = k.sum(dice, axis=0)
    return dice_t


def iou_coef(y_true, y_pred, smooth=1):
    intersection = k.sum(k.abs(y_true * y_pred), axis=[2, 3])
    union = k.sum(y_true, [2, 3]) + k.sum(y_pred, [2, 3]) - intersection
    iou = k.mean((intersection + smooth) / (union + smooth), axis=1)
    iou_t = k.sum(iou, axis=0)
    return iou_t


def ssim_loss(y_true, y_pred):
    # mid_t = k.mean(y_true, axis=1)
    # mid_p = k.mean(y_pred, axis=1)
    return 1 - tf.reduce_sum(tf.image.ssim_multiscale(y_true, y_pred, 1.0))
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


def jsd(y_true, y_pred):
    mean = 0.5*(y_true + y_pred)
    alpha = losses.kld(y_true, mean)
    beta = losses.kld(y_pred, mean)
    return 0.5*alpha + 0.5*beta
