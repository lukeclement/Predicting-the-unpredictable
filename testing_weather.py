import imageio
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
import skimage.measure


def extract_chain_info(data, frames, future_look):
    # Extracting from data file that should looks like [days, minutes, window_number, size, size]

    days = np.shape(data)[0]
    minutes = np.shape(data)[1]
    windows = np.shape(data)[2]
    image_size = np.shape(data)[3]

    questions = np.zeros((days*(minutes-frames-future_look)*windows, frames, image_size, image_size, 1))
    answers = np.zeros((days*(minutes-frames-future_look)*windows, 2, image_size, image_size, 1))
    for window in range(windows):
        for day in range(days):
            for minute in range(minutes-frames-future_look):
                key_index = (window*days + day)*(minutes-frames-future_look) + minute
                for frame in range(frames):
                    questions[key_index, frame, :, :, 0] = data[day, minute + frame, window, :, :]
                answers[key_index, 0, :, :, 0] = data[day, minute + frames, window, :, :]
                answers[key_index, 1, :, :, 0] = data[day, minute + frames + future_look, window, :, :]

    print(np.shape(questions))
    batch_size = 8
    print("Turning into dataset...")
    testing_data = tf.data.Dataset.from_tensor_slices((questions, answers))
    print("Batching...")
    testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    print("Sending off...")
    return testing_data


def simplify_data(original_data, window_sizes, window_downscaling=1, mod=1500):

    data = np.tanh(skimage.measure.block_reduce(original_data, (1, 1, window_downscaling, window_downscaling), np.mean)/mod)
    del original_data
    print(np.shape(data))
    maximum_windows_per_axis = np.shape(data)[2] // window_sizes
    windows = np.zeros((np.shape(data)[0], np.shape(data)[1], maximum_windows_per_axis**2, window_sizes, window_sizes))
    for window_x in range(0, maximum_windows_per_axis):
        for window_y in range(0, maximum_windows_per_axis):
            start_x = window_x * window_sizes
            start_y = window_y * window_sizes
            end_x = start_x + window_sizes
            end_y = start_y + window_sizes
            window_number = window_x * maximum_windows_per_axis + window_y
            windows[:, :, window_number, :, :] = data[:, :, start_x:end_x, start_y:end_y]
    return windows


def main():
    print("Yo")
    dataset = np.load("Meterology_data/data8.npz")
    print(dataset.files)
    print(np.shape(dataset["x_osgb"]))
    print(np.shape(dataset["y_osgb"]))
    data = dataset["data"][:1, :, :, :]
    del dataset
    print(np.shape(data))
    data = simplify_data(data, 64, window_downscaling=1)
    print(data)
    # plt.imshow(data[0, 0, 0, :, :])
    # plt.show()
    extract_chain_info(data, 4, 10)
    # normalised_data = np.tanh(data/500)

    # output =
    # Crop and format
    print()
    del data
    # image_size = size
    # image_frames = frames
    # future_look = future

    exit()


    crop_range = [0, np.shape(normalised_data)[1]]
    # crop_range = [140, 150]
    maximum_crops = ((crop_range[1] - crop_range[0])//image_size)**2
    maximum_times = np.shape(normalised_data)[0] - future_look - image_frames

    x = crop_range[1]-image_size
    y = crop_range[1]-image_size
    yo = (x - crop_range[0])*(crop_range[1] - crop_range[0])//(image_size**2) + (y - crop_range[0])//image_size
    actual_max = yo * maximum_times + maximum_times-1
    print(yo * maximum_times + maximum_times-1)

    questions = np.zeros((maximum_crops*maximum_times, image_frames, image_size, image_size, 1))
    answers = np.zeros((maximum_crops*maximum_times, 2, image_size, image_size, 1))
    bar = tqdm.tqdm(total=maximum_crops*maximum_times)
    for x in range(crop_range[0], crop_range[1]-image_size, image_size):
        for y in range(crop_range[0], crop_range[1]-image_size, image_size):
            for f in range(maximum_times):
                bar.update(1)
                x_index = (x - crop_range[0])*(crop_range[1] - crop_range[0])//(image_size**2)
                y_index = (y - crop_range[0])//image_size
                for frame in range(image_frames):
                    print((x_index + y_index)*maximum_times + f)
                    print(maximum_crops*maximum_times)
                    questions[
                        (x_index + y_index)*maximum_times + f,
                        frame,
                        :, :, 0
                    ] = normalised_data[f + frame, x:x + image_size, y:y + image_size]
                for i in range(2):
                    answers[
                        (x_index + y_index)*maximum_times + f,
                        i,
                        :, :, 0
                    ] = normalised_data[f + image_frames + i*future_look, x:x + image_size, y:y + image_size]
    bar.close()

    image_converts = questions[:maximum_times, 0, :, :, 0] * 255
    image_converts = image_converts.astype(np.uint8)
    images = []
    for i in image_converts:
        images.append(i)
    imageio.mimsave("Met_test.gif", images)
    batch_size = 8
    print(np.shape(questions))
    print(np.shape(answers))
    testing_data = tf.data.Dataset.from_tensor_slices((questions, answers))
    testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return testing_data


if __name__ == "__main__":
    main()