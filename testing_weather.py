import imageio
import numpy as np
import matplotlib.pyplot as plt
import tqdm


def main():
    print("Yo")
    dataset = np.load("Meterology_data/data8.npz")
    print(dataset.files)
    print(np.shape(dataset["x_osgb"]))
    print(np.shape(dataset["y_osgb"]))
    data = dataset["data"][0, :, :, :]
    print(np.shape(data))
    normalised_data = np.tanh(data/500)

    del data
    del dataset
    image_size = 64
    image_frames = 4
    future_look = 5

    crop_range = [0, np.shape(normalised_data)[1]-image_size]
    crop_range = [0, 10]
    maximum_crops = (crop_range[1] - crop_range[0])**2
    maximum_times = np.shape(normalised_data)[0] - future_look - image_frames

    questions = np.zeros((maximum_crops*maximum_times, image_frames, image_size, image_size, 1))
    answers = np.zeros((maximum_crops*maximum_times, 2, image_size, image_size, 1))

    bar = tqdm.tqdm(total=maximum_crops*maximum_times)
    for x in range(crop_range[0], crop_range[1]):
        for y in range(crop_range[0], crop_range[1]):
            for f in range(maximum_times):
                bar.update(1)
                for frame in range(image_frames):
                    questions[
                        (x*(crop_range[1] - crop_range[0]) + y)*maximum_times + f,
                        frame,
                        :, :, 0
                    ] = normalised_data[f + frame, x:x + image_size, y:y + image_size]
                for i in range(2):
                    answers[
                        (x*(crop_range[1] - crop_range[0]) + y)*maximum_times + f,
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


if __name__ == "__main__":
    main()