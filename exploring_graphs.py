import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import glob
from tensorflow.keras import layers, models, Model, initializers, losses, optimizers, activations


def make_gif(image, name):
    images = []
    for i in image:
        images.append(i)
    imageio.mimsave("{}.gif".format(name), images)


def read_file(file_path):
    """Takes in a filepath and extracts a set of x and y coordinates of the bubble edge.
    Input:
        A string of the file path of a .dat file to read in
    Output:
        A pair of 1D  arrays of floats (x and y)
    """
    x = []
    y = []
    try:
        file = open(file_path, "r")
        main_data = False
        for line in file:
            # Excluding data that is irrelevant (the walls of the image)
            if "boundary4" in line:
                main_data = True
            if main_data and "ZONE" not in line:
                data_points = line.strip().split(" ")
                x.append(float(data_points[1]))
                y.append(float(data_points[0]))
        file.close()
    except IOError:
        print("File {} not found.".format(file_path))
        x = []
        y = []
    except ValueError:
        print("One of the lines in {} was unexpected.".format(file_path))
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def create_network(alpha: float, kernel: int, channels: int, number_of_points: int):
    initializer = initializers.HeNormal()
    activation_function = activations.swish
    optimizer = optimizers.Adam(learning_rate=0.001, epsilon=0.1)
    model = models.Sequential()
    if kernel != 2:
        span = int(np.ceil(alpha * number_of_points/(kernel - 2)))
    else:
        span = int(np.ceil(alpha * number_of_points/(kernel - 1)))

    model.add(layers.Conv2D(channels, 1,
                            activation=activation_function, kernel_initializer=initializer,
                            input_shape=(number_of_points, 2, 1)))
    for i in range(0, span):
        model.add(layers.Conv2D(channels, (kernel, 2),
                                padding='same', activation=activation_function, kernel_initializer=initializer))
    model.add(layers.Conv2D(1, 1, padding='same', activation='sigmoid', kernel_initializer=initializer))
    model.compile(optimizer=optimizer, loss=losses.mean_squared_logarithmic_error, run_eagerly=False, metrics=[
        losses.binary_crossentropy, losses.mean_squared_logarithmic_error
    ])
    return model


def main():
    # test_data_source = "Simulation_data_extrapolated/Simulation_False_0_0.001_0"
    # # test_data_source = "Simulation_data/r0.54eps18.6"
    # images = []
    # number_of_points = 10
    # pbar = tqdm(total=971-3)
    # reference = [-1, -1]
    # first_run = True
    # positions = np.zeros((971-3, 2, number_of_points))
    # for i in range(3, 971):
    #     pbar.update(1)
    #     test_array = np.load("{}/data_{}.npy".format(test_data_source, i))
    #     # y, x = read_file("{}/boundaries_{}.dat".format(test_data_source, i))
    #     # test_array = np.zeros((2, len(x)))
    #     # test_array[1, :] = x
    #     # test_array[0, :] = y
    #     total_data_points = np.shape(test_array)[1]
    #     smallest_dist = 99
    #     ref_index = 0
    #     for j in range(0, total_data_points):
    #         distance = (test_array[0, j] - reference[0])**2 + (test_array[1, j] - reference[1])**2
    #         if distance < smallest_dist:
    #             ref_index = j
    #             smallest_dist = distance
    #     test_array_new = np.zeros((2, total_data_points))
    #     test_array_new[:, :total_data_points-ref_index] = test_array[:, ref_index:]
    #     test_array_new[:, total_data_points-ref_index:] = test_array[:, :ref_index]
    #     test_array = test_array_new
    #     reference = [test_array[0, 0], test_array[1, 0]]
    #     actual_y = []
    #     actual_x = []
    #     lapse_point = total_data_points/number_of_points
    #     if first_run:
    #         references = np.zeros((2, number_of_points))
    #         for j in range(0, number_of_points):
    #             central_point = int((j+0.5)*lapse_point)
    #             actual_x.append(test_array[1][central_point])
    #             actual_y.append(test_array[0][central_point])
    #             positions[i-3, 0, j] = test_array[1, central_point]
    #             positions[i-3, 1, j] = test_array[0, central_point]
    #             references[0, j] = test_array[0, central_point]
    #             references[1, j] = test_array[1, central_point]
    #     else:
    #         for k in range(0, number_of_points):
    #             ref_index = 0
    #             smallest_dist = 99
    #             for j in range(0, total_data_points):
    #                 distance = (test_array[0, j] - references[0, k]) ** 2 + (test_array[1, j] - references[1, k]) ** 2
    #                 if distance < smallest_dist:
    #                     ref_index = j
    #                     smallest_dist = distance
    #             references[0, k] = test_array[0, ref_index]
    #             references[1, k] = test_array[1, ref_index]
    #             actual_x.append(references[1, k])
    #             actual_y.append(references[0, k])
    #             positions[i-3, 0, k] = references[1, k]
    #             positions[i-3, 1, k] = references[0, k]
    #     first_run = False
    #     plt.xlim([-1, 1])
    #     plt.ylim([-1, 1])
    #     plt.scatter(actual_x, actual_y)
    #     plt.scatter(test_array[1, 0], test_array[0, 0])
    #     plt.savefig("graph_gif_sources/{}.png".format(i))
    #     images.append(imageio.imread('graph_gif_sources/{}.png'.format(i)))
    #     plt.clf()
    # pbar.close()
    # make_gif(images, "matplotlib_tests")
    # exit()
    # dx = positions[:-1, 0, :] - positions[1:, 0, :]
    # dy = positions[:-1, 1, :] - positions[1:, 1, :]
    # print(np.shape(dx))
    # plt.figure(figsize=(20, 10))
    # # plt.ylim([-0.01, 0.01])
    # plt.scatter(np.linspace(0, 966, 967), dy[:, 0])
    # plt.scatter(np.linspace(0, 966, 967), dy[:, 50])
    # plt.savefig("dy_interpolated.png", dpi=500)
    # plt.clf()
    # plt.figure(figsize=(20, 10))
    # plt.scatter(np.linspace(0, 967, 968), positions[:, 1, 0])
    # plt.scatter(np.linspace(0, 967, 968), positions[:, 1, 50])
    # plt.savefig("y_interpolated.png", dpi=500)
    # plt.clf()
    number_of_points = 100
    amplification = 50
    correction = 1
    all_data_sources = glob.glob("Simulation_data/*")
    print(all_data_sources)
    questions = np.zeros((14913, number_of_points, 2, 1))
    answers = np.zeros((14913, number_of_points, 2, 1))
    total_sims = 0
    for new_data_source in all_data_sources:
        number_of_boundaries = len(glob.glob(new_data_source+"/b*"))
        positions = np.zeros((2, number_of_boundaries-3, number_of_points))
        first_run = True
        for i in range(3, number_of_boundaries):
            y, x = read_file("{}/boundaries_{}.dat".format(new_data_source, i))
            test_array = np.zeros((2, len(x)))
            test_array[1, :] = x
            test_array[0, :] = y
            reference = [test_array[0, 0], test_array[1, 0]]
            actual_y = []
            actual_x = []
            lapse_point = len(x)/number_of_points
            if first_run:
                references = np.zeros((2, number_of_points))
                for j in range(0, number_of_points):
                    central_point = int((j+0.5)*lapse_point)
                    actual_x.append(test_array[1][central_point])
                    actual_y.append(test_array[0][central_point])
                    positions[0, i-3, j] = test_array[1, central_point]
                    positions[1, i-3, j] = test_array[0, central_point]
                    references[0, j] = test_array[0, central_point]
                    references[1, j] = test_array[1, central_point]
            else:
                for k in range(0, number_of_points):
                    ref_index = 0
                    smallest_dist = 99
                    for j in range(0, len(x)):
                        distance = (test_array[0, j] - references[0, k]) ** 2 + (test_array[1, j] - references[1, k]) ** 2
                        if distance < smallest_dist:
                            ref_index = j
                            smallest_dist = distance
                    references[0, k] = test_array[0, ref_index]
                    references[1, k] = test_array[1, ref_index]
                    actual_x.append(references[1, k])
                    actual_y.append(references[0, k])
                    positions[0, i-3, k] = references[1, k]
                    positions[1, i-3, k] = references[0, k]
            first_run = False

        dx = positions[0, 1:, :] - positions[0, :-1, :]
        dy = positions[1, 1:, :] - positions[1, :-1, :]
        print(np.shape(dx))
        print(total_sims)
        print(np.shape(positions))
        questions[total_sims:total_sims+np.shape(positions)[1]-1, :, 0, 0] = positions[0, :-1, :]
        questions[total_sims:total_sims+np.shape(positions)[1]-1, :, 1, 0] = positions[1, :-1, :]
        answers[total_sims:total_sims+np.shape(positions)[1]-1, :, 0, 0] = dx*amplification + correction
        answers[total_sims:total_sims+np.shape(positions)[1]-1, :, 1, 0] = dy*amplification + correction
        total_sims += np.shape(positions)[1]-1
        # print(positions[0, 0, 0])
        # print("--")
        # print(positions[0, 1, 0])
        # print(positions[0, 0, 0] + dx[0, 0])
        # print("--")
        # print(positions[0, 2, 0])
        # print(positions[0, 1, 0] + dx[1, 0])
        # exit()
    print(total_sims)
    print(np.shape(answers))
    model = create_network(1, 5, 32, number_of_points)
    print(model.summary())
    model.fit(questions, answers, epochs=5, validation_split=0.1)

    predictions = np.zeros((900, number_of_points, 2))
    actuals = np.zeros((900, number_of_points, 2))

    actuals[:, :, :] = questions[:900, :, :, 0]
    predictions[0, :, :] = questions[0, :, :, 0]
    pbar = tqdm(total=899)
    for i in range(1, 900):
        pbar.update(1)
        next_point = np.zeros((1, number_of_points, 2, 1))
        next_point[0, :, :, 0] = predictions[i-1, :, :]
        predictions[i, :, :] = (next_point + (model(next_point)-correction)/amplification)[0, :, :, 0]
    pbar.close()
    images = []
    pbar = tqdm(total=899)
    for i in range(1, 900):
        pbar.update(1)
        a_x = actuals[i, :, 0]
        a_y = actuals[i, :, 1]
        p_x = predictions[i, :, 0]
        p_y = predictions[i, :, 1]
        plt.xlim([1, -1])
        plt.ylim([1, -1])
        plt.scatter(a_x, a_y, label="Actual")
        plt.scatter(p_x, p_y, label="Prediction")
        plt.legend()
        plt.savefig("graph_gif_sources/{}.png".format(i))
        plt.clf()
        images.append(imageio.imread('graph_gif_sources/{}.png'.format(i)))
    pbar.close()
    make_gif(images, "predictions")
    # plt.scatter(np.linspace(0, 975, 976), dx[:, 0])
    # plt.scatter(np.linspace(0, 975, 976), dx[:, 1])
    # plt.scatter(np.linspace(0, 975, 976), dx[:, 2])
    # plt.figure(figsize=(20, 10))
    # plt.scatter(np.linspace(0, 976, 977), positions[1, :, 0])
    # plt.scatter(np.linspace(0, 976, 977), positions[1, :, 50])
    # plt.savefig("y_non_interpolated.png", dpi=500)
    # plt.clf()
    # plt.figure(figsize=(20, 10))
    # # plt.ylim([-0.01, 0.01])
    # plt.scatter(np.linspace(0, 975, 976), dy[:, 0])
    # plt.scatter(np.linspace(0, 975, 976), dy[:, 50])
    # plt.savefig("dy_non_interpolated.png", dpi=500)


if __name__ == "__main__":
    main()
