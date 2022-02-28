import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm


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


def main():
    test_data_source = "Simulation_data_extrapolated/Simulation_False_0_0.001_0"
    images = []
    number_of_points = 10
    pbar = tqdm(total=971-3)
    reference = [-1, -1]
    first_run = True
    positions = np.zeros((971-3, 2, number_of_points))
    for i in range(3, 971):
        pbar.update(1)
        test_array = np.load("{}/data_{}.npy".format(test_data_source, i))
        total_data_points = np.shape(test_array)[1]
        smallest_dist = 99
        ref_index = 0
        for j in range(0, total_data_points):
            distance = (test_array[0, j] - reference[0])**2 + (test_array[1, j] - reference[1])**2
            if distance < smallest_dist:
                ref_index = j
                smallest_dist = distance
        test_array_new = np.zeros((2, total_data_points))
        test_array_new[:, :total_data_points-ref_index] = test_array[:, ref_index:]
        test_array_new[:, total_data_points-ref_index:] = test_array[:, :ref_index]
        test_array = test_array_new
        reference = [test_array[0, 0], test_array[1, 0]]
        actual_y = []
        actual_x = []
        lapse_point = total_data_points/number_of_points
        if first_run:
            references = np.zeros((2, number_of_points))
            for j in range(0, number_of_points):
                central_point = int((j+0.5)*lapse_point)
                actual_x.append(test_array[1][central_point])
                actual_y.append(test_array[0][central_point])
                positions[i-3, 0, j] = test_array[1, central_point]
                positions[i-3, 1, j] = test_array[0, central_point]
                references[0, j] = test_array[0, central_point]
                references[1, j] = test_array[1, central_point]
        else:
            for k in range(0, number_of_points):
                ref_index = 0
                smallest_dist = 99
                for j in range(0, total_data_points):
                    distance = (test_array[0, j] - references[0, k]) ** 2 + (test_array[1, j] - references[1, k]) ** 2
                    if distance < smallest_dist:
                        ref_index = j
                        smallest_dist = distance
                references[0, k] = test_array[0, ref_index]
                references[1, k] = test_array[1, ref_index]
                actual_x.append(references[1, k])
                actual_y.append(references[0, k])
                positions[i-3, 0, k] = references[1, k]
                positions[i-3, 1, k] = references[0, k]
        first_run = False
        # plt.xlim([-1, 1])
        # plt.ylim([-1, 1])
        # plt.scatter(actual_x, actual_y)
        # plt.scatter(test_array[1, 0], test_array[0, 0])
        # plt.savefig("graph_gif_sources/{}.png".format(i))
        # images.append(imageio.imread('graph_gif_sources/{}.png'.format(i)))
        # plt.clf()
    pbar.close()
    # make_gif(images, "matplotlib_tests")
    dx = positions[:-1, 0, :] - positions[1:, 0, :]
    dy = positions[:-1, 1, :] - positions[1:, 1, :]
    print(np.shape(dx))
    plt.figure(figsize=(20, 10))
    plt.ylim([-0.01, 0.01])
    plt.scatter(np.linspace(0, 966, 967), dy[:, 0])
    plt.savefig("dy_interpolated.png", dpi=500)
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.scatter(np.linspace(0, 967, 968), positions[:, 1, 0])
    plt.savefig("y_interpolated.png", dpi=500)
    plt.clf()
    new_data_source = "Simulation_data/r0.54eps18.6"
    positions = np.zeros((2, 977, number_of_points))
    first_run = True
    for i in range(0, 977):
        y, x = read_file("{}/boundaries_{}.dat".format(new_data_source, i))
        test_array = np.zeros((2, len(x)))
        # positions[0, i, :] = x[:30]
        # positions[1, i, :] = y[:30]
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
                positions[0, i, j] = test_array[1, central_point]
                positions[1, i, j] = test_array[0, central_point]
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
                positions[0, i, k] = references[1, k]
                positions[1, i, k] = references[0, k]
        first_run = False

    dx = positions[0, :-1, :] - positions[0, 1:, :]
    dy = positions[1, :-1, :] - positions[1, 1:, :]
    print(np.shape(dx))
    # plt.scatter(np.linspace(0, 975, 976), dx[:, 0])
    # plt.scatter(np.linspace(0, 975, 976), dx[:, 1])
    # plt.scatter(np.linspace(0, 975, 976), dx[:, 2])
    plt.figure(figsize=(20, 10))
    plt.scatter(np.linspace(0, 976, 977), positions[1, :, 0])
    plt.savefig("y_non_interpolated.png", dpi=500)
    plt.clf()
    plt.figure(figsize=(20, 10))
    plt.ylim([-0.01, 0.01])
    plt.scatter(np.linspace(0, 975, 976), dy[:, 0])
    plt.savefig("dy_non_interpolated.png", dpi=500)


if __name__ == "__main__":
    main()
