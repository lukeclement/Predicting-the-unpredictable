import glob
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

BASE_SIZE = 540


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


def generate_rail(input_image):
    """Generates the central rail used in the bubble experiments.
    Can be modified as different functions, but current model is a simple triangle function.
    Input:
        input_image: A 3d numpy array containing the bubble edge data without a rail.
    Output:
        A 3d numpy array containing bubble edge data with a rail.
    """
    image_size = np.shape(input_image)[0]
    for i in range(0, image_size):
        if i < image_size/2:
            rail = i / (image_size / 2)
        else:
            rail = 2 - i / (image_size / 2)
        runway = np.zeros(image_size) + rail
        input_image[i, :, 2] = runway
    return input_image


def make_lines(x, y, resolution):
    """Creates a series of interpolated points between raw bubble edge data points.
    Inputs:
        x: A 1D array of floats from raw data points
        y: A 1D array of floats from raw data points
        resolution: A float, representing how close the interpolated points should be.
    Outputs:
        filled_x: A 1D array of interpolated data points
        filled_y: A 1D array of interpolated data points
    """
    current_x = x[0]
    current_y = y[0]
    visited = [0]
    while len(visited) < len(x):
        checked = []
        values = []
        for i in range(0, len(x)):
            if i not in visited:
                checked.append(i)
                values.append(
                    (current_x - x[i])**2 + (current_y - y[i])**2
                )
        closest = min(values)
        smallest = checked[values.index(closest)]
        visited.append(smallest)
        current_x = x[smallest]
        current_y = y[smallest]

    new_x = []
    new_y = []
    for i in visited:
        new_x.append(x[i])
        new_y.append(y[i])

    filled_x = []
    filled_y = []

    for i in range(0, len(new_x)):
        current_x = float(new_x[i])
        current_y = float(new_y[i])
        if i+1 != len(new_x):
            next_x = float(new_x[i+1])
            next_y = float(new_y[i+1])
        else:
            next_x = float(new_x[0])
            next_y = float(new_y[0])
        angle_to_next = np.arctan2(next_x - current_x, next_y - current_y)
        distance = np.sqrt((current_x - next_x)**2 + (current_y - next_y)**2)
        loops = 0
        while resolution*loops < distance:
            filled_x.append(current_x)
            filled_y.append(current_y)
            loops += 1
            current_x += resolution * np.sin(angle_to_next)
            current_y += resolution * np.cos(angle_to_next)
    filled_x = np.asarray(filled_x)
    filled_y = np.asarray(filled_y)

    return filled_x, filled_y


def transform_to_numpy_array(x, y, variant, invert):
    """Transforms a set of x and y coordinates to a numpy array of size 64x64 by default.
    Inputs:
        x:          A 1d numpy array of x coordinates.
        y:          A 1d numpy array of y coordinates.
        variant:    A float to offset the bubble position in the x direction.
        invert:     A boolean that will flip the y-axis through the center of the image.
        image_size: (default 64) An integer for the number of pixels to be used per axis.
    Output:
        A numpy array of shape (image_size, image_size) with elements 1 or 0 for the bubble edge.
    """
    h, x_edge, y_edge = np.histogram2d(
        ((-1)**(not invert))*x, y + variant / (BASE_SIZE / 2),
        # range=[[-1, 1], [-1, 1]], bins=(image_size*multiply + kernel_size - 1, image_size*multiply + kernel_size - 1)
        range=[[-1, 1], [-1, 1]], bins=(BASE_SIZE, BASE_SIZE)
    )

    # Preparing memory for the output array, then filling the bubble edge
    # print(np.max(h))
    output_array = np.minimum(h, np.zeros((BASE_SIZE, BASE_SIZE)) + 1)
    # Adding the central rail
    # output_array = generate_rail(output_array)
    output_array = 255*output_array
    output_array = output_array.astype(np.uint8)
    return output_array


def convert_dat_files(variant_range, resolution=0.0001):
    """Converts all .dat files to numpy arrays, and saves them as .bmp files.
    These .bmp files are stored in Simulation_data_extrapolated/Simulation_X,
    where X is the reference number for the simulation.
    These aren't necessarily actual simulations, but can be variants of these 'base' simulations,
    where the physics remains constant.
    Input:
        variant_range:  An array of two floats, defining the [minimum, maximum]
                            amount to shift the original images in the x-axis. This range is inclusive.
        resolution: (default 0.0001) A float defining the distance between points when the raw data is interpolated.
    Output:
        Nothing
    """
    simulation_names = glob.glob("Simulation_data/*")
    simulation_index = 0
    for simulation in simulation_names:
        dat_files = glob.glob("{}/b*.dat".format(simulation))
        tracking_index = 0
        for inversion in [True, False]:
            for variant in range(variant_range[0], variant_range[1]+1):
                if inversion:
                    print("File {}, flipped, shifted {} is now Simulation_{}_{}_{}_{}".format(
                        simulation, variant, inversion, variant, resolution, simulation_index
                    ))
                else:
                    print("File {}, shifted {} is now Simulation_{}_{}_{}_{}".format(
                        simulation, variant, inversion, variant, resolution, simulation_index
                    ))
                # Making a directory for the images
                try:
                    os.mkdir("Simulation_data_extrapolated/Simulation_{}_{}_{}_{}".format(
                        inversion, variant, resolution, simulation_index
                    ))
                except OSError:
                    print("Folder already exists!")
                # Now the heavy lifting
                pbar = tqdm(total=len(dat_files))
                for file in dat_files:
                    pbar.update(1)
                    # Extracting data
                    x, y = read_file(file)
                    # Finding the actual frame number
                    step_number = int(file[file.find("s_")+2:-4])
                    # Converting to array
                    x, y = make_lines(x, y, resolution)
                    # Saving to memory
                    data = np.array([((-1)**(not inversion))*x, y + (variant / BASE_SIZE)])
                    np.save("Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}".format(
                        inversion, variant, resolution, simulation_index, step_number
                    ), data)
                tracking_index += 1
                pbar.close()
        simulation_index += 1


def get_sim_result(source):
    outcomes = [3, 3, 0, 1, 3, 3, 3, 3, 0, 2, 3, 2, 3, 0, 3, 3]
    sim_metadata = source.split("/")[1].split("_")[1:]
    simulation_flippage = sim_metadata[0] == "True"
    simulation_offset = int(sim_metadata[1])
    simulation_resolution = float(sim_metadata[2])
    simulation_number = int(sim_metadata[3])
    result = outcomes[simulation_number]
    output = result
    if simulation_flippage and result == 0:
        output = 1
    if simulation_flippage and result == 1:
        output = 0
    final_array = np.zeros(4)
    final_array[output] = 1
    return final_array


def create_training_data(
        frames: int, timestep: int, validation_split=0.1, image_size=64,
        variants=None, flips_allowed=True, resolution=0.001, excluded_sims=None, easy_mode=False, var=False):
    if excluded_sims is None:
        excluded_sims = []
    if variants is None:
        variants = [0]
    batch_size = 8
    simulation_names = glob.glob("Simulation_data_extrapolated/*")
    data_sources = []
    refs = []
    print("Found {} simulation folders".format(len(simulation_names)))
    total = 0
    sub_total = 0
    print("Gathering references...")
    test_simulation_max = "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}".format(
        True, max(variants), resolution, 15)
    test_simulation_min = "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}".format(
        True, min(variants), resolution, 15)
    if test_simulation_max not in simulation_names or test_simulation_min not in simulation_names:
        print("Required training data not generated, now creating...")
        convert_dat_files([min(variants), max(variants)], resolution)
        simulation_names = glob.glob("Simulation_data_extrapolated/*")
    for simulation in simulation_names:
        sim_metadata = simulation.split("/")[1].split("_")[1:]
        simulation_flippage = sim_metadata[0] == "True"
        simulation_offset = int(sim_metadata[1])
        simulation_resolution = float(sim_metadata[2])
        simulation_number = int(sim_metadata[3])

        flip_test = flips_allowed or not simulation_flippage
        offset_test = simulation_offset in variants
        number_test = simulation_number not in excluded_sims
        if flip_test and offset_test and simulation_resolution == resolution and number_test:
            files = glob.glob("{}/*".format(simulation))
            number_of_files = len(files)
            sub_total += len(files)
            for i in range(3, number_of_files-timestep*frames*2):
                total += 1
                data_sources.append("{}/data_{}.npy".format(simulation, i))
                refs.append([simulation, i])
    print("Loading data...")
    source_array = np.zeros((len(data_sources), image_size, image_size, 3))
    pbar = tqdm(total=len(data_sources))
    for index, data in enumerate(data_sources):
        # print_progress(index, len(data_sources))
        pbar.update(1)
        source_array[index, :, :, :] = process_bmp(data, image_size)
    pbar.close()
    questions_array = np.zeros((
        int(np.floor(len(data_sources)*(1-validation_split))), frames, image_size, image_size, 3
    ), dtype="float16")
    if easy_mode:
        answers_array = np.zeros((
            int(np.floor(len(data_sources) * (1 - validation_split))), 4
        ), dtype="float16")
    else:
        answers_array = np.zeros((
            int(np.floor(len(data_sources)*(1-validation_split))), 1, image_size, image_size, 1
        ), dtype="float16")
    questions_array_valid = np.zeros((
        int(np.ceil(len(data_sources)*validation_split)), frames, image_size, image_size, 3
    ), dtype="float16")
    if easy_mode:
        answers_array_valid = np.zeros((
            int(np.ceil(len(data_sources) * validation_split)), 4
        ), dtype="float16")
    else:
        answers_array_valid = np.zeros((
            int(np.ceil(len(data_sources) * validation_split)), 1, image_size, image_size, 1
        ), dtype="float16")
    print("Getting training data with shapes:")
    print(np.shape(questions_array))
    print(np.shape(answers_array))
    print(np.shape(questions_array_valid))
    print(np.shape(answers_array_valid))

    accessed_index_q = []
    accessed_index_qv = []
    accessed_index_a = []
    accessed_index_av = []

    pbar = tqdm(total=len(data_sources))
    print(int(1.0/validation_split))
    for index, file in enumerate(data_sources):
        pbar.update(1)
        if index % int(1.0/validation_split) != 0:
            for frame in range(0, frames * timestep, timestep):
                target_file = "{}/data_{}.npy".format(refs[index][0], refs[index][1] + frame)
                array_index = index-1*int(np.floor(index*validation_split) + 1)
                accessed_index_q.append(array_index)
                try:
                    location = data_sources.index(target_file)
                    questions_array[array_index, int(frame / timestep), :, :, :] = source_array[location, :, :, :]
                except:
                    questions_array[array_index, int(frame / timestep), :, :, :] = process_bmp(target_file, image_size)
            for frame in range(frames * timestep, (frames + 1) * timestep, timestep):
                target_file = "{}/data_{}.npy".format(refs[index][0], refs[index][1] + frame)
                array_index = index - 1 * int(np.floor(index * validation_split) + 1)
                accessed_index_a.append(array_index)
                try:
                    if easy_mode:
                        answers_array[array_index, :] = get_sim_result(refs[index][0])
                    else:
                        location = data_sources.index(target_file)
                        answers_array[array_index, int(frame / timestep) - frames, :, :, 0] = source_array[location, :, :, 1]
                except:
                    answers_array[array_index, int(frame / timestep) - frames, :, :, 0] = process_bmp(target_file, image_size)[:, :, 1]
        else:
            for frame in range(0, frames * timestep, timestep):
                target_file = "{}/data_{}.npy".format(refs[index][0], refs[index][1] + frame)
                array_index = int(index*validation_split)
                accessed_index_qv.append(array_index)
                try:
                    location = data_sources.index(target_file)
                    questions_array_valid[array_index, int(frame / timestep), :, :, :] = source_array[location, :, :, :]
                except:
                    questions_array_valid[array_index, int(frame / timestep), :, :, :] = process_bmp(target_file, image_size)
            for frame in range(frames * timestep, (frames + 1) * timestep, timestep):
                target_file = "{}/data_{}.npy".format(refs[index][0], refs[index][1] + frame)
                array_index = int(index*validation_split)
                accessed_index_av.append(array_index)
                try:
                    if easy_mode:
                        answers_array_valid[array_index, :] = get_sim_result(refs[index][0])
                    else:
                        location = data_sources.index(target_file)
                        answers_array_valid[array_index, int(frame / timestep) - frames, :, :, 0] = source_array[location, :, :, 1]
                except:
                    answers_array_valid[array_index, int(frame / timestep) - frames, :, :, 0] = process_bmp(target_file, image_size)[:, :, 1]

    pbar.close()
    print("Converting to datasets...")
    # normalisation_options = [
    #     np.max(answers_array[:, :, :, :, 0]),
    #     np.max(answers_array_valid[:, :, :, :, 0]),
    #     np.max(questions_array[:, :, :, :, 1]),
    #     np.max(questions_array_valid[:, :, :, :, 1])
    # ]
    if not easy_mode:
        answers_array_valid = np.reshape(answers_array_valid, (int(np.ceil(len(data_sources)*validation_split)), image_size, image_size, 1))
        answers_array = np.reshape(answers_array, (int(np.floor(len(data_sources)*(1-validation_split))), image_size, image_size, 1))
    print(np.shape(accessed_index_a))
    print(np.shape(accessed_index_av))
    # normalisation_best = max(normalisation_options)
    # print(normalisation_options)
    print(np.shape(questions_array_valid))
    print(np.shape(answers_array_valid))
    testing_data = tf.data.Dataset.from_tensor_slices((questions_array, answers_array))
    testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_data = tf.data.Dataset.from_tensor_slices((questions_array_valid, answers_array_valid))
    validation_data = validation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #
    if var:
        testing_data = tf.data.Dataset.from_tensor_slices(questions_array[:, 0, :, :, 1:2])
        testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        validation_data = tf.data.Dataset.from_tensor_slices(questions_array_valid[:, 0, :, :, 1:2])
        validation_data = validation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return [testing_data, validation_data]
    # return questions_array, answers_array


def process_bmp(filename, image_size):
    x, y = np.load(filename)
    h, x_edge, y_edge = np.histogram2d(
        x, y,
        range=[[-1, 1], [-1, 1]], bins=(image_size, image_size)
    )
    output_array = np.zeros((image_size, image_size, 3))
    h = np.tanh(0.05 * h)
    output_array[:, :, 1] = h
    output_array = generate_rail(output_array)
    return output_array
