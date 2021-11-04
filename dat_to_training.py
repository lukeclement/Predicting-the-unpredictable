import glob
import os
import numpy as np
import tensorflow as tf
from PIL import Image


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


def transform_to_numpy_array(x, y, variant, invert, image_size=64):
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
        x + variant / (image_size / 2), (-1**(not invert))*y,
        range=[[-1, 1], [-1, 1]], bins=(image_size, image_size)
    )
    # Preparing memory for the output array, then filling the bubble edge
    output_array = np.zeros((image_size, image_size, 3))
    output_array[:, :, 1] = np.minimum(h, np.zeros((image_size, image_size)) + 1)
    # Adding the central rail
    output_array = generate_rail(output_array)
    output_array = 255*output_array
    output_array = output_array.astype(np.uint8)
    return output_array


def convert_dat_files(variant_range, image_size=64):
    """Converts all .dat files to numpy arrays, and saves them as .npy files.
    These .npy files are stored in Simulation_images/Simulation_X, where X is the reference number for the simulation.
    These aren't necessarily actual simulations, but can be variants of these 'base' simulations,
    where the physics remains constant.
    Input:
        variant_range:  An array of two floats, defining the [minimum, maximum]
                            amount to shift the original images in the x-axis. This range is inclusive.
        image_size:     (default 64) An integer for the number of pixels to be used per axis.
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
                    print("File {}, flipped, shifted {} is now Simulation_{}".format(
                        simulation, variant, simulation_index + tracking_index * np.size(simulation_names)
                    ))
                else:
                    print("File {}, shifted {} is now Simulation_{}".format(
                        simulation, variant, simulation_index + tracking_index * np.size(simulation_names)
                    ))
                # Making a directory for the images
                try:
                    os.mkdir("Simulation_images/Simulation_{}".format(
                        simulation_index + tracking_index*np.size(simulation_names)
                    ))
                except OSError:
                    print("Folder already exists!")
                # Now the heavy lifting
                for file in dat_files:
                    # Extracting data
                    x, y = read_file(file)
                    # Finding the actual frame number
                    step_number = int(file[file.find("s_")+2:-4])
                    # Converting to array
                    resulting_array = transform_to_numpy_array(x, y, variant, inversion, image_size=image_size)
                    # Saving to memory
                    image = Image.fromarray(resulting_array)
                    image.save("Simulation_images/Simulation_{}/img_{}.bmp".format(
                        simulation_index + tracking_index*np.size(simulation_names), step_number
                    ))
                    del resulting_array
                tracking_index += 1
        simulation_index += 1


def create_training_data(frames, timestep):
    simulation_names = glob.glob("Simulation_images/*")
    print(simulation_names)
    data_sources = []
    refs = []
    for simulation in simulation_names:
        files = glob.glob("{}/*".format(simulation))
        number_of_files = len(files)
        for i in range(0, number_of_files-timestep*frames):
            data_sources.append("{}/img_{}.bmp".format(simulation, i))
            refs.append([simulation, i])

    questions = []
    answers = []
    for index, file in enumerate(data_sources):
        new_question_dataset = []
        for frame in range(0, frames*timestep, timestep):
            source = tf.io.read_file("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + frame))
            data = tf.io.decode_bmp(source)
            new_question_dataset.append(data)
        question = tf.stack(new_question_dataset)
        source = tf.io.read_file("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + frames*timestep))
        data = tf.io.decode_bmp(source)
        answer = data
        questions.append(question)
        answers.append(answer)
    questions_final = tf.stack(questions)
    answers_final = tf.stack(answers)
    return [questions_final, answers_final]