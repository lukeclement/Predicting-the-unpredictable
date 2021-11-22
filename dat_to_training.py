import glob
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.signal import convolve2d
import psutil


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
    multiply = 3
    kernel_size = 7
    h, x_edge, y_edge = np.histogram2d(
        ((-1)**(not invert))*x, y + variant / (image_size / 2),
        range=[[-1, 1], [-1, 1]], bins=(image_size*multiply + kernel_size - 1, image_size*multiply + kernel_size - 1)
    )
    kernel = np.ones((kernel_size, kernel_size))
    h = convolve2d(h, kernel, mode='valid')
    h = h[::multiply, ::multiply]
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


def create_training_data(frames, timestep, validation_split=0.1, image_size=64):
    print(tf.executing_eagerly())
    simulation_names = glob.glob("Simulation_images/*")
    data_sources = []
    refs = []
    print(len(simulation_names))
    total = 0
    sub_total = 0
    for simulation in simulation_names[:]:
        files = glob.glob("{}/*".format(simulation))
        number_of_files = len(files)
        sub_total += len(files)
        for i in range(3, number_of_files-timestep*frames):
            total += 1
            data_sources.append("{}/img_{}.bmp".format(simulation, i))
            refs.append([simulation, i])

    questions_array = np.zeros((
        int(np.floor(len(data_sources)*(1-validation_split))), frames, image_size, image_size, 3
    ), dtype="float16")
    answers_array = np.zeros((
        int(np.floor(len(data_sources)*(1-validation_split))), image_size, image_size, 1
    ), dtype="float16")
    questions_array_valid = np.zeros((
        int(np.ceil(len(data_sources)*validation_split)), frames, image_size, image_size, 3
    ), dtype="float16")
    answers_array_valid = np.zeros((
        int(np.ceil(len(data_sources)*validation_split)), image_size, image_size, 1
    ), dtype="float16")
    print("Running...")
    print(np.shape(questions_array))
    print(np.shape(answers_array))
    for index, file in enumerate(data_sources):
        if index % int(1.0/validation_split) == 0:
            for frame in range(0, frames * timestep, timestep):
                questions_array[index-1*int(np.floor(index*validation_split) + 1), int(frame / timestep), :, :, :] = np.asarray(
                    Image.open("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + frame))
                ) / 255
            answers_array[index-1*int(np.floor(index*validation_split) + 1), :, :, 0] = np.asarray(
                Image.open("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + timestep * frames))
            )[:, :, 1] / 255
        else:
            for frame in range(0, frames * timestep, timestep):
                questions_array_valid[int(index*validation_split), int(frame / timestep), :, :, :] = np.asarray(
                    Image.open("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + frame))
                ) / 255
            answers_array_valid[int(index*validation_split), :, :, 0] = np.asarray(
                Image.open("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + timestep * frames))
            )[:, :, 1] / 255

    print("Converting to datasets...")
    batch_size = 8
    testing_data = tf.data.Dataset.from_tensor_slices((questions_array, answers_array))
    testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    validation_data = tf.data.Dataset.from_tensor_slices((questions_array_valid, answers_array_valid))
    validation_data = validation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return [testing_data, validation_data]
    # return [questions_array, answers_array]

    # Previous tensorflow method.
    # questions = []
    # questions_valid = []
    # answers = []
    # answers_valid = []
    # validation_point = len(data_sources)/(1-validation_split)
    # for index, file in enumerate(data_sources):
    #     if index < validation_point:
    #         extract_bmp(answers, frames, index, questions, refs, timestep, image_size)
    #     else:
    #         extract_bmp(answers_valid, frames, index, questions_valid, refs, timestep, image_size)
    # questions_final = tf.data.Dataset.from_tensors(questions)
    # answers_final = tf.data.Dataset.from_tensors(answers)
    # questions_final_valid = tf.data.Dataset.from_tensors(questions_valid)
    # answers_final_valid = tf.data.Dataset.from_tensors(answers_valid)
    # # print(answers_final)
    # print(questions_final)
    # m_0 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    # print(tf.data.Dataset.zip((questions_final, answers_final)))
    # m_1 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    # print(m_1 - m_0)
    # return [
    #     tf.data.Dataset.zip((questions_final, answers_final)),
    #     tf.data.Dataset.zip((questions_final_valid, answers_final_valid))
    # ]


def extract_bmp(answers, frames, index, questions, refs, timestep, image_size):
    new_question_dataset = []
    for frame in range(0, frames * timestep, timestep):
        source = tf.io.read_file("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + frame))
        data = tf.cast(tf.io.decode_bmp(source), tf.float16)
        new_question_dataset.append(data / 255)
    question = tf.stack(new_question_dataset)
    source = tf.io.read_file("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + frames * timestep))
    data = tf.cast(tf.io.decode_bmp(source), tf.float16) / 255
    answer = [data[:, :, 2]]
    questions.append(question)
    answers.append(tf.reshape(answer, [image_size, image_size, 1]))
