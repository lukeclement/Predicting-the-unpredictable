import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.signal import convolve2d
import psutil
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
        if i < image_size:
            rail = i / (image_size)
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


def convert_dat_files(variant_range):
    """Converts all .dat files to numpy arrays, and saves them as .bmp files.
    These .bmp files are stored in Simulation_images/Simulation_X, where X is the reference number for the simulation.
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
                    resulting_array = transform_to_numpy_array(
                        x, y, variant, inversion
                    )
                    # Saving to memory
                    image = Image.fromarray(resulting_array)
                    image.save("Simulation_images/Simulation_{}/img_{}.bmp".format(
                        simulation_index + tracking_index*np.size(simulation_names), step_number
                    ))
                    del resulting_array
                tracking_index += 1
        simulation_index += 1


def load_training_data(frames, timestep, validation_split=0.001, image_size=64, focus=1, simulation_num=999, numpy_=False):
    refs = []
    data_sources = []
    sub_total = 0
    total = 0
    print("Loading training data: ")
    training_names = glob.glob("training_images/*")
    training_names = training_names[:6]
    if simulation_num != 999:
        simulation = training_names[simulation_num]
        print("Loading simulation number: ", simulation)
        files = glob.glob("{}/*".format(simulation))
        number_of_files = len(files)
        sub_total += len(files)
        for i in range(3, number_of_files - timestep * frames * 2):
            total += 1
            data_sources.append("{}/img_{}.bmp".format(simulation, i))
            refs.append([simulation, i])
    else:
        for simulation in training_names:
            files = glob.glob("{}/*".format(simulation))
            number_of_files = len(files)
            sub_total += len(files)
            for i in range(3, number_of_files - timestep * frames * 2):
                total += 1
                data_sources.append("{}/img_{}.bmp".format(simulation, i))
                refs.append([simulation, i])

    source_array = np.zeros((len(data_sources), image_size, image_size, 3))
    index = 0
    pbar = tqdm(total=len(data_sources))
    if simulation_num != 999:
        training_directory = glob.glob(training_names[simulation_num] + "/*")
        number_of_files = len(training_directory)
        for i in range(3, number_of_files - timestep * frames * 2):
            training_data = np.load(training_directory[i])
            source_array[index, :, :, :] = training_data
            index += 1
            pbar.update(1)
    else:
        for simulation in training_names:
            training_directory = glob.glob(simulation+"/*")
            number_of_files = len(training_directory)
            for i in range(3, number_of_files - timestep * frames * 2):
                training_data = np.load(training_directory[i])
                source_array[index, :, :, :] = training_data
                index += 1
                pbar.update(1)
    pbar.close()

    questions_array = np.zeros((
        int(np.floor(len(data_sources)*(1-validation_split))), frames, image_size, image_size, 3
    ), dtype="float16")
    answers_array = np.zeros((
        int(np.floor(len(data_sources)*(1-validation_split))), frames, image_size, image_size, 1
    ), dtype="float16")
    questions_array_valid = np.zeros((
        int(np.ceil(len(data_sources)*validation_split)), frames, image_size, image_size, 3
    ), dtype="float16")
    answers_array_valid = np.zeros((
        int(np.ceil(len(data_sources)*validation_split)), frames, image_size, image_size, 1
    ), dtype="float16")
    print("Converting training data:")
    pbar = tqdm(total=len(data_sources))
    for index, file in enumerate(data_sources):
        pbar.update(1)
        if index % int(1.0/validation_split) != 0:
            for frame in range(0, frames * timestep, timestep):
                target_file = "{}/img_{}.npy".format(refs[index][0], refs[index][1] + frame)
                array_index = index-1*int(np.floor(index*validation_split) + 1)
                try:
                    location = data_sources.index(target_file)
                    questions_array[array_index, int(frame / timestep), :, :, :] = source_array[location, :, :, :]
                except:
                    questions_array[array_index, int(frame / timestep), :, :, :] = np.load(target_file)
            for frame in range(frames * timestep, frames * timestep * 2, timestep):
                target_file = "{}/img_{}.npy".format(refs[index][0], refs[index][1] + frame)
                array_index = index - 1 * int(np.floor(index * validation_split) + 1)
                try:
                    location = data_sources.index(target_file)
                    answers_array[array_index, int(frame / timestep) - frames, :, :, 0] = source_array[location, :, :, 1]
                except:
                    answers_array[array_index, int(frame / timestep) - frames, :, :, 0] = np.load(target_file)[:, :, 1]
        else:
            for frame in range(0, frames * timestep, timestep):
                target_file = "{}/img_{}.npy".format(refs[index][0], refs[index][1] + frame)
                array_index = int(index*validation_split)
                try:
                    location = data_sources.index(target_file)
                    questions_array_valid[array_index, int(frame / timestep), :, :, :] = source_array[location, :, :, :]
                except:
                    questions_array_valid[array_index, int(frame / timestep), :, :, :] = np.load(target_file)
            for frame in range(frames * timestep, frames * timestep * 2, timestep):
                target_file = "{}/img_{}.npy".format(refs[index][0], refs[index][1] + frame)
                array_index = int(index*validation_split)
                try:
                    location = data_sources.index(target_file)
                    answers_array_valid[array_index, int(frame / timestep) - frames, :, :, 0] = source_array[location, :, :, 1]
                except:
                    answers_array_valid[array_index, int(frame / timestep) - frames, :, :, 0] = np.load(target_file)[:, :, 1]
    pbar.close()
    if numpy_:
        return [[questions_array, answers_array], [questions_array_valid, answers_array_valid]]

    testing_data = tf.data.Dataset.from_tensor_slices((questions_array, answers_array))
    # testing_data = testing_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    validation_data = tf.data.Dataset.from_tensor_slices((questions_array_valid, answers_array_valid))
    # validation_data = validation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return [testing_data, validation_data]


def save_training_data(focus, image_size):
    print("Creating training data: ")
    simulation_names = glob.glob("Simulation_images/*")
    data_sources = []
    save_refs = []
    save_counter = 0
    total = 0
    sub_total = 0
    for simulation in simulation_names[:]:
        files = glob.glob("{}/*".format(simulation))
        number_of_files = len(files)
        sub_total += len(files)
        for i in range(0, number_of_files):
            total += 1
            data_sources.append("{}/img_{}.bmp".format(simulation, i))
            save_refs.append([save_counter, i])
        save_counter += 1
    source_array = np.zeros((len(data_sources), image_size, image_size, 3))
    pbar = tqdm(total=len(data_sources))
    for index, data in enumerate(data_sources):
        pbar.update(1)
        source_array[index, :, :, :] = process_bmp(data, image_size, focus=focus)
    pbar.close()
    refs_length = len(save_refs)
    print("Saving training data: ")
    pbar = tqdm(total=len(data_sources))
    for i in range(refs_length):
        pbar.update(1)
        simulation_number = save_refs[i][0]
        image_number = save_refs[i][1]
        image = source_array[i, :, :, :]
        try:
            np.save("training_images/Simulation_{}/img_{}.npy".format(simulation_number, image_number), image)
        except:
            os.mkdir("training_images/Simulation_{}".format(simulation_number))
            np.save("training_images/Simulation_{}/img_{}.npy".format(simulation_number, image_number), image)

    pbar.close()


def print_progress(pos, total):
    buffer = ""
    for i in range(0, 20):
        if float(i)/20. > pos / float(total):
            buffer += "|"
        else:
            buffer += "="
    print("hi")
    print("{}{:.2f}%".format(buffer, pos * 100 / float(total)), end="\r")


def process_bmp(filename, image_size, focus=1):
    h = np.asarray(Image.open(filename)) / 255
    kernel_size = int((BASE_SIZE/image_size)*focus)
    kernel = np.ones((kernel_size, kernel_size))
    h = convolve2d(h, kernel, mode='same')
    h = h[::int(BASE_SIZE / image_size), ::int(BASE_SIZE / image_size)]
    output_array = np.zeros((image_size, image_size, 3))
    h = np.tanh(0.5 * h)
    output_array[:, :, 1] = h
    output_array = generate_rail(output_array)
    return output_array


def create_multiframe_data(leading_steps, loss_steps, timestep, validation_split=0.1, image_size=64):
    simulation_names = glob.glob("Simulation_images/*")
    print(simulation_names)
    data_sources = []
    refs = []
    for simulation in simulation_names[:1]:
        files = glob.glob("{}/*".format(simulation))
        number_of_files = len(files)
        for i in range(5, number_of_files - timestep * (leading_steps + loss_steps)):
            data_sources.append("{}/img_{}.bmp".format(simulation, i))
            refs.append([simulation, i])

    print("Generating arrays of size {}...".format(len(data_sources)))
    questions_array = np.zeros((len(data_sources), leading_steps, image_size, image_size, 3), dtype="float16")
    answers_array = np.zeros((len(data_sources), loss_steps, image_size, image_size, 3), dtype="float16")
    print("Running...")
    print(np.shape(questions_array))
    print(np.shape(answers_array))
    for index, file in enumerate(data_sources):
        for frame in range(0, leading_steps * timestep, timestep):
            question = np.asarray(
                Image.open(
                    "{}/img_{}.bmp".format(
                        refs[index][0], refs[index][1] + frame
                    )
                )
            ) / 255
            questions_array[index, int(frame / timestep), :, :, :] = question

        for frame in range(0, loss_steps * timestep, timestep):
            answer = np.asarray(
                Image.open(
                    "{}/img_{}.bmp".format(
                        refs[index][0], refs[index][1] + timestep * leading_steps + frame
                    )
                )
            ) / 255
            answers_array[index, int(frame / timestep), :, :, :] = answer
    print("Saving...")
    # questions_array = tf.data.Dataset.from_tensor_slices((questions_array, answers_array)).batch(32).prefetch(buffer_size=1000)
    return [questions_array, answers_array]


def create_small_multiframe_data(leading_steps, loss_steps, timestep, validation_split=0.1, image_size=64):
    simulation_names = glob.glob("Simulation_images/*")
    print(simulation_names)
    data_sources = []
    refs = []
    for simulation in simulation_names[:1]:
        files = glob.glob("{}/*".format(simulation))
        number_of_files = len(files)
        for i in range(5, number_of_files - timestep * (leading_steps + loss_steps)):
            data_sources.append("{}/img_{}.bmp".format(simulation, i))
            refs.append([simulation, i])

    print("Generating arrays of size {}...".format(len(data_sources)))
    questions_array = np.zeros((len(data_sources), leading_steps, image_size, image_size, 1), dtype="float32")
    answers_array = np.zeros((len(data_sources), loss_steps, image_size, image_size, 1), dtype="float32")
    print("Running...")
    print(np.shape(questions_array))
    print(np.shape(answers_array))
    for index, file in enumerate(data_sources):
        for frame in range(0, leading_steps * timestep, timestep):
            question = np.asarray(
                Image.open(
                    "{}/img_{}.bmp".format(
                        refs[index][0], refs[index][1] + frame
                    )
                )
            ) / 255
            questions_array[index, int(frame / timestep), :, :, 0] = question[:, :, 1]

        for frame in range(0, loss_steps * timestep, timestep):
            answer = np.asarray(
                Image.open(
                    "{}/img_{}.bmp".format(
                        refs[index][0], refs[index][1] + timestep * leading_steps + frame
                    )
                )
            ) / 255
            answers_array[index, int(frame / timestep), :, :, 0] = answer[:, :, 1]
    print("Saving...")
    # questions_array = tf.data.Dataset.from_tensor_slices((questions_array, answers_array)).batch(32)
    return [questions_array, answers_array]


def create_small_training_data(frames, timestep, validation_split=0.1, image_size=64):
    simulation_names = glob.glob("Simulation_images/*")
    print(simulation_names)
    data_sources = []
    refs = []
    for simulation in simulation_names[:2]:
        files = glob.glob("{}/*".format(simulation))
        number_of_files = len(files)
        for i in range(5, number_of_files-timestep*frames):
            data_sources.append("{}/img_{}.bmp".format(simulation, i))
            refs.append([simulation, i])

    print("Generating arrays of size {}...".format(len(data_sources)))
    questions_array = np.zeros((len(data_sources), frames, image_size, image_size, 1), dtype="float16")
    answers_array = np.zeros((len(data_sources), image_size, image_size, 1), dtype="float16")
    print("Running...")
    print(np.shape(questions_array))
    print(np.shape(answers_array))
    for index, file in enumerate(data_sources):
        for frame in range(0, frames * timestep, timestep):
            questions_array[index, int(frame / timestep), :, :, :] = np.asarray(
                Image.open("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + frame))
            )[:, :, 1:2] / 255
        answers_array[index, :, :, 0] = np.asarray(
            Image.open("{}/img_{}.bmp".format(refs[index][0], refs[index][1] + timestep * frames))
        )[:, :, 1] / 255
    print("Saving...")

    questions_array = [questions_array, answers_array]

    return questions_array




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
    questions.append(new_question_dataset)
    answers.append(tf.reshape(answer, [image_size, image_size, 1]))
