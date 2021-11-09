import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import imageio
import Generate_sources
import os


def iou_coef(y_true, y_pred, smooth=1):
    # print(y_true)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def bce_dice(y_true, y_pred):
    bce = losses.binary_crossentropy(y_true, y_pred)
    # bce = losses.MeanSquaredError(y_true, y_pred)
    di = K.log(dice_coef(y_true, y_pred))
    iou = K.log(iou_coef(y_true, y_pred))
    com = K.log(com_coef(y_true, y_pred))
    # return bce - di - iou + com
    return bce - di - iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


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
    return K.abs(difference)


def mass_preservation(y_true, y_pred, smooth=1):
    true_mass = K.sum(y_true)
    pred_mass = K.sum(y_pred)
    return K.exp(-K.sqrt(K.abs(true_mass - pred_mass)) / 2)


def calculate_com(bubble_image):
    size = np.size(bubble_image[0]) / 3
    total_mass = np.sum(bubble_image)
    x = 0
    y = 0
    for i in range(0, size):
        for j in range(0, size):
            x += bubble_image[j][i][1] * i
            y += bubble_image[j][i][1] * j
    x = x / total_mass
    y = y / total_mass
    return [x, y]


def get_source_arrays(sims, timestep_size=5, frames=4):
    """Get the arrays from simulated data.
    Input:
        sims:              list of simulations (list of strings)
        timestep_size:      int of the timestep to use (1 is minimum, default is 5)
    Output:
        training_images:    source files of training images
            a 2d array containing training set and solutions
    """
    training_questions = []
    training_solutions = []
    in_use = 0
    for sim in sims:
        print("Running {}...".format(sim))
        files = glob.glob("{}/*.npy".format(sim))
        number_of_steps = np.size(files)
        for file in files:
            # print("{:.1f}%...".format(in_use*100/(np.size(files)*np.size(sims))))
            try:
                loc = file.find("/img_") + 5
                step_number = int(file[loc:-4])
                if step_number + timestep_size < number_of_steps and step_number - frames > 0:
                    in_use += 1
                    start_array = []
                    for i in range(frames, 0, -1):
                        source_array = np.load("{}/img_{}.npy".format(sim, step_number - i))
                        start_array.append(source_array)
                    # Normalisation

                    source_array = np.load("{}/img_{}.npy".format(sim, step_number + timestep_size))
                    # Normalisation
                    training_questions.append(np.stack([x.tolist() for x in start_array]))
                    training_solutions.append(source_array)
            except Exception as e:
                print("Missed on {}:".format(file))
                print(e)
    print("Using {} images".format(in_use))
    print("Processing questions...")
    training_questions = np.stack([x.tolist() for x in training_questions])
    print("Processing answers...")
    training_solutions = np.stack([x.tolist() for x in training_solutions])
    print(np.shape(training_questions))
    print(np.shape(training_solutions))
    return [training_questions, training_solutions]


def inception_cell(model, activation, axis):
    shape = model.output_shape
    li = list(shape)
    li.pop(0)
    shape = tuple(li)
    input_tower = layers.Input(shape=shape)

    tower_1 = layers.Conv2D(32, (1, 1), padding='same', activation=activation)(input_tower)

    tower_2 = layers.Conv2D(32, (1, 1), padding='same', activation=activation)(input_tower)
    tower_2 = layers.Conv2D(32, (3, 3), padding='same', activation=activation)(tower_2)

    tower_3 = layers.Conv2D(32, (1, 1), padding='same', activation=activation)(input_tower)
    tower_3 = layers.Conv2D(32, (5, 5), padding='same', activation=activation)(tower_3)

    # tower_4 = layers.MaxPooling2D((3, 3), strides=1)(input_tower)
    tower_4 = layers.Conv2D(32, (3, 3), padding='same', activation=activation)(input_tower)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)
    model.add(Model(input_tower, merged))

    return model

def create_neural_net(activation, optimizer, loss, frames=4, size=128, channels=3):
    """Creates the CNN.
    Inputs:
        activation: The activation function used on the neurons (string)
        optimizer:  The optimisation function used on the model (string)
        loss:       The loss function to be used in training    (function)
        size:       The size of the image, defaults to 128      (int)
    Output:
        The model, ready to be fitted!
    """

    model = models.Sequential()
    model.add(layers.Conv3D(32, (4, 7, 7), activation=activation, input_shape=(frames, size, size, channels)))
    model.add(layers.Reshape((58, 58, 32)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.BatchNormalization())
    model = inception_cell(model, activation=activation, axis=1)
    model = inception_cell(model, activation=activation, axis=2)

    model.add(layers.Conv2D(32, (1, 1), activation=activation))
    model.add(layers.Conv2D(32, (3, 3), activation=activation))

    model.add(layers.Conv2D(32, (5, 5), activation=activation, strides=(5, 5)))
    model = inception_cell(model, activation=activation, axis=1)
    model = inception_cell(model, activation=activation, axis=2)
    model.add(layers.Conv2DTranspose(32, (6, 6), activation=activation))
    model.add(layers.Conv2DTranspose(32, (6, 6), activation=activation))
    model.add(layers.Conv2D(1, (3, 3), activation=activation))


    # model = models.Sequential()
    # model.add(layers.Conv3D(32, (2, 2, 2), activation=activation, input_shape=(frames, size, size, channels)))
    # for frame in range(1, frames-1):
    #     model.add(layers.Conv3D(32, (2, 1, 1), activation=activation))
    # model.add(layers.Reshape((63, 63, 32)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation=activation))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation=activation))
    # model.add(layers.Conv2DTranspose(128, (3, 3), activation=activation))
    # model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.Conv2DTranspose(64, (4, 4), activation=activation))
    # model.add(layers.UpSampling2D((2, 2)))
    # model.add(layers.Conv2DTranspose(32, (3, 3), activation=activation))
    # model.add(layers.Conv2DTranspose(1, (1, 1), activation='sigmoid'))
    # model.add(layers.Conv2DTranspose(1, (1, 1), activation=activation))

    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=[iou_coef, dice_coef, mass_preservation, com_coef])
    # model.compile(optimizer=optimizer, loss=loss, metrics=[mass_preservation])
    return model


def train_model(model, training_images, validation_split=0.1, epochs=20):
    """Trains the model. This can take a while!
    Inputs:
        model:              The model
        training_images:    A 2d array of training images
        validation_split:   The split of training vs testing data, default is 0.1
        epochs:             The number of evolutions to perform
    Outputs:
        model:              The fitted model
        history:            The history of improvements
    """
    questions = training_images[0]
    answers = training_images[1]
    history = model.fit(questions, answers, epochs=epochs, validation_split=validation_split, shuffle=True)

    return model, history


def predict_future_2(model, timestep_size, simulation_number, start_number,
                     frames=4, size=128, channels=3, predicition_length=100, name="Model_test"):
    input_frames = np.zeros((1, frames, size, size, channels))
    for frame in range(0, frames):
        input_frames[0, frame, :, :, :] = np.load("Simulation_images/Simulation_{}/img_{}.npy".format(
            simulation_number, start_number + frame * timestep_size
        ))
    generated_frames = []
    try:
        os.mkdir("Machine/{}".format(name))
    except OSError:
        print("Folder exists!")
    for prediction in range(0, predicition_length):
        output_frame = model(input_frames)
        for frame in range(0, frames):
            if frame != frames - 1:
                input_frames[0, frame, :, :, :] = input_frames[0, frame + 1, :, :, :]
            else:
                input_frames[0, frame, :, :, 1] = np.around(output_frame[0, :, :, 0])
                # input_frames[0, frame, :, :, 1] = output_frame[0, :, :, 0]
                for i in range(0, size):
                    if i < size / 2:
                        rail = i / (size / 2)
                    else:
                        rail = 2 - i / (size / 2)
                    runway = np.zeros(size) + rail
                    input_frames[0, frame, i, :, 2] = runway
        plt.imshow(input_frames[0, frames-1])
        plt.savefig("Machine/{}/Test_Predict_{}.png".format(name, prediction))
        plt.close()
        # test_img = np.load("Simulation_images/Simulation_{}/img_{}.npy".format(
        #    simulation_number, start_number + (prediction+frames) * timestep_size
        # ))
        # plt.imshow(test_img)
        # plt.savefig("Machine/Test_Actual_{}.png".format(prediction))
    return 0


def predict_future(model, start_image_number, sim, number_of_steps, timestep_size, name, frames=4):
    initial = []
    for f in range(0, frames):
        temp = np.load("Simulation_images/{}/img_{}.npy".format(sim, start_image_number + frames * timestep_size))
        initial.append(temp)
    current_frames = np.stack([x.tolist() for x in initial])
    plt.imshow(current_frames[0])
    plt.savefig("Machine_predictions/setup.png")
    saved_names = []
    comparison_names = []
    distances = []
    for i in range(0, number_of_steps):
        current_imgs = model(current_imgs)
        plt.imshow(current_imgs[0], cmap=plt.get_cmap(name))
        plt.savefig("Machine_predictions/{}.png".format(i))
        saved_names.append("Machine_predictions/{}.png".format(i))
        try:
            actual = np.load(
                "Simulation_images/{}/img_{}.npy".format(sim, start_image_number + (i + 1) * timestep_size))
            # Centre of mass difference
            # Shape difference? Similar to chi squared? But centred in mid image?
            machine_guess = np.asarray(current_imgs[0])
            # plt.imshow(actual)
            plt.imshow(machine_guess)
            guess_com = calculate_com(machine_guess)
            actual_com = calculate_com(actual)
            difference = np.asarray(guess_com) - np.asarray(actual_com)
            distances.append(np.sqrt(difference[0] ** 2 + difference[1] ** 2))
            plt.scatter(guess_com[0], guess_com[1], label="Prediction COM")
            plt.scatter(actual_com[0], actual_com[1], label="Actual COM")
            plt.legend(loc='lower right')
            plt.savefig("Machine_predictions/Compararison_{}.png".format(i))
            comparison_names.append("Machine_predictions/Compararison_{}.png".format(i))
        except Exception as e:
            print(e)

        plt.clf()

        # current_imgs = K.round(current_imgs)
    plt.plot(distances)
    plt.ylabel("Distance of COM")
    plt.xlabel("Number of steps")
    plt.savefig("Machine_predictions/COM_distances.png")
    make_gif(saved_names, "Current_Guess")
    make_gif(comparison_names, "Comparison")


def make_gif(filenames, name):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('Machine_predictions/{}.gif'.format(name), images)


def run_metaparameter_tests(files):
    loss_results = []
    frame_numbers = []
    timestep_sizes = []
    active = 'LeakyReLU'
    optimizer = 'adam'
    #loss = bce_dice
    loss = losses.binary_crossentropy
    for j in range(3, 9):
        for i in range(2, 7):
            timestep_sizes.append(i)
            frame_numbers.append(j)
            training_data = Generate_sources.get_source_arrays_2(files, timestep_size=i, frames=j, size=64, channels=3)
            # training_data = [np.load("Questions.npy"), np.load("Answers.npy")]
            model = create_neural_net(active, optimizer, loss, size=64, frames=j)
            model, history = train_model(model, training_data, epochs=1)
            del training_data[:]
            loss_results.append(history.history['loss'][0])
            for k in range(0, 500, 50):
                print(k)
                predict_future_2(model, i, 48, k, size=64, frames=j, name="{}_{}_{}".format(i, j, k))
                plt.close('all')
            print(loss_results)
            print(frame_numbers)
            print(timestep_sizes)
    plt.hist2d(frame_numbers, timestep_sizes, weights=loss_results, bins=(19, 9))
    plt.savefig("Overall_picture.png")
    plt.close()


def main():
    files = glob.glob("Simulation_images/*")
    run_metaparameter_tests(files)
    exit()
    active = 'LeakyReLU'
    optimizer = 'adam'
    # loss = losses.BinaryCrossentropy()
    loss = bce_dice
    # loss = com_coef
    # loss = losses.CategoricalCrossentropy()
    # loss = losses.KLDivergence()
    # loss = losses.CosineSimilarity()#Works goodish
    # loss = losses.Hinge()
    # loss = losses.SquaredHinge()
    # loss = losses.MeanSquaredError() #Semi-Okay
    # loss = losses.MeanAbsoluteError()
    # loss = losses.MeanAbsolutePercentageError()
    # loss = losses.MeanSquaredLogarithmicError()
    # loss = losses.LogCosh()
    # loss = losses.Huber()
    timestep_size = 20
    print("Getting source files...")
    training_data = [np.load("Questions.npy"), np.load("Answers.npy")]
    print(np.shape(training_data[0]))
    print("Do you want to generate a new model? [Y/N]")
    choice = input(">>")
    if choice == "Y":
        # training_data = get_source_arrays(files[:], timestep_size)
        # np.save("Qs", training_data[0])
        # np.save("As", training_data[1])
        print("Creating CNN...")
        model = create_neural_net(active, optimizer, loss, size=64)

        print("Training montage begins...")
        model, history = train_model(model, training_data, epochs=1)
        model.save("Model_{}_{}_{}_{}".format(active, optimizer, "BinaryCrossEntropy", timestep_size))
    else:
        model = models.load_model("Model_{}_{}_{}_{}".format(active, optimizer, "BinaryCrossEntropy", timestep_size),
                                  custom_objects={"iou_coef": iou_coef, "dice_coef": dice_coef,
                                                  "mass_preservation": mass_preservation}, compile=False)

    print("Diagnosing...")
    out = model(training_data[0][0:1])
    # plt.imshow(out[0]*255.0)
    name = 'Greys'
    plt.imshow(out[0])
    plt.savefig("Machine.png")
    # plt.show()
    plt.imshow(training_data[0][0][3])
    plt.savefig("First.png")
    plt.imshow(training_data[1][0])
    plt.savefig("Second.png")
    plt.clf()
    print("Getting metrics info...")
    # plt.plot(history.history['dice_coef'], label='dice_coef')
    # plt.plot(history.history['val_dice_coef'], label='val_dice_coef')
    # plt.plot(history.history['iou_coef'], label='iou_coef')
    # plt.plot(history.history['val_iou_coef'], label='val_iou_coef')
    # plt.plot(history.history['mass_preservation'], label='mass_preservation')
    # plt.plot(history.history['val_mass_preservation'], label='val_mass_preservation')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig("Metrics.png")
    # test_loss, test_acc = model.evaluate(test_set, test_solutions, verbose=2)
    # plt.show()
    plt.clf()
    # Max is 820
    starting = 200
    print("Performing predictions...")
    start_sim = "Simulation_10"
    max_sim_num = np.size(glob.glob("Simulation_images/{}/*".format(start_sim)))
    max_steps = int((max_sim_num - starting) / timestep_size)
    predict_future_2(model, timestep_size, 12, 10, size=64)


if __name__ == "__main__":
    main()
