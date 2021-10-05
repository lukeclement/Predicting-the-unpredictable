import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import imageio

def iou_coef(y_true, y_pred, smooth=1):
    #print(y_true)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
  
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def com_coef(y_true, y_pred):
    actual = calculate_com(y_true.eval(session=tf.compat.v1.Session()))
    predict = calculate_com(y_pred.eval(session=tf.compat.v1.Session()))
    difference = predict - actual
    radius_sq = difference[0]**2 + difference[1]**2
    return np.exp(radius_sq)

def mass_preservation(y_true, y_pred, smooth=1):
    true_mass = K.sum(y_true)
    pred_mass = K.sum(y_pred)
    return K.exp(-K.sqrt(K.abs(true_mass - pred_mass))/2)
    
def calculate_com(bubble_image):
    size = np.size(bubble_image[0])
    total_mass = np.sum(bubble_image)
    x = 0
    y = 0
    for i in range(0,size):
        for j in range(0,size):
            x += bubble_image[j][i] * i
            y += bubble_image[j][i] * j
    x = x/total_mass
    y = y/total_mass
    return [x, y]

def get_source_arrays(sims, timestep_size=5):
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
        files = glob.glob("{}/*.npy".format(sim))
        number_of_steps = np.size(files)
        for file in files:
            adding_question = np.array([])
            adding_solution = np.array([])
            try:
                loc = file.find("/img_")+5
                step_number = int(file[loc:-4])
                if step_number + timestep_size < number_of_steps:
                    in_use += 1
                    source_array = np.load(file)
                    #Normalisation
                    #training_questions.append(source_array/255.0)
                    adding_question = source_array
                    
                    source_array = np.load("{}/img_{}.npy".format(sim,step_number + timestep_size))
                    #Normalisation
                    #training_solutions.append(source_array/255.0)
                    adding_solution = source_array
                if np.size(adding_question) != 0 and np.size(adding_solution) != 0:
                    training_solutions.append(adding_solution)
                    training_questions.append(adding_question)
            except:
                print("Missed on {}".format(file))
    print(np.shape(training_questions[0]))
    print(np.shape(training_solutions[0]))
    training_questions = np.stack([x.tolist() for x in training_questions])
    training_solutions = np.stack([x.tolist() for x in training_solutions])
    print("Using {} images".format(in_use))
    return [training_questions, training_solutions]
    
def create_neural_net(activation, optimizer, loss, size=128):
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
    model.add(layers.Conv2D(32, (3, 3), activation=activation, input_shape=(size, size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activation))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation=activation))
    model.add(layers.Conv2DTranspose(128, (3, 3), activation=activation))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(64, (4, 4), activation=activation))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(32, (3, 3), activation=activation))
    model.add(layers.Conv2DTranspose(3, (1, 1), activation='sigmoid'))
    
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=[iou_coef, dice_coef, mass_preservation])
    #model.compile(optimizer=optimizer, loss=loss, metrics=[mass_preservation])
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
    history = model.fit(questions, answers, epochs=epochs, validation_split=validation_split)

    return model, history

def predict_future(model, start_image_number, sim, number_of_steps, timestep_size, name):
    initial = np.load("Simulation_images/{}/img_{}.npy".format(sim,start_image_number))
    initial = [initial]
    current_imgs = np.stack([x.tolist() for x in initial])
    plt.imshow(current_imgs[0], cmap=plt.get_cmap(name))
    plt.savefig("Machine_predictions/setup.png")
    saved_names = []
    comparison_names = []
    distances = []
    for i in range(0,number_of_steps):
        current_imgs = model(current_imgs)
        plt.imshow(current_imgs[0], cmap=plt.get_cmap(name))
        plt.savefig("Machine_predictions/{}.png".format(i))
        saved_names.append("Machine_predictions/{}.png".format(i))
        try:
            actual = np.load("Simulation_images/{}/img_{}.npy".format(sim,start_image_number + (i+1)*timestep_size))
            #Centre of mass difference
            #Shape difference? Similar to chi squared? But centred in mid image?
            machine_guess = np.asarray(current_imgs[0])
            plt.imshow(actual, cmap=plt.get_cmap('Blues'))
            plt.imshow(machine_guess, cmap=plt.get_cmap('Reds'), alpha=0.5)
            guess_com = calculate_com(machine_guess)
            actual_com = calculate_com(actual)
            difference = np.asarray(guess_com) - np.asarray(actual_com)
            distances.append(np.sqrt(difference[0]**2 + difference[1]**2))
            plt.scatter(guess_com[0], guess_com[1], label="Prediction COM")
            plt.scatter(actual_com[0], actual_com[1], label="Actual COM")
            plt.legend(loc='lower right')
            plt.savefig("Machine_predictions/Compararison_{}.png".format(i))
            plt.clf()
            comparison_names.append("Machine_predictions/Compararison_{}.png".format(i))
        except:
            print("Fail")
        
        #current_imgs = K.round(current_imgs)
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
    

def main():
    print("Getting source files...")
    files = glob.glob("Simulation_images/*")
    print("Do you want to generate a new model? [Y/N]")
    active='LeakyReLU'
    optimizer='adam'
    loss = losses.BinaryCrossentropy()
    #loss = losses.MeanSquaredError()
    choice = input(">>")
    if choice == "Y":
        timestep_size = 20
        training_data = get_source_arrays(files[:], timestep_size)
        print("Creating CNN...")
        model = create_neural_net(active, optimizer, loss, size=64)
        
        print("Training montage begins...")
        model, history = train_model(model, training_data, epochs=10)
        model.save("Model_{}_{}_{}".format(active,optimizer,"BinaryCrossEntropy"))
    else:
        saved_model = models.load_model("Model_{}_{}_{}".format(active,optimizer,"BinaryCrossEntropy"))
    
    
    print("Diagnosing...")
    out = model(training_data[0][0:1])
    #plt.imshow(out[0]*255.0)
    name = 'Greys'
    plt.imshow(out[0], cmap=plt.get_cmap(name))
    plt.savefig("Machine.png")
    #plt.show()
    plt.imshow(training_data[0][0], cmap=plt.get_cmap(name))
    plt.savefig("First.png")
    plt.imshow(training_data[1][0], cmap=plt.get_cmap(name))
    plt.savefig("Second.png")
    plt.clf()
    print("Getting metrics info...")
    plt.plot(history.history['dice_coef'], label='dice_coef')
    plt.plot(history.history['val_dice_coef'], label = 'val_dice_coef')
    plt.plot(history.history['iou_coef'], label='iou_coef')
    plt.plot(history.history['val_iou_coef'], label = 'val_iou_coef')
    plt.plot(history.history['mass_preservation'], label='mass_preservation')
    plt.plot(history.history['val_mass_preservation'], label = 'val_mass_preservation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("Metrics.png")
    #test_loss, test_acc = model.evaluate(test_set, test_solutions, verbose=2)
    #plt.show()
    plt.clf()
    #Max is 820
    starting = 200
    print("Performing predictions...")
    start_sim = "Simulation_10"
    max_sim_num = np.size(glob.glob("Simulation_images/{}/*".format(start_sim)))
    max_steps = int((max_sim_num - starting)/timestep_size)
    predict_future(model, starting, start_sim, 100, timestep_size, name)
        

if __name__ == "__main__":
    main()
