import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


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

def get_source_arrays(files, timestep_size=5):
    """Get the arrays from simulated data.
    Input:
        files:              list of files (list of strings)
        timestep_size:      int of the timestep to use (1 is minimum, default is 5)
    Output:
        training_images:    source files of training images
            a 2d array containing training set and solutions
    """
    number_of_steps = np.size(files)
    training_questions = []
    training_solutions = []
    in_use = 0
    for file in files:
        loc = file.find("/")+1
        step_number = int(file[loc:-4])
        if step_number + timestep_size < number_of_steps:
            in_use += 1
            source_array = np.load(file)
            #Normalisation
            #training_questions.append(source_array/255.0)
            training_questions.append(source_array)
            
            source_array = np.load("Simulation_images/{}.npy".format(step_number + timestep_size))
            #Normalisation
            #training_solutions.append(source_array/255.0)
            training_solutions.append(source_array)
                        
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
    model.add(layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(size, size, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activation))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activation))
    model.add(layers.Conv2DTranspose(64, (3, 3), activation=activation))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(64, (3, 3), activation=activation))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(32, (3, 3), activation=activation))
    model.add(layers.Conv2DTranspose(1, (3, 3), activation='sigmoid'))
    
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=[iou_coef, dice_coef])
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


def main():
    print("Getting source files...")
    files = glob.glob("Simulation_images/*.npy")
    timestep_size = 5
    training_data = get_source_arrays(files[:], timestep_size)
    
    print("Creating CNN...")
    active='LeakyReLU'
    optimizer='adam'
    loss = losses.BinaryCrossentropy()
    model = create_neural_net(active, optimizer, loss, size=64)
    
    print("Training montage begins...")
    model, history = train_model(model, training_data, epochs=40)
    
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
    plt.plot(history.history['dice_coef'], label='dice_coef')
    plt.plot(history.history['val_dice_coef'], label = 'val_dice_coef')
    plt.plot(history.history['iou_coef'], label='iou_coef')
    plt.plot(history.history['val_iou_coef'], label = 'val_iou_coef')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig("Metrics.png")
    #test_loss, test_acc = model.evaluate(test_set, test_solutions, verbose=2)
    #plt.show()
    plt.clf()
    initial = np.load("Simulation_images/200.npy")
    initial = [initial]
    current_imgs = np.stack([x.tolist() for x in initial])
    plt.imshow(current_imgs[0], cmap=plt.get_cmap(name))
    plt.savefig("Machine_predictions/setup.png")
    for i in range(0,100):
        current_imgs = model(current_imgs)
        plt.imshow(current_imgs[0], cmap=plt.get_cmap(name))
        plt.savefig("Machine_predictions/{}.png".format(i))
        

if __name__ == "__main__":
    main()
