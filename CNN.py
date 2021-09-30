import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import glob
import matplotlib.pyplot as plt
    
def get_source_arrays(files):
    """Get the arrays from simulated data.
    Input:
        files:              list of files (list of strings)
    Output:
        training_images:    source files of training images
            a 2d array containing training set and solutions
    """
    number_of_steps = np.size(files)
    training_questions = []
    training_solutions = []
    for file in files:
        step_number = int(file[file.find("/")+1:-4])
        if step_number+5 < number_of_steps:
            source_array = np.load(file)
            #Normalisation
            training_questions.append(source_array/255.0)
            
            source_array = np.load("Simulation_images/{}.npy".format(step_number+5))
            #Normalisation
            training_solutions.append(source_array/255.0)
                        
    training_questions = np.stack([x.tolist() for x in training_questions])
    training_solutions = np.stack([x.tolist() for x in training_solutions])
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
    model.add(layers.Conv2D(32, (3, 3), activation=activation, input_shape=(size, size, 1)))
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
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
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
    training_data = get_source_arrays(files)
    
    print("Creating CNN...")
    active='tanh'
    optimizer='adam'
    loss = losses.BinaryCrossentropy()
    model = create_neural_net(active, optimizer, loss)
    
    print("Training montage begins...")
    model, history = train_model(model, training_data, epochs=10)
    
    print("Diagnosing...")
    out = model(training_data[0][0:1])
    plt.imshow(out[0]*255.0)
    plt.savefig("Machine.png")
    plt.show()
    plt.imshow(training_data[0][0])
    plt.savefig("First.png")
    plt.imshow(training_data[1][0])
    plt.savefig("Second.png")
    plt.clf()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig("Loss.png")
    #test_loss, test_acc = model.evaluate(test_set, test_solutions, verbose=2)
    #plt.show()

if __name__ == "__main__":
    main()
