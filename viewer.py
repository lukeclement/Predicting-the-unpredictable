import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

"""for i in range(0,16):
    images = []
    simulation = "Simulation_{}".format(i)
    print("Working on {}".format(simulation))
    end_of_directory = False
    index = 0
    missed = 0
    while not end_of_directory:
        try:
            img = np.load("Simulation_images/{}/img_{}.npy".format(simulation, index))
            plt.imshow(img, cmap=plt.get_cmap('Greys'))
            plt.text(5,5,"{}".format(index))
            plt.savefig("Temp_data/{}_{}.png".format(simulation,index))
            images.append("Temp_data/{}_{}.png".format(simulation,index))
            plt.clf()
            missed = 0
        except:
            missed += 1
            if missed > 200:
                end_of_directory = True
        index += 1
        
    #filenames = []
    #for i in range(0,100):
    #    filenames.append("Machine_predictions/{}.png".format(i))
    files = []
    for filename in images:
        files.append(imageio.imread(filename))
    imageio.mimsave('Temp_data/{}.gif'.format(simulation), files)
   
"""
all = np.load("Answers.npy")
print(np.shape(all))
img = all[44498]
#img = np.load("Simulation_images/Simulation_0/img_700.npy")
plt.imshow(img)
plt.show()
