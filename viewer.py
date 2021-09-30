import numpy as np
import matplotlib.pyplot as plt
import imageio

images = []
filenames = []
for i in range(0,100):
    filenames.append("Machine_predictions/{}.png".format(i))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('Machine_predictions/timestep_5_leaky.gif', images)
#img = np.load("Simulation_images/420.npy")

#plt.imshow(img)
#plt.show()
