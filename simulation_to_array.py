import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def read_file(file_path):
    """Takes in a filepath and extracts a set of x and y coordanates of the bubble edge.
    Input: file path
    Output: A pair of 1D arrays (x and y)
    """

    x = []
    y = []
    file = open(file_path, "r")
    main_data = False
    for line in file:
        if "boundary4" in line:
            main_data = True
        if main_data and not "ZONE" in line:
            data_points = line.strip().split(" ")
            x.append(float(data_points[0]))
            y.append(float(data_points[1]))
    return x, y


def transform_into_array(x, y, size=128):
    """Transforms a set of x and y coordanates to an array of the bubble.
    Inputs:
        x:      a 1d array of x coordanates
        y:      a 1d array of y coordinates
        size:   an integer for the number of pixels
                to be used in a single axis (defult 128)
    Output:
        A 2d array of shape (size, size) showing the edges of the bubble plus any overlap in pixels
    """
    h, x_edge, y_edge, image = plt.hist2d(x, y, range=[[-1,1], [-1,1]], bins=(size,size))
    news = []
    max_val = np.max(h)
    for i in range(0,np.size(h[0])):
        li = []
        for j in range(0,np.size(h[0])):
            temp = h[j][np.size(h[0]) - 1 - i]
            #colour = int((float(temp)/float(max_val))*255.0)
            if temp >= 1:
                colour = 1
            else:
                colour = 0
            li.append([colour])
        news.append(li)
    return news


def main():
    """Extracting data from the simulation and making a series of arrays
    """
    sim_names = glob.glob("Simulation_data/*")
    #print(sim_names)
    for sim_number in range(0, np.size(sim_names)):
        files = glob.glob("{}/b*.dat".format(sim_names[sim_number]))
        print("Running ({} files found in simulation {})".format(np.size(files),sim_names[sim_number]))
        try:
            os.mkdir("Simulation_images/Simulation_{}".format(sim_number))
        except:
            print("")
        for file in files:
            x, y = read_file(file)
            step_number = int(file[file.find("s_")+2:-4])
            np.save("Simulation_images/Simulation_{}/img_{}".format(sim_number,step_number), transform_into_array(x, y, size=64))
    
if __name__ == "__main__":
    main()
