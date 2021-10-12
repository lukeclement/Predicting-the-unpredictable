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


def transform_into_array(x, y, varient, size=128):
    """Transforms a set of x and y coordanates to an array of the bubble.
    Inputs:
        x:      a 1d array of x coordanates
        y:      a 1d array of y coordinates
        varient:an integer between -1 and infinity to give the offset from the original image. If -1, image is flipped.
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
            if varient != -1:
                if varient + j < np.size(h[0]):
                    temp = h[j+varient][np.size(h[0]) - 1 - i]
                else:
                    temp = h[j][np.size(h[0]) - 1 - i]
            else:
                temp = h[j][i]
            
            #colour = int((float(temp)/float(max_val))*255.0)
            if temp >= 1:
                bubble = 1
            else:
                bubble = 0
            li.append(bubble)
        news.append(li)
    
    centre = calculate_com(news)
    final = []
    alpha = -np.size(news[0])*np.sqrt(2)/np.log(0.01)
    for i in range(0,np.size(news[0])):
        entry = []
        for j in range(0,np.size(news[0])):
            bubble_val = news[i][j]
            if i < np.size(h[0])/2:
                rail = i / (np.size(news[0])/2)
            else:
                rail = 2 - i / (np.size(news[0])/2)
            rail_val = rail
            distance = np.exp(-np.sqrt((centre[0]-i)**2 + (centre[1]-j)**2)/alpha)
            distance = 0
            #entry.append([distance, bubble_val, rail_val])
            entry.append([0 ,bubble_val, rail_val])
        final.append(entry)
    return final
    
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

def main():
    """Extracting data from the simulation and making a series of arrays
    """
    sim_names = glob.glob("Simulation_data/*")
    #print(sim_names)
    num_of_sims = np.size(sim_names)
    for sim_number in range(0, num_of_sims):
        files = glob.glob("{}/b*.dat".format(sim_names[sim_number]))
        for i in range(0, 3):
            print("Running ({} files found in simulation {})".format(np.size(files),sim_names[sim_number]))
            try:
                os.mkdir("Simulation_images/Simulation_{}".format(sim_number + i*num_of_sims))
            except:
                print("Folder exists!")
            for file in files:
                x, y = read_file(file)
                step_number = int(file[file.find("s_")+2:-4])
                np.save("Simulation_images/Simulation_{}/img_{}".format(sim_number + i*num_of_sims,step_number), transform_into_array(x, y, i-1, size=64))
    
if __name__ == "__main__":
    main()
