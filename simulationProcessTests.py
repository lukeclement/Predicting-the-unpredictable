import numpy as np
import matplotlib.pyplot as plt
test_file_path = "Example_Simulation/boundaries_828.dat"
test_file = open(test_file_path,"r")

x = []
y = []
interesting_data_hit = False
for line in test_file:
    if "boundary4" in line:
        interesting_data_hit = True
    if interesting_data_hit and not "ZONE" in line:
        data_points = line.strip().split(" ")
        x.append(float(data_points[0]))
        y.append(float(data_points[1]))
h, x_edge, y_edge, image = plt.hist2d(x, y, range=[[-1,1], [-1,1]], bins=(128,128))
print(np.shape(h))
np.save("TestArray",h)
newH = np.load("TestArray.npy")
print(np.shape(newH))
plt.imshow(newH)
plt.imshow(h)
plt.show()
