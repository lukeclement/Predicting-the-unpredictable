import create_network
import loss_functions
from tensorflow.keras import layers, losses, models, optimizers
import glob
import numpy as np
import dat_to_training
import matplotlib.pyplot as plt
import bubble_prediction
import model_analysis

activation = layers.LeakyReLU()
optimizer = optimizers.Adam(learning_rate=0.001, epsilon=0.1)
loss = loss_functions.UBERLOSS
input_frames = 1
model = create_network.create_inception_network(activation, optimizer, loss, input_frames, image_size=64, channels=3,
                                                encode_size=2,
                                                allow_upsampling=True, allow_pooling=True, kernel_size=3,
                                                max_transpose_layers=3,
                                                dropout_rate=0.2, inception=True, simple=False)
print(model.summary())

# training_data = dat_to_training.create_training_data(
#     1, 5, image_size=64,
#     excluded_sims=[12], variants=[0], resolution=0.001, flips_allowed=True, easy_mode=True)
# print(training_data[0])
variants = [0]

model_analysis.cross_check_easy("Alpha", [13, 20])
# resolution = 0.001
# for num in range(15, 16):
#     test_simulation = "Simulation_data_extrapolated/Simulation_{}_{}_{}_{}".format(False, max(variants), resolution,
#                                                                                    num)
#     files = glob.glob("{}/*".format(test_simulation))
#     number_of_files = len(files)
#     print("{}=>{}".format(num, number_of_files))
#     pictures = []
#     for i in range(0, number_of_files):
#         pictures.append(dat_to_training.process_bmp("{}/data_{}.npy".format(test_simulation, i), 64))
#     bubble_prediction.make_gif(pictures, "something_odd")
