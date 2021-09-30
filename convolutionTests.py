import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import glob
import matplotlib.pyplot as plt
files = glob.glob("Simulation_images/*")
num_of_data = np.size(files)
training_limit = num_of_data*0.9
train_images_source = files[:int(training_limit)]
training_solutions = []
test_images_source = files[int(training_limit):]
testing_solutions = []
for training_image in train_images_source:
    number = int(training_image[training_image.find("/")+1:-4])
    training_solutions.append("Simulation_images/{}.npy".format(number+1))
for testing_image in test_images_source:
    number = int(testing_image[testing_image.find("/")+1:-4])
    testing_solutions.append("Simulation_images/{}.npy".format(number+1))

test_set = []
test_solutions = []
train_set = []
train_solutions = []

for i in range(0,np.size(train_images_source)):
    if not str(num_of_data-1) in train_images_source[i]:
        loading_test_file = np.load(train_images_source[i])
        loading_sol_file = np.load(training_solutions[i])
        train_set.append(loading_test_file/255.0)
        train_solutions.append(loading_sol_file/255.0)
        #train_set.append(tf.cast(loading_test_file, tf.float32))
        #train_solutions.append(tf.cast(loading_sol_file, tf.float32))

for i in range(0,np.size(test_images_source)):
    if not str(num_of_data-1) in test_images_source[i]:
        loading_test_file = np.load(test_images_source[i])
        loading_sol_file = np.load(testing_solutions[i])
        test_set.append(loading_test_file/255.0)
        test_solutions.append(loading_sol_file/255.0)
        #test_set.append(tf.cast(loading_test_file, tf.float32))
        #test_solutions.append(tf.cast(loading_sol_file, tf.float32))
print(len(test_set))
print(len(test_solutions))
print(len(train_set))
print(len(train_solutions))



active='tanh'
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=active))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation=active))
model.add(layers.Conv2DTranspose(64, (3, 3), activation=active))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2DTranspose(64, (3, 3), activation=active))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2DTranspose(32, (3, 3), activation=active))
model.add(layers.Conv2DTranspose(1, (3, 3), activation='sigmoid'))

print(model.summary())
loss_fn = losses.BinaryCrossentropy()
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
test_set = np.stack([i.tolist() for i in test_set])
test_solutions = np.stack([i.tolist() for i in test_solutions])
train_set = np.stack([i.tolist() for i in train_set])
train_solutions = np.stack([i.tolist() for i in train_solutions])


history = model.fit(train_set, train_solutions, epochs=20, validation_data=(test_set, test_solutions))

out = model(test_set[0:1])
plt.imshow(out[0]*255.0)
plt.savefig("Machine.png")
plt.show()
plt.imshow(test_set[0])
plt.savefig("First.png")
plt.imshow(test_solutions[0])
plt.savefig("Second.png")
plt.clf()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig("Loss.png")
#test_loss, test_acc = model.evaluate(test_set, test_solutions, verbose=2)
#plt.show()
