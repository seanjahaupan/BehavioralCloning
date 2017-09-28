#Clone Behavior training network

import csv
import cv2
import numpy as np

lines = []
#edit this line
with open('../Resources/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

#For every line, add to the training and output lists
for line in lines:
    #cycle through center, left, and right images
    for i in range(3):
        #correction factors are applied to left and right images
        correction = 0
        if i == 1:
            correction = 0.1
        elif i == 2:
            correction = -0.1

        
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '../Resources/IMG/' + filename
        #Uses cv2.imread to read the image
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Resize opportunity
        #######################

        ######################
        images.append(image)
        #add output to measurement list with correction factor
        measurement = float(line[3]) + correction
        measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    #flip the image to get another data point
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as pyplot

#Create model using NVIDIA architecture
model = Sequential()
#Load previous model as a starting point
#model.load_weights("model.h5")
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
'''
#relu activation
model.add(Convolution2D(24,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
'''

#elu activation
model.add(Convolution2D(24,5,5, subsample=(2,2), activation = 'elu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'elu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = 'elu'))
model.add(Convolution2D(64,3,3, activation = 'elu'))
model.add(Convolution2D(64,3,3, activation = 'elu'))
model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Uses Adam optimizer
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, verbose = 1)
'''
history_object = model.fit_generator(train_generator, samples_per_epoch =
                                     len(train_samples), validation_data =
                                     validation_generator,
                                     nv_val_samples = len(validation_samples),
                                     nb_epoch-5, verbose = 1)
print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlavel('epoch')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.show()
'''
model.save('model.h5')
print('Model Saved')
