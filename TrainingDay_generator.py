#Clone Behavior training network

import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn

def preprocess(samples, minThresh = 0.001, correctionFactor = 0.2):
    '''
    make list cleaner, remove all points that have an angle of 0
    Creates two lists, one of the images, and one for the angles
    '''
    filenames = []
    angles = []
    for sample in samples:
        if abs(float(sample[3])) < minThresh:
            #if the angle is below the minThresh, don't add this file to the dataset
            continue
        for i in range(3):
            #center
            correction = 0
            if i == 1:
                #left
                correction = correctionFactor
            elif i == 2:
                #right
                correction = -correctionFactor
                
            source_path = sample[i]
            filenames.append('../Resources/IMG/' + source_path.split('/')[-1])
            angles.append(float(sample[3]) + correction)

    filenames = np.array(filenames)
    angles = np.array(angles)
    return (filenames, angles)
            
        
def visualizeDistribution(angles):
    '''
    create a histogram to see the distribution of the dataset after selection
    returns bins and average
    '''
    num_bins = 21
    average = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)
    width = 0.5 *(bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) /2
    plt.bar(center, hist, align = 'center', width = width)
    #show average line
    plt.plot((np.min(angles), np.max(angles)), (average, average), 'k-')
    plt.savefig('dataDistribution.png')
    return (bins, hist, average)
    
def equalize(dataSet, distribution):
    '''
    takes in a dataset and randomly removes values if there's too much of a certan distribution
    '''
    
    filename = dataSet[0]
    angles = dataSet[1]
    bins = distribution[0]
    hist = distribution[1]
    average = distribution[2]


    newX = []
    newY = []
    for i in range(len(angles)):
        #iterate through all values
        
        for j in range(len(bins)-1):
            if angles[i] > bins[j] and angles[i] <= bins[j+1]:
                if np.random.rand() < average/hist[j]:
                    newX.append(filename[i])
                    newY.append(angles[i])
    newX = np.array(newX)
    newY = np.array(newY)
    dataSet =(newX, newY)
    return dataSet
            
    
def generator(dataSet, batch_size = 32, threshold = 0.3):
    #print(dataSet)
    filenames = dataSet[0]
    angles = dataSet[1]
    num_samples = len(angles)


    ##########################
    #initialize temp arrays
    imageOutput = []
    angleOutput = []
    while True:
        #generators loop forever
        filenames, angles = sklearn.utils.shuffle(filenames, angles)
        for i in range(num_samples):
            #uses cv2.imread to read image, convert to RGB so it works on video.py
            image = cv2.imread(filenames[i])
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            #opportunity for more preprocessing here
            imageOutput.append(image)
            angleOutput.append(angles[i])

            if len(imageOutput) == batch_size:
                #When we fill the batch, sent it out
                X_train = np.array(imageOutput)
                y_train = np.array(angleOutput)
                #empty temp arrays
                imageOutput = []
                angleOutput = []
                yield sklearn.utils.shuffle(X_train, y_train)

            if abs(angles[i]) > threshold:
                #create a flipped version if the angle is above a certain threshold
                imageOutput.append(cv2.flip(image,1))
                angleOutput.append(angles[i]*-1.0)

                if len(imageOutput) == batch_size:
                    #When we fill the batch, sent it out
                    X_train = np.array(imageOutput)
                    y_train = np.array(angleOutput)
                    #empty temp arrays
                    imageOutput = []
                    angleOutput = []
                    yield sklearn.utils.shuffle(X_train, y_train)



########Code starts here #########################
samples = []
with open('../Resources/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#delete's zero angle and creates dataset, a tuple that contains the filenames and the angles in two lists
dataSet = preprocess(samples)
distribution = visualizeDistribution(dataSet[1])
dataSet = equalize(dataSet, distribution)
distribution = visualizeDistribution(dataSet[1])

train_file_path, validation_file_path, train_angles, validation_angles = train_test_split(dataSet[0], dataSet[1], test_size = 0.2)

train_samples = (train_file_path, train_angles)
validation_samples = (validation_file_path, validation_angles)

train_generator = generator(train_samples, batch_size = 256, threshold = 0)
validation_generator = generator(validation_samples, batch_size = 256, threshold = 0)
                


#Create model using NVIDIA architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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

#Model Fit with Generator
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_file_path),
                                     validation_data = validation_generator, nb_val_samples = len(validation_file_path),nb_epoch=5, verbose = 1)
model.save('model.h5')
print('Model Saved')
#print(history_object.history.keys())

plt.gcf().clear()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.savefig('training_curve.png')
plt.ion()
plt.show()

