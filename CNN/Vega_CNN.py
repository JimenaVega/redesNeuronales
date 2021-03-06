##
# CNN LeNet
# 0. Cargado de dataset e imagenes de testeo
# 1. Convolución
# 2. Función de activación / No linealidad (ReLU)
# 3. Pooling o Submuestreo
# 4. Clasificador (capas completamente conectada)
# #

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

datasize = 43

def readTrafficSigns(rootpath):

    images = [] # images
    labels = [] # corresponding labels

    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header

        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()

    return images, labels

def getTestImages(path):

    test_images = []
    test_labels = []
    

    i = 0
    #path = './GTSRB_Final_Test_GT/GT-final_test.csv'
    csvFile = open(path + '/GT-final_test.csv')
    csvReader = csv.reader(csvFile, delimiter=';')
    csvReader.__next__()

    for row in csvReader:
        test_images.append(plt.imread(path + '/' + row[0]))
        test_labels.append(row[7])
        i += 1
        
        #if(i > 12000 and i < 1210):print("row[7]: {0} type. {1}".format(row[7], type(row[7])))
    print("i = ", i)
    print("sizeof test_labels ", len(test_labels))
    csvFile.close()
#print("INSIDE test_labels.shape = ", test_labels.shape)
 
    return test_images, test_labels


# In[0]: cargado de imagenes

train_images, train_labels = readTrafficSigns('GTSRB/Training')
test_images, test_labels = getTestImages('./GTSRB_Final_Test_Images/GTSRB/Final_Test/Images')
# Acondicionamiento de images de entrenamiento
for i in range(len(train_images)):
	train_images[i] = cv2.bitwise_not(train_images[i])
	train_images[i] = cv2.resize(train_images[i],(32, 32))


train_images = np.array(train_images)
#train_labels = np.array(train_labels).astype(np.uint8)
# #train_labels = map(int, train_labels)
train_labels = list(map(int, train_labels))
train_labels = np.array(train_labels)
#train_labels = train_labels.astype('int')
# for i in range(len(train_labels)):
#     train_labels[i] = int(train_labels[i])

train_images = train_images / 255.0


#Acondicionamiento de imagenes de test
for i in range(len(test_images)):
    test_images[i] = cv2.bitwise_not(test_images[i])
    test_images[i] = cv2.resize(test_images[i], (32, 32))

test_images = np.array(test_images) / 255.0
#test_labels = map(int, test_labels)

test_labels = list(map(int, test_labels))
test_labels = np.asarray(test_labels)
#test_labels = train_labels.astype('int')
#test_labels = np.array(test_labels).astype(np.uint8)
#map(int, test_labels)

print("----------------------------------------")
print("DATASET INFO:")
print("train_images shape: ", train_images.shape)
print("train_images type: ", type(train_images))
print("train_labels shape: ", train_labels.shape)
print("train_label[0]: ", type(train_labels[0]))
print("train_label type: ", type(train_labels))
print("test_images shape: ", test_images.shape)
print("test_images type: ", type(test_images))
print("test_labels shape: ", test_labels.shape)
print("----------------------------------------")
# # In[1]:

model = models.Sequential()
# 1st convolution layer: 32 filters of 3x3, ReLU activation function
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 1st max pooling layer: 2x2
model.add(layers.MaxPooling2D((2, 2)))
# 2nd convolution layer: 64 filters of 3x3, ReLU activation function
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 2nd max pooling layer: 2x2
model.add(layers.MaxPooling2D((2, 2)))
# 3st convolution layer: 64 filters of 3x3, ReLU activation function
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#In[2]:Añadir capas densas (clasificador)

# Flatten layer
model.add(layers.Flatten())
# Dense layer
model.add(layers.Dense(64, activation='relu'))
# Output layer
model.add(layers.Dense(datasize, activation='softmax'))


#In[3]: Compilar y entrenar el modelo

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, 
                    validation_data=(test_images, test_labels))

# In[4]: Evaluar el modelo

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("accuracy: ", test_acc)