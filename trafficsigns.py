
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from os import read
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import os


dataSetSIZE = 43
# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
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

def getTestImages():

    test_images = []
    path = os.walk("./test")
    imagesInD = []

    i = 0
    for root, directories, files in path:
        imagesInD = files
        for file in files:
        
            path = "./test/{0}".format(file)
            print(path)
            the_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            the_img = cv2.bitwise_not(the_img)
            test_images.append(cv2.resize(the_img,(28,28)))
            #Para mostrar las imagenes de prueba descomentar:
            # plt.figure(i)
            # plt.imshow(test_images[i], cmap='gray')
            i += 1

#   plt.show()


    return test_images, i, imagesInD

class_names = ['20 sign', '30 sign','50 sign', '60 sign', '70 sign', '80 sign','80 sign', '100 sign', '120 sign', 'two cars']

for i in range(48-len(class_names)):
    class_names.append('other')

train_images, train_labels = readTrafficSigns('GTSRB/Training')

# In[0]: Tratamiento de las iamgenes
for i in range(len(train_images)):

	train_images[i] = cv2.cvtColor(train_images[i], cv2.COLOR_BGR2GRAY)
	train_images[i] = cv2.bitwise_not(train_images[i])
	train_images[i] = cv2.resize(train_images[i],(28,28))


# In[1]: Se analizará cómo son las imágenes y cuántas imágenes tiene el dataset en total y en cada clase

train_images = np.array(train_images)
train_labels = np.array(train_labels).astype(np.uint8)
train_images = train_images / 255.0

class_count = [0]*dataSetSIZE
for i in train_labels:
  class_count[int(i)] += 1

print("----------------------------------------")
print("DATASET INFO:")
print("train images shape: ", train_images.shape)
print("train label: ", type(train_labels[0]))
print("amount of classes: ", len(class_count))
print("----------------------------------------")


# n_bins = dataSetSIZE
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# X = np.arange(n_bins)
# ax.bar(X,class_count,width = 0.9, label='Clases')
# labels = [item.get_text() for item in ax.get_xticklabels()]
# plt.xticks(X)
# plt.show()
#test_images = test_images / 255.0

# #print(train_images[i].shape)
# plt.figure(1)
# plt.title("Image #1000")
# plt.imshow(train_images[1000],  cmap='gray')
# plt.show()


# plt.figure(2, figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
# # plt.xlabel(class_names[train_labels[i]])
# plt.show()

#In[2]: Se entrenará un nuevo modelo con estos datos.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(dataSetSIZE, activation='softmax')
])


model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_images, train_labels, epochs=15)

# In[3]:
# Cambiar la arquitectura de la red y evaluar sus resultados (1 arquitectura mejor y 1 peor).
# Se descargarán 4 imágenes de Google que tengan señales de tránsito, se adaptará la imagen 
# para que sea de 28x28 y escala de gris y se usará el modelo para predecir su clase.


test_images, numOftestIm, imagesInD = getTestImages()
test_images = np.array(test_images)

test_labels = []
for i in range(numOftestIm):
    test_labels.append(i)
test_labels = np.array(test_labels)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
#print('\nAccuracy: {0} Loss; {1}'.format(test_acc, test_loss))
predictions = model.predict(test_images)

# In[] Predicciones
print(predictions.shape)
print("Original     |    Prediction")
for i in range(numOftestIm):
    
    pred_label = np.argmax(predictions[i])
    print(imagesInD[i],  class_names[pred_label], sep="             ")
