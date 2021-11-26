import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf 
from   tensorflow import keras, Tensor
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import plot_model
import pydot
from IPython.display import SVG
import time
import os

#mandatory 8 classes
#prohibitory 12
#danger 15
#other 8
#DATASET en:
#/home/paprika/Documents/Redes_Neuronales/datasets/GTSRB_Final_Test_Images-idk/GTSRB/Final_Test/Images

path_img = '/home/paprika/Documents/Redes_Neuronales/datasets/GTSRB_Final_Test_Images-idk/GTSRB/Final_Test/Images'
path_csv = '/home/paprika/Documents/Redes_Neuronales/datasets/GTSRB_Final_Test_Images-idk/GTSRB/Final_Test/Images/GT-final_test.csv'
path_train_img = '/home/paprika/Documents/Redes_Neuronales/datasets/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
width  = 64 
height = 64

def load_images(path_directory, path_csv, classID):

    #classID para cada señal de tránsito
    mandatory_class = np.asarray(['33','34','35','36','37','38','39','40'])
    danger_class = np.asarray(['11','18','19','20','21','22','23','24','25','26','27','28','29','30','31'])
    prohibitory_class = np.asarray(['0','1','2','3','4','5','7','8','9','10','15','16'])
    other_class = np.asarray(['6','12','13','14','17','32','41','42'])
    
    #path_csv: path del archivo .csv que contiene información sobre cada imagen del set (train.csv o test.csv)
    labels = []
    images = []

    data = open(path_csv).read().strip().split("\n")[1:] #leo el archivo .csv y separo las filas en una lista

    for (count, values) in enumerate(data): #count=contador - values=valor de cada elemento de data mientras itera
        
        line_csv = values.strip().split(';') #separo el classID (label) y el path de la imagen
        label = line_csv[-1]
        image_path = line_csv[0] 
        

        if classID=='prohibitory' and label in prohibitory_class:
          image_path = os.path.join(path_directory, image_path) #crea el path completo de la imagen
          image = cv2.imread(image_path) #lee la imagen
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = cv2.resize(image, (64, 64)) #reescala la imagen al tamaño indicado: 64x64 pixeles  
          
          #Actualizo la lista de imagenes y de labels
          images.append(image)
          labels.append(int(label))

        elif classID=='danger' and label in danger_class:
          image_path = os.path.join(path_directory, image_path) #crea el path completo de la imagen
          image = cv2.imread(image_path) #lee la imagen
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = cv2.resize(image, (64, 64)) #reescala la imagen al tamaño indicado: 64x64 pixeles    
          
          #Actualizo la lista de imagenes y de labels
          images.append(image)
          labels.append(int(label))
        
        elif classID=='mandatory' and label in mandatory_class: 
        #   print("soy mandatory")
        #   print('label ={0}| image_path ={1}|'.format(label, image_path))
          image_path = os.path.join(path_directory, image_path) #crea el path completo de la imagen
          image = cv2.imread(image_path) #lee la imagen
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = cv2.resize(image, (64, 64)) #reescala la imagen al tamaño indicado: 64x64 pixeles   

          #Actualizo la lista de imagenes y de labels
          images.append(image)
          labels.append(int(label))

        elif classID=='other' and label in other_class:
          image_path = os.path.join(path_directory, image_path) #crea el path completo de la imagen
          image = cv2.imread(image_path) #lee la imagen
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = cv2.resize(image, (64, 64)) #reescala la imagen al tamaño indicado: 64x64 pixeles   
          
          #Actualizo la lista de imagenes y de labels
          images.append(image)
          labels.append(int(label))
          #labels.append(label)
        

    images = np.asarray(images).astype(float)/255 #convierte las imagenes finales en arreglo numpy para facilitar el manejo posterior
    labels = np.asarray(labels) #convierte la lista de labels en un arreglo numpy
    
    return images, labels



def relu_bn(inputs: Tensor) -> Tensor:
	relu = keras.layers.ReLU()(inputs)
	bn = keras.layers.BatchNormalization()(relu)
	return bn


def residual_block(x: Tensor, downsample: bool, filters: int,										kernel_size: int = 3) -> Tensor:
	y = keras.layers.Conv2D(kernel_size=kernel_size,
			   strides= (1 if not downsample else 2),
			   filters=filters,
			   padding="same")(x)
	y = relu_bn(y)
	y = keras.layers.Conv2D(kernel_size=kernel_size,
			   strides=1,
			   filters=filters,
			   padding="same")(y)

	if downsample:
		x = keras.layers.Conv2D(kernel_size=1,
				   strides=2,
				   filters=filters,
				   padding="same")(x)
	out = keras.layers.Add()([x, y])
	out = relu_bn(out)
	return out

def readTrafficSigns(rootpath):
	images = [] 
	labels = [] 
	# loop over all 43 classes
	for c in range(0,43):
		prefix = rootpath + '/' + format(c, '05d') + '/'			# subdirectory for class
		gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv')	# annotations file
		gtReader = csv.reader(gtFile, delimiter=';')				# csv parser for annotations file
		gtReader.__next__()		# skip header
		for row in gtReader:
			images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
			labels.append(row[7])					   # the 8th column is the label
		gtFile.close()

	return images, labels


def readTrafficSignsTest(rootpath):
	images = [] 
	labels = []

	# loop over all 43 classes
	prefix = rootpath 
	gtFile = open(rootpath+'/GT-final_test.csv') # annotations file
	gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
	gtReader.__next__() # skip header
	
	for row in gtReader:
		images.append(plt.imread(prefix + '/'+row[0])) # the 1th column is the filename
		labels.append(row[7])						  # the 8th column is the label
	gtFile.close()
	return images, labels

def resize_images(images):
	resized_images = []
 
	for image in images:
		im = cv2.bitwise_not(image)
		im = cv2.resize(im,(28,28))
		resized_images.append(im)
  
	return resized_images
#-----------------------------------------------------------
#Import Images
#-----------------------------------------------------------	   
# trainImages, trainLabels = readTrafficSigns(path_train_img)
# testImages, test_labels= readTrafficSignsTest(path_img)


# for i in range(len(trainLabels)):
# 	print(trainLabels[i])
# print(len(trainLabels))
#Separacin de imagenes segun las siguientes 4 clases
# mandatoryImages, mandatoryLabels = load_images(path_img, path_csv, 'mandatory')
# dangerImages, dangerLabels = load_images(path_img, path_csv, 'danger')
# prohibitoryImages, prohibitoryLabels = load_images(path_img, path_csv, 'prohibitory')
# otherImages, otherLabels = load_images(path_img, path_csv, 'other')

# #FALTAN LAS IMAGENES DE TEST
# # testImages, test_labels= readTrafficSignsTest(path_img)
# # trainImages_resize = []

# #resize images to 28,28
# mandatoryImages   = resize_images(mandatoryImages)
# dangerImages 	  = resize_images(dangerImages)
# prohibitoryImages = resize_images(prohibitoryImages)
# otherImages       = resize_images(otherImages)

# #rescale images
# mandatoryImages   = np.array(mandatoryImages)/255.0
# dangerImages 	  = np.array(dangerImages)/255.0
# prohibitoryImages = np.array(prohibitoryImages)/255.0
# otherImages       = np.array(otherImages)/255.0


# testImages_resize = []

# for image in testImages:
# 	im = cv2.bitwise_not(image)
# 	im = cv2.resize(im,(28,28))
# 	testImages_resize.append(im)

# trainImages_resize = np.array(trainImages_resize)/255.0
# testImages_resize  = np.array(testImages_resize)/255.0



#In[1]: Red Neuronal Convolucional Optimidaza: Arquitectura 2


# model = keras.Sequential([
# 	keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
# 	keras.layers.MaxPooling2D((2, 2)),
# 	keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding="valid", strides=(1, 1), depth_multiplier= 10, activation='relu'),
# 	keras.layers.MaxPooling2D((2, 2)),
# 	keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding="valid", strides=(1, 1), depth_multiplier= 10, activation='relu'),
# 	keras.layers.Flatten(),
# 	keras.layers.Dropout(0.4),
# 	keras.layers.Dense(50, activation='sigmoid'),
# 	keras.layers.Dropout(0.4),
# 	keras.layers.Dense(43, activation='softmax')
# ])
# epochs = 7
# model.summary()

# model.compile(optimizer='adam',
# 			  loss='sparse_categorical_crossentropy',
# 			  metrics=['accuracy'])

# plot_model(model, to_file='CNN.png')
# #-----------------------------------------------------------
# # Train Dataset
# #-----------------------------------------------------------
# inicio = time.time()
# history = model.fit(mandatoryImages, np.array(mandatoryLabels), epochs=epochs, validation_data=(testImages_resize, np.array(test_labels)), batch_size=64)
# fin = time.time()
# print(f"\n Tiempo de entrenamiento: {fin - inicio} segundos")

# # Guardar el Modelo
# model.save('mandatory_signs.h5')

# test_loss, test_acc = model.evaluate(testImages_resize,  np.array(test_labels), verbose=1)
# %%
