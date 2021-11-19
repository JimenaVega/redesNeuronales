import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf 
from   tensorflow import keras, Tensor
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import plot_model
# from tensorflow.keras.utils.vis_utils import model_to_dot
import pydot
from IPython.display import SVG
import time

#-----------------------------------------------------------
# Functions for reading Images 
# returns: list of images, list of corresponding labels 
#-----------------------------------------------------------  


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
#-----------------------------------------------------------
#Import Images
#-----------------------------------------------------------	   
  
trainImages, trainLabels = readTrafficSigns('/home/manu/Descargas/GTSRB/Final_Training/Images')
testImages, test_labels= readTrafficSignsTest('/home/manu/Descargas/GTSRB/Final_Test/Images')
# print(trainImages[0])
# plt.imshow(trainImages[42])
# plt.show()

#-----------------------------------------------------------
# Dataset analysis
#-----------------------------------------------------------

# Count images of each class

class_count = [0]*43

for i in trainLabels:
	class_count[int(i)] += 1

print(class_count)

# Histogram 

n_bins = 43

X = np.arange(n_bins)
plt.figure()
plt.stem(X,class_count)
plt.show()

#-----------------------------------------------------------
# Changes to dataset 
#-----------------------------------------------------------

# Images RGB to GRAY SCALE

trainImages_gray = []
for image in trainImages:
	trainImages_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

testImages_gray = []
for image in testImages:
	testImages_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# Verify GRAY SCALE

# plt.figure()
# plt.imshow(trainImages_gray[0], cmap='gray')
# plt.colorbar()
# plt.grid(False)
# plt.show()

#Resize

trainImages_resize = []

for image in trainImages:
	im = cv2.bitwise_not(image)
	im = cv2.resize(im,(28,28))
	trainImages_resize.append(im)

testImages_resize = []

for image in testImages:
	im = cv2.bitwise_not(image)
	im = cv2.resize(im,(28,28))
	testImages_resize.append(im)
# Verify Resize

# plt.figure(figsize=(10,10))
# for i in range(25):
#	 plt.subplot(5,5,i+1)
#	 plt.xticks([])
#	 plt.yticks([])
#	 plt.grid(False)
#	 plt.imshow(trainImages_resize[i*500], cmap=plt.cm.binary)
#	 plt.xlabel(trainLabels[i*500])
# plt.show()

# Normalize

trainImages_resize = np.array(trainImages_resize)/255.0
testImages_resize  = np.array(testImages_resize)/255.0

# Cast labels of class as int

for i in range(len(trainLabels)):
	trainLabels[i] = int(trainLabels[i])

for i in range(len(test_labels)):
	test_labels[i] = int(test_labels[i])

#-----------------------------------------------------------
# Neural Network 
# - con el agregado de la regulación por Dropout se observó 
# una disminución de la accuracy en el entrenamiento pero  
# un incremento de la misma en la validación
#-----------------------------------------------------------
text= """ Modelos disponibles (ingrese numero):
1 - Modelo con bloque residual y dropout
2 - Modelo de conv 2D basico
3 - Modelo Nico
4 - Modelo con Depthwise 
"""
case=int(input(text))

if case == 1 :
	inputs = keras.layers.Input(shape=(28, 28, 3))
	num_filters = 32

	t = keras.layers.BatchNormalization()(inputs)
	t = keras.layers.Conv2D(kernel_size=3,
				strides=2,
				filters=32,
				padding="same")(t)
	t = relu_bn(t)
	t = residual_block(t, downsample=False, filters=num_filters)
	t = keras.layers.AveragePooling2D(4)(t)
	t = keras.layers.Flatten()(t)
	t = keras.layers.Dropout(0.2)(t)
	t = keras.layers.Dense(50, activation='sigmoid')(t)
	t = keras.layers.Dropout(0.2)(t)
	outputs = keras.layers.Dense(43, activation='softmax')(t)
	model = keras.Model(inputs, outputs)
	epochs = 10

elif case == 2:
	model = keras.Sequential([
		keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
		keras.layers.MaxPooling2D((2, 2)),
		keras.layers.Conv2D(64, (3, 3), activation='relu'),
		keras.layers.MaxPooling2D((2, 2)),
		keras.layers.Conv2D(64, (3, 3), activation='relu'),
		keras.layers.Flatten(),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(50, activation='sigmoid'),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(43, activation='softmax')
	])
	epochs = 7

elif case == 3:

	##################################################################################
	# Lo de Nico
	model = keras.Sequential()

	#1º Capa convolucional de 5x5 para aprender características mas generales de la imagen. Se agrega padding para no modificar
	#el tamaño de la imagen y poder aplicar más capas convolucionales luego.
	model.add(keras.layers.Conv2D(64,(5, 5), activation='relu', padding='same', input_shape=(28,28,3)))
	#model.add(keras.layers.Conv2D(32,(5, 5), activation='relu', padding='same'))
	#1º capa de pooling con una ventana 2x2
	#model.add(keras.layers.MaxPooling2D((2, 2)))

	#2º Capa convolucional de 64 filtros de 3x3 con activacion ReLu
	model.add(keras.layers.Conv2D(128,(3, 3), activation='relu'))
	#model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
	#2º capa de pooling con una ventana 2x2
	#model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D((2, 2)))

	#3º Capa convolucional de 64 filtros de 3x3 con activacion ReLu
	model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
	#3º capa de pooling con una ventana 2x2
	#model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Dropout(rate=0.25))

	#Vemos un resumen de las capas
	#model.summary()

	"""# Añadimos capas densas (clasificador): Una vez que detectamos las principales caracteristicas de cada imagen del set."""

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(256))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(128))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(43, activation='softmax'))

	epochs = 6


elif case == 4:

	##################################################################################
	# Lo de Nico
	model = keras.Sequential()

	#1º Capa convolucional de 5x5 para aprender características mas generales de la imagen. Se agrega padding para no modificar
	#el tamaño de la imagen y poder aplicar más capas convolucionales luego.
	model.add(keras.layers.Conv2D(64,(5, 5), activation='relu', padding='same', input_shape=(28,28,3)))
	#model.add(keras.layers.Conv2D(32,(5, 5), activation='relu', padding='same'))
	#1º capa de pooling con una ventana 2x2
	#model.add(keras.layers.MaxPooling2D((2, 2)))

	#2º Capa convolucional de 64 filtros de 3x3 con activacion ReLu
	# model.add(keras.layers.Conv2D(128,(3, 3), activation='relu'))
	model.add(keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding="valid", strides=(1, 1), depth_multiplier= 10, activation='relu'))
	#model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
	#2º capa de pooling con una ventana 2x2
	#model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D((2, 2)))

	#3º Capa convolucional de 64 filtros de 3x3 con activacion ReLu
	# model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
	model.add(keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding="valid", strides=(1, 1), depth_multiplier= 10, activation='relu'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D((2, 2)))
	# model.add(keras.layers.Conv2D(64,(3, 3), activation='relu'))
	model.add(keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding="valid", strides=(1, 1), depth_multiplier= 10, activation='relu'))
	#3º capa de pooling con una ventana 2x2
	#model.
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.MaxPooling2D((2, 2)))
	model.add(keras.layers.Dropout(rate=0.25))

	#Vemos un resumen de las capas
	#model.summary()

	"""# Añadimos capas densas (clasificador): Una vez que detectamos las principales caracteristicas de cada imagen del set."""

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(256))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(128))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('relu'))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(43, activation='softmax'))

	epochs = 7


elif case == 5:
	model = keras.Sequential([
		keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
		keras.layers.MaxPooling2D((2, 2)),
		keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding="valid", strides=(1, 1), depth_multiplier= 10, activation='relu'),
		keras.layers.MaxPooling2D((2, 2)),
		keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding="valid", strides=(1, 1), depth_multiplier= 10, activation='relu'),
		keras.layers.Flatten(),
		keras.layers.Dropout(0.4),
		keras.layers.Dense(50, activation='sigmoid'),
		keras.layers.Dropout(0.4),
		keras.layers.Dense(43, activation='softmax')
	])
	epochs = 7



model.compile(optimizer='adam',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

model.summary()

plot_model(model, to_file='Net.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

input("Press enter to continue with training")


#-----------------------------------------------------------
# Train Dataset
#-----------------------------------------------------------
inicio = time.time()
history = model.fit(trainImages_resize, np.array(trainLabels), epochs=epochs, validation_data=(testImages_resize, np.array(test_labels)), batch_size=64)
fin = time.time()
print(f"\n Tiempo de entrenamiento: {fin - inicio} segundos")

# Guardar el Modelo
model.save('traffic_signs.h5')


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
# Recrea exactamente el mismo modelo solo desde el archivo


test_loss, test_acc = model.evaluate(testImages_resize,  np.array(test_labels), verbose=1)

