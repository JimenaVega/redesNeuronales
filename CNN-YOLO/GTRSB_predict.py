import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf 
from   tensorflow import keras
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras import metrics
from sklearn.metrics import confusion_matrix
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
from skimage import transform
from skimage import exposure
from skimage import io
from skimage.color import rgb2gray
from random import randint
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import seaborn as sns
import pandas as pd



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
# Plot Functions
#-----------------------------------------------------------

def plot_image(i, predictions_array, true_label, img, cat):
	predictions_array, true_label, img = predictions_array, true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	plt.imshow(img, cmap=plt.cm.binary)

	predicted_label = np.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(cat[predicted_label],
								100*np.max(predictions_array),
								cat[true_label]),
								color=color)

def plot_value_array(i, predictions_array, true_label):
	predictions_array, true_label = predictions_array, true_label[i]
	plt.grid(False)
	plt.xticks(range(43))
	plt.yticks([])
	thisplot = plt.bar(range(43), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_array)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')



categories={0:'Max 20',
			1:'Max 30',
			2:'Max 50',
			3:'Max 60',
			4:'Max 70',
			5:'Max 80',
			6:'Fmax 80',
			7:'Max 100',
			8:'Max 120',
			9:'No Adto',
			10:'No Adto Cmns',
			11:'Itscn Ptra',
			12:'Czda Ptra',
			13:'Ced Paso',
			14:'STOP',
			15:'Circ Prohi',
			16:'Prohi trpt',
			17:'Ctrmn',
			18:'Pgro',
			19:'Crva Pgrs Izq',
			20:'Crva Pgrs Der',
			21:'Crvas Pgrs Izq',
			22:'Crvas Pgrs Der',
			23:'Pvto Dzte',
			24:'Etcht Der',
			25:'Obras',
			26:'Sfro',
			27:'Crce Ptnl',
			28:'Ninios',
			29:'Cclts',
			30:'Hielo',
			31:'Amls Lbrs',
			32:'Fin Prohi',
			33:'Stdo Obgto Der',
			34:'Stdo Obgto Izq',
			35:'Stdo Obgto Adte',
			36:'Unicas Drcns Der',
			37:'Unicas Drcns Izq',
			38:'Pso Obgto Der',
			39:'Pso Obgto Izq',
			40:'Itscn stdo Obgto',
			41:'Fin Adto',
			42:'Fin Adto Cmns'}

print(categories.values())


testImages, test_labels= readTrafficSignsTest('/home/manu/Descargas/GTSRB/Final_Test/Images')

class_count = [0]*43

for i in test_labels:
	class_count[int(i)] += 1

print(class_count)

# Histogram 

n_bins = 43

X = np.arange(n_bins)
plt.figure()
plt.stem(X,class_count)
plt.show()


testImages_gray = []
for image in testImages:
	testImages_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


testImages_resize = []

for image in testImages:
	im = cv2.bitwise_not(image)
	im = cv2.resize(im,(28,28))
	testImages_resize.append(im)

testImages_resize  = np.array(testImages_resize)/255.0


for i in range(len(test_labels)):
	test_labels[i] = int(test_labels[i])


text= """ Modelos disponibles (ingrese numero):
1 - Comun
2 - Modelo con bloque residual 
3 - Modelo Nico
"""


case=int(input(text))

if case == 1:
	model = keras.models.load_model('traffic_signs.h5')
elif case == 2:
	model = keras.models.load_model('traffic_signs_residual.h5')
elif case == 3:
	model = keras.models.load_model('traffic_signs_nico.h5')
else:
	print("Entrada incorrecta")
	exit()

#-----------------------------------------------------------
# Test Network
#-----------------------------------------------------------

predictions = model.predict(testImages_resize)
test_loss, test_acc = model.evaluate(testImages_resize,  np.array(test_labels), verbose=2)

print('\nTest accuracy:', test_acc)

failed = []
for i in range(len(predictions)):
	if np.argmax(predictions[i]) != test_labels[i] :
		failed.append(i)

print("Failed Images: ", 100*len(failed)/len(predictions),'%')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for k in range(num_images):
	i = failed[k]
	plt.subplot(num_rows, 2*num_cols, 2*k+1)
	plot_image(i, predictions[i], test_labels, testImages, categories)
	plt.subplot(num_rows, 2*num_cols, 2*k+2)
	plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()

num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    i*10
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, testImages, categories)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()



cmat = metrics.confusion_matrix(test_labels, np.argmax(predictions, axis=1))
plt.figure(figsize=(15,15))
sns.heatmap(cmat, annot = True, cbar = False, cmap='Paired', fmt="d", xticklabels=categories, yticklabels=categories);
plt.show()

print('\n'*3)
text=metrics.classification_report(test_labels, np.argmax(predictions, axis=1), target_names = categories.values(),output_dict=True)

print(text[test_labels[0]]['precision'])

"""# Precisión por clase"""

# classwise_acc = cmat.diagonal()/cmat.sum(axis=1) * 100 
# cls_acc = pd.DataFrame({'Class_Label':[class_names[i] for i in range(43)], 'Accuracy': classwise_acc.tolist()}, columns = ['Class_Label', 'Accuracy'])
# cls_acc.style.format({"Accuracy": "{:,.3f}",}).hide_index().bar(subset=["Accuracy"], color='royalblue')
