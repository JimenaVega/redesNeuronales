

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import json

# In[0]: Load the model from json and hs5

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# In[1]: Evaluar el modelo

test_loss, test_acc = loaded_model.evaluate(test_images,  test_labels, verbose=2)
print("accuracy: ", test_acc)

#Predicciones sobre las imagenes de testeo
predictions = loaded_model.predict(test_images)

print("Predicciones: \n", predictions[1:10])

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,3,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,3,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()