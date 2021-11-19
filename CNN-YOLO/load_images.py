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
        (label, image_path) = values.strip().split(',')[-2:] #separo el classID (label) y el path de la imagen

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
          #labels.append(int(label))
          labels.append(label)
        

    images = np.asarray(images).astype(float)/255 #convierte las imagenes finales en arreglo numpy para facilitar el manejo posterior
    labels = np.asarray(labels) #convierte la lista de labels en un arreglo numpy
    
    return images, labels