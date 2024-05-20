
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# !apt-get update && apt-get install -y git

# !pip install gitpython

# from google.colab import files
import zipfile
import io

# # Selecciona el archivo comprimido para subir
# # uploaded = files.upload()

# # Crear la carpeta 'dataset' si no existe
# # La carpeta dataset la crea en content, el dataset debe traer las imagenes
# import os
# dataset_dir = '/content/dataset'
# if not os.path.exists(dataset_dir):
#     os.makedirs(dataset_dir)

# # Descomprimir el archivo en la carpeta 'dataset'
# for fn in uploaded.keys():
#     print('Se ha subido el archivo "{name}" con una longitud de {length} bytes'.format(
#         name=fn, length=len(uploaded[fn])))

#     # Descomprimir el archivo
#     with zipfile.ZipFile(io.BytesIO(uploaded[fn]), 'r') as zip_ref:
#         zip_ref.extractall(dataset_dir)

# Downloading  the github repository containing the landmark detection model
import git
import urllib.request

shape_predictor_url = "https://github.com/codeniko/shape_predictor_81_face_landmarks"
shape_predictor_path = "/content/shape_predictor_81_face_landmarks"


#Descarga el contenido que se encuentra en shape_predictor_url y lo guarda en la ubicación especificada en shape_predictor_path
#urllib.request.urlretrieve(shape_predictor_url, shape_predictor_path)

# git.Repo.clone_from(shape_predictor_url, shape_predictor_path)

# print("Descarga completada:", shape_predictor_path)

# !apt-get update
# !apt-get install -y cmake
# !pip install numpy

# !pip install dlib

import dlib
import os

shape_predictor_path = "/content/shape_predictor_81_face_landmarks"

#model_path = "/content/shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat"
model_path = os.path.join(shape_predictor_path, "shape_predictor_81_face_landmarks.dat")

#Se carga el archivo y se crea un objeto de modelo, el cual se utilizará para predecir los puntos de referencia faciales
# Se carga un modelo pre-entrenado el cual ayuda a ubicar los puntos claves sobre el rostro
model = dlib.shape_predictor(model_path)

#Se crea un objeto detector para detectar rostros frontales en una imagen
# es capaz de identificar rostros humanos en imágenes y proporcionar las coordenadas de las regiones faciales detectadas.
detector = dlib.get_frontal_face_detector()


#Juntos estos dos se usan para detectar rostros y luego predecir los puntos de referencia en esos rostros

import math

def distance(point1, point2):
    # Unpack the coordinates from the input arrays
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the distance
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance

def angle(p1, p2, p3):
    # Unpack the coordinates from the input arrays
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculate the vectors of the two lines
    vector1 = (x1 - x2, y1 - y2)
    vector2 = (x3 - x2, y3 - y2)

    # Calculate the dot product and magnitudes of the vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate the cosine of the angle between the lines
    cosine = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians and convert to degrees
    angle_rad = math.acos(cosine)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def midpoint(point1, point2):
    # Unpack the coordinates from the input arrays
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the midpoint coordinates
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2

    # Create an array to store the midpoint coordinates
    midpoint = [midpoint_x, midpoint_y]

    return midpoint

import cv2
import numpy as np
from imutils import face_utils
import os

data_pth = "content/dataset"

faces = []
labels = []

for shape in os.listdir(data_pth):
    print(shape)
    k = shape
    shape_pth = os.path.join(data_pth, shape)
    print(shape_pth)
    for img in os.listdir(shape_pth):

        img_pth = os.path.join(shape_pth, img)
        # Convierte una imagen en un objeto de imagen que se puede manipular y procesar en python
        pic = cv2.imread(img_pth)

        if pic is None:
            print("Error: Failed to load image or image file does not exist.")
            continue
        else:
            # Convert the image to grayscale or perform other operations
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

        #Pasamos la imagen a través del detector de rostros para detectar rostros en la imagen.
        #El detector de rostros analiza la imagen y devuelve las regiones faciales detectadas en forma de rectángulos delimitadores que rodean las caras encontradas.
        dets = detector(pic)

        #Recorremos las regiones faciales detectadas para obtener los puntos de referencia faciales
        for face in dets:
            #Le pasamos la imagen y la región facial detectada al modelo predictor de forma facial.
            #El modelo utiliza esta información para predecir las coordenadas de los puntos de referencia faciales dentro de la región facial detectada
            shape = model(pic, face)
            #Se obtiene los puntos ubicados en el rostro
            coords = face_utils.shape_to_np(shape)   # our coordinates of face landmarks are stored here

            # getting some facial features from the coordinates of the landmarks

            l1 = coords[0]
            l2 = coords[1]
            l3 = coords[2]          #pomulo izquierda
            l4 = coords[3]          #pomulo izquierda
            l5 = coords[4]          #pomulo izquierda
            l6 = coords[5]          #pomulo izquierda
            l7 = coords[6]          #pomulo izquierda
            l8 = coords[7]          #pomulo izquierda
            l9 = coords[8]          #parte mas baja del rostro, menton o barbilla
            l10 = coords[9]        #parte media del menton o barbilla a la altura de donde acaba los labios, del lado derecho
            l11 = coords[10]        #parte alta del menton o barbilla a la altura de donde acaba los labios, del lado derecho
            l12 = coords[11]        #parte alta del menton o barbilla a la altura de donde acaba los labios, del lado derecho
            l13 = coords[12]        #mejilla a la altura del labio inferior, del lado derecho
            l14 = coords[13]        #mejilla a la altura del labio inferior, del lado derecho
            l15 = coords[14]        #pomulo derecha
            l16 = coords[15]        #pomulo derecha
            l17 = coords[16]        #pomulo derecha
            l69 = coords[68]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
            l70 = coords[69]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
            l71 = coords[70]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
            l72 = coords[71]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
            l73 = coords[72]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
            l74 = coords[73]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
            l75 = coords[74]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
            l76 = coords[75]        #parte media entre el fin de la ceja y la frente, del lado derecho
            l77 = coords[76]        #parte media entre el fin de la ceja y la frente, del lado derecho
            l78 = coords[77]        #parte media entre el fin de la ceja y la frente, del lado derecho
            l79 = coords[78]        #parte media entre el fin de la ceja y la frente, del lado derecho
            l80 = coords[79]        #parte media entre el fin de la ceja y la frente, del lado derecho
            l81 = coords[80]
#------------------------------------------------------
            d1 = distance(l3, l15)                    #distancia entre los pomulos
            d2 = distance(l76, l75)
            d3 = distance(midpoint(l70, l73), l9)     #distacia de lado a lado de la frente
            d4 = distance(l9, l13)                    #distancia entre la barbilla y la mejilla
            d5 = distance(l5, l13)                    #distancia entre mejillas
            d6 = distance(l7, l11)                    #distancia entre parte alta del menton
            d7 = distance(l8, l10)                    #distancia entre parte media del menton
            d8 = distance(l6, l12)
            d9 = distance(l4, l14)
            d10 = distance(l1, l17)
            d11 = distance(l78, l79)
            d12 = distance(l77,l80)
            d13 = distance(l69,l6)
            d14 = distance(l70,l7)
            d15 = distance(l71,l6)

#------------------------------------------------------

            DD = d1 + d2 + d3 + d4 + d5 + d6 + d7  + d8 + d9 + d10 + d11 + d12 + d13 + d14 + d15  #suma de las distancias
#------------------------------------------------------
            D1 = d1/DD
            D2 = d2/DD
            D3 = d3/DD
            D4 = d4/DD
            D5 = d5/DD
            D6 = d6/DD
            D7 = d7/DD
            D8 = d8/DD
            D9 = d9/DD
            D10 = d10/DD
            D11 = d11/DD
            D12 = d12/DD
            D13 = d13/DD
            D14 = d14/DD
            D15 = d15/DD
#------------------------------------------------------
            R1 = D2/D1
            R2 = D1/D3
            R3 = D2/D3
            R4 = D1/D5
            R5 = D6/D5
            R6 = D4/D6
            R7 = D6/D1
            R8 = D5/D2
            R9 = D4/D5
            R10 = D7/D6
#------------------------------------------------------
            A1 = angle(midpoint(l70, l73), l9, l11) / 180
            A2 = angle(midpoint(l70, l73), l9, l13) / 180
            A3 = angle(l3, l15, l13) / 180
            A4 = angle(l8, l9, l10) / 180
            A5 = angle(l3, l4, l7) / 180
            A6 = angle(l7, l11, l13) / 180

#------------------------------------------------------
            features = []
            features.append(R1)
            features.append(R2)
            features.append(R3)
            features.append(R4) #
            features.append(R5)
            features.append(R6)
            features.append(R7)
            features.append(R8) #
            features.append(R9)
            features.append(R10)
            features.append(D1)
            features.append(D2)
            features.append(D3)
            features.append(D5)
            features.append(D6)
            features.append(D7)
            features.append(D8)
            features.append(D9)
            features.append(D10)
            features.append(D11)
            features.append(D12)
            features.append(D13)
            features.append(D14)
            features.append(D15)
            features.append(A1) #
            features.append(A2) #
            features.append(A3) #
            features.append(A4) #
            features.append(A5) #
            features.append(A6)#

            #features = np.array(features)

            if k == "Heart":
                labels.append(0)
            elif k == "Oblong":
                labels.append(1)
            elif k == "Oval":
                labels.append(2)
            elif k == "Round":
                labels.append(3)
            elif k == "Square":
                labels.append(4)
            else:
                print('ERROR')
                break

            faces.append(features)

# saving the dataset

# import pickle

# # Abrir el archivo en modo de escritura binaria ('wb')
# with open('/content/df.pickle', 'wb') as f:
#   # Guardar los datos en el archivo
#     pickle.dump({'data': faces, 'labels': labels}, f)

# # loading the saved features

# #import pickle

# # Cargar los datos desde el archivo
# with open('/content/df.pickle', 'rb') as f:
#     df = pickle.load(f)

# Extraer datos y etiquetas del diccionario cargado
# faces = df['data']
# labels = df['labels']

print("Número de muestras en faces:", len(faces))
print("Número de muestras en labels:", len(labels))

n_datos = len(faces)

# faces

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

# Convertir listas en arreglos numpy
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# saving the dataset test

# Abrir el archivo en modo de escritura binaria ('wb')
# with open('/content/test.pickle', 'wb') as f:
#   # Guardar los datos en el archivo
#     pickle.dump({'data': y_train, 'labels': y_test}, f)

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Flatten

# Definir el modelo de la red neuronal
modelo = Sequential([
    Dense(100,activation='relu',input_shape=(30,)),
    #Dense(50, activation='relu', input_shape=(15,1)),
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(5, activation='softmax')
])

# Compilar el modelo
modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Tamaño del lote
Tamano_lote = 32
# Entrenar el modelo
history = modelo.fit(x_train, y_train, epochs=2000,steps_per_epoch = math.ceil(n_datos / Tamano_lote))#), validation_data=(x_test, y_test))

# Evaluar el modelo en los datos de prueba
test_loss, test_accuracy = modelo.evaluate(x_test, y_test)

print("Precisión en datos de prueba:", test_accuracy)

import matplotlib.pyplot as plt

plt.xlabel('epocas')
plt.ylabel('loss')
plt.plot(history.history['loss'])

# Evaluar el modelo en los datos de prueba
test_loss, test_accuracy = modelo.evaluate(x_test, y_test)

print("Precisión del modelo:", test_accuracy)
modelo.save('v3.h5')
# modelo.export('modelo/modelo1')



# import cv2
# # from google.colab.patches import cv2_imshow  # Para mostrar imágenes en Google Colab
# # Ruta de la imagen
# image_path = "1.jpg"

# # Leer la imagen
# image = cv2.imread(image_path)

# # Convertir a escala de grises para detección de rostros
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Detectar rostros
# faces_prueba = detector(gray)

# model = dlib.shape_predictor(model_path)
# features = []
# # Dibujar los puntos de referencia para cada rostro detectado
# for face in faces_prueba:
#     # Obtener los puntos de referencia
#     landmarks = model(gray, face)
#     coords = face_utils.shape_to_np(landmarks)  # Convertir a coordenadas NumPy

#     l3 = coords[3]          #pomulo izquierda
#     l15 = coords[15]        #pomulo derecha
#     l70 = coords[70]        #parte alta de la frente, altura de la mitad de la ceja, del lado izquierdo
#     l76 = coords[76]        #parte media entre el fin de la ceja y la frente, del lado izquierdo
#     l80 = coords[80]        #parte media entre el fin de la ceja y la frente, del lado derecho
#     l73 = coords[73]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
#     l9 = coords[9]          #parte mas baja del rostro, menton o barbilla
#     l13 = coords[13]        #mejilla a la altura del labio inferior, del lado derecho
#     l5 = coords[5]          #mejilla a la altura del labio inferior, del lado izquierdo
#     l7 = coords[7]          #parte alta del menton o barbilla a la altura de donde acaba los labios, del lado izquierdo
#     l11 = coords[11]        #parte alta del menton o barbilla a la altura de donde acaba los labios, del lado derecho
#     l8 = coords[8]          #parte media del menton o barbilla a la altura de donde acaba los labios, del lado izquierdo
#     l10 = coords[10]        #parte media del menton o barbilla a la altura de donde acaba los labios, del lado derecho
# #------------------------------------------------------
#     d1 = distance(l3, l15)                    #distancia entre los pomulos
#     d2 = distance(l76, l80)
#     d3 = distance(midpoint(l70, l73), l9)     #distacia de lado a lado de la frente
#     d4 = distance(l9, l13)                    #distancia entre la barbilla y la mejilla
#     d5 = distance(l5, l13)                    #distancia entre mejillas
#     d6 = distance(l7, l11)                    #distancia entre parte alta del menton
#     d7 = distance(l8, l10)                    #distancia entre parte media del menton

#     DD = d1 + d2 + d3 + d4 + d5 + d6 + d7     #suma de las distancias
# #------------------------------------------------------
#     D1 = d1/DD
#     D2 = d2/DD
#     D3 = d3/DD
#     D4 = d4/DD
#     D5 = d5/DD
#     D6 = d6/DD
#     D7 = d7/DD
# #------------------------------------------------------
#     R1 = D2/D1
#     R2 = D1/D3
#     R3 = D2/D3
#     R4 = D1/D5
#     R5 = D6/D5
#     R6 = D4/D6
#     R7 = D6/D1
#     R8 = D5/D2
#     R9 = D4/D5
#     R10 = D7/D6
# #------------------------------------------------------
#     A1 = angle(midpoint(l70, l73), l9, l11)
#     A2 = angle(midpoint(l70, l73), l9, l13)
#     A3 = angle(l3, l15, l13)
# #------------------------------------------------------
#     features.append(R1)
#     features.append(R2)
#     features.append(R3)
#     features.append(R4) #
#     features.append(R5)
#     features.append(R6)
#     features.append(R7)
#     features.append(R8) #
#     features.append(R9)
#     features.append(R10)
#     features.append(D1)
#     features.append(D2)
#     features.append(D3)
#     features.append(D5)
#     features.append(D6)
#     features.append(D7)
#     features.append(A1) #
#     features.append(A2) #
#     features.append(A3) #

#     print(features)

#     #features = np.array(features)

#   # Dibujar los puntos sobre la imagen
#     for (x, y) in coords:
#         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Punto verde para los landmarks

# # Mostrar la imagen con los puntos de referencia
# # cv2_imshow(image)  # Para Google Colab
# # 
# features = np.array(features)  # Convertir a matriz NumPy
# features = np.expand_dims(features, axis=0)

# # Verificar la forma de features y la entrada esperada por el modelo
# print("Forma de features:", features.shape)
# print("Forma esperada por el modelo:", modelo.input_shape)

# y_predm = modelo.predict(features)

# # Obtener la clase predicha para cada muestra
# clases_predichas = np.argmax(y_predm)

# # Imprimir las clases predichas
# print("Clases predichas:", clases_predichas)

# modelo.save('v2.h5')
# modelo.export('modelo/modelo1')

# !pip install tensorflowjs

# !mkdir carpeta_salida

# !tensorflowjs_converter --input_format keras tipo_rostros.h5 carpeta_salida

# from collections import Counter

# # Contar las ocurrencias de cada clase
# #conteo_clases = Counter(clases_predichas)

# # Diccionario de correspondencia entre índices de clase y etiquetas de clase
# diccionario_clases = {0: "HEART", 1: "OBLONG", 2: "OVAL", 3: "ROUND", 4: "SQUARE"}

# # Obtener la clase más común y su conteo
# #clase_mas_comun, conteo_mas_comun = conteo_clases.most_common(1)[0]

# # Calcular el porcentaje de la clase más común
# #porcentaje_clase_mas_comun = (conteo_mas_comun / len(clases_predichas)) * 100

# etiqueta = diccionario_clases[clases_predichas]

# print(f"Categoría más común: {etiqueta}")
# #print(f"Porcentaje de la categoría más común: {porcentaje_clase_mas_comun:.2f}%")

# #import numpy as np

# #X_train = np.array(faces)
# #y_train = np.array(labels)

# #X_train.size

# #X_train.shape

# # Pre-processing the data

# import numpy as np

# X_train = np.array(faces)
# y_train = np.array(labels)

# # X_train = np.reshape(X_train, (3999, 14))

# X_train = np.reshape(X_train, (192, 14, 1))

# # y_train.reshape(3999, 1)

# # y_train = np.reshape(y_train, (3999, 1))

# X_train = np.reshape(X_train, (192, 14))

# # print(X_train)

# indices = np.arange(X_train.shape[0])
# np.random.shuffle(indices)

# # Reorder the samples using the shuffled indices
# shuffled_X = X_train[indices]
# shuffled_y = y_train[indices]
# # print(X_train)
# # print(shuffled_data.shape)

# """SimpleRNN model"""

# # from keras.models import Sequential
# # from keras.layers import SimpleRNN, Dense
# # from keras.utils import to_categorical

# # # Convert y_train to one-hot encoded labels
# # num_classes = 5
# # y_train_encoded = to_categorical(shuffled_y, num_classes=num_classes)

# # # Define the model
# # smodel = Sequential()
# # smodel.add(SimpleRNN(units=64, activation='relu', input_shape=(14,1)))
# # smodel.add(Dense(units=num_classes, activation='softmax'))

# # # Compile the model
# # smodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # # Train the model
# # smodel.fit(shuffled_X, y_train_encoded, batch_size=800, epochs=100)

# """Random Forest Model"""

# # from sklearn.ensemble import RandomForestClassifier

# # smodel = RandomForestClassifier()

# # smodel.fit(shuffled_X, shuffled_y)

# """Gradient Boosting Model"""

# from sklearn.ensemble import GradientBoostingClassifier
# smodel = GradientBoostingClassifier()

# smodel.fit(shuffled_X, shuffled_y)

# #loading test dataset and evaluating the model

# testdf = pickle.load(open('/content/test.pickle', 'rb'))

# faces_t = testdf['data']
# labels_t = testdf['labels']

# X_test = np.array(faces_t)
# y_test = np.array(labels_t)


# predictions = smodel.predict(X_test)
# # predictions = np.argmax(predictions, axis = 1)

# from sklearn.metrics import accuracy_score
# accuracy_score(y_test, predictions )

# # Saving the model

# #import pickle

# with open('/content/FaceShapeModel.p', 'wb') as ff:
#   # Guardar los datos en el archivo
#     pickle.dump({'model': smodel, 'labels': labels}, ff)