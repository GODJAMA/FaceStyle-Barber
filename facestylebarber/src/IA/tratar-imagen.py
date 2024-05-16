import cv2
import dlib
import os
import math
import numpy as np
from imutils import face_utils

shape_predictor_path = "/content/shape_predictor_81_face_landmarks"
model_path = os.path.join(shape_predictor_path, "shape_predictor_81_face_landmarks.dat")
model = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()


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

# from google.colab.patches import cv2_imshow  # Para mostrar imágenes en Google Colab
# Ruta de la imagen
image_path = "1.jpg"

detector = dlib.get_frontal_face_detector()

# Leer la imagen
image = cv2.imread(image_path)

# Convertir a escala de grises para detección de rostros
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostros
faces_prueba = detector(gray)

model = dlib.shape_predictor(model_path)
features = []
# Dibujar los puntos de referencia para cada rostro detectado
for face in faces_prueba:
    # Obtener los puntos de referencia
    landmarks = model(gray, face)
    coords = face_utils.shape_to_np(landmarks)  # Convertir a coordenadas NumPy

    l3 = coords[3]          #pomulo izquierda
    l15 = coords[15]        #pomulo derecha
    l70 = coords[70]        #parte alta de la frente, altura de la mitad de la ceja, del lado izquierdo
    l76 = coords[76]        #parte media entre el fin de la ceja y la frente, del lado izquierdo
    l80 = coords[80]        #parte media entre el fin de la ceja y la frente, del lado derecho
    l73 = coords[73]        #parte alta de la frente, altura de la mitad de la ceja, del lado derecho
    l9 = coords[9]          #parte mas baja del rostro, menton o barbilla
    l13 = coords[13]        #mejilla a la altura del labio inferior, del lado derecho
    l5 = coords[5]          #mejilla a la altura del labio inferior, del lado izquierdo
    l7 = coords[7]          #parte alta del menton o barbilla a la altura de donde acaba los labios, del lado izquierdo
    l11 = coords[11]        #parte alta del menton o barbilla a la altura de donde acaba los labios, del lado derecho
    l8 = coords[8]          #parte media del menton o barbilla a la altura de donde acaba los labios, del lado izquierdo
    l10 = coords[10]        #parte media del menton o barbilla a la altura de donde acaba los labios, del lado derecho
#------------------------------------------------------
    d1 = distance(l3, l15)                    #distancia entre los pomulos
    d2 = distance(l76, l80)
    d3 = distance(midpoint(l70, l73), l9)     #distacia de lado a lado de la frente
    d4 = distance(l9, l13)                    #distancia entre la barbilla y la mejilla
    d5 = distance(l5, l13)                    #distancia entre mejillas
    d6 = distance(l7, l11)                    #distancia entre parte alta del menton
    d7 = distance(l8, l10)                    #distancia entre parte media del menton

    DD = d1 + d2 + d3 + d4 + d5 + d6 + d7     #suma de las distancias
#------------------------------------------------------
    D1 = d1/DD
    D2 = d2/DD
    D3 = d3/DD
    D4 = d4/DD
    D5 = d5/DD
    D6 = d6/DD
    D7 = d7/DD
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
    A1 = angle(midpoint(l70, l73), l9, l11)
    A2 = angle(midpoint(l70, l73), l9, l13)
    A3 = angle(l3, l15, l13)
#------------------------------------------------------
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
    features.append(A1) #
    features.append(A2) #
    features.append(A3) #

    print(features)

    #features = np.array(features)

  # Dibujar los puntos sobre la imagen
    for (x, y) in coords:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Punto verde para los landmarks

# Mostrar la imagen con los puntos de referencia
# cv2_imshow(image)  # Para Google Colab
# 
features = np.array(features)  # Convertir a matriz NumPy
features = np.expand_dims(features, axis=0)

# Verificar la forma de features y la entrada esperada por el modelo
print("Forma de features:", features.shape)
