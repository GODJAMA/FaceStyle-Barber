import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PIL import Image
import keras
import base64
from io import BytesIO
import cv2
import dlib
import math
import numpy as np
from imutils import face_utils

app = Flask(__name__)
CORS(app)

shape_predictor_path = "/content/shape_predictor_81_face_landmarks"
model_path = os.path.join(shape_predictor_path, "shape_predictor_81_face_landmarks.dat")
model = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("/content/shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat")

# Ruta donde se guardarán las imágenes
upload_dir = 'imagenes'


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



# Endpoint para recibir la imagen
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        # Verificar si el directorio de destino existe, si no, crearlo
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        # Obtener la imagen desde la solicitud
        image_file = request.files['image']
        original_filename = image_file.filename

        try:
            # Procesar la imagen con Pillow (PIL)
            image = Image.open(image_file)
            # image.thumbnail((200, 300))  # Cambiar el tamaño de la imagen
            image.save(os.path.join(upload_dir, original_filename))  # Guardar la imagen con el nombre original
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            features, processed_image = process_image(os.path.join(upload_dir, original_filename))

            print('Características extraídas:', features)
            print('Imagen procesada con éxito.')

            # Guardar la imagen procesada utilizando Pillow (PIL)
            processed_image_pil = Image.fromarray(processed_image)
            processed_image_path = os.path.join(upload_dir, 'processed_' + original_filename)
            processed_image_pil.save(processed_image_path)

            # Convertir la imagen procesada a bytes
            # buffered = BytesIO()
            # processed_image.save(buffered, format="JPEG")
            # img_bytes = buffered.getvalue()

            # # Codificar los bytes de la imagen en base64
            # img_base64 = base64.b64encode(img_bytes).decode()


            # Devolver una respuesta exitosa al cliente
            return jsonify({
                'success': True,
                'message': 'Imagen recibida y procesada exitosamente.',
                'processed_image_path': processed_image_path,
                # 'processed_image_base64': img_base64,
                'face_type': features
            }), 200
        except Exception as e:
            print('Error al procesar la imagen:', str(e))
            raise e

    except Exception as e:
        print('Error al procesar la imagen:', str(e))
        # Envía una respuesta de error
        return 'Error al procesar la imagen.', 500
    

def process_image(image_path):
    detector = dlib.get_frontal_face_detector()
    model = dlib.shape_predictor(model_path)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_prueba = detector(gray)

    features = []
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
    # Convertir la imagen procesada a RGB
    
    # 
    features = np.array(features)  # Convertir a matriz NumPy
    features = np.expand_dims(features, axis=0)

    # Verificar la forma de features y la entrada esperada por el modelo
    print("Forma de features:", features.shape)

    #forma de exportart 
# Cargar el modelo desde el directorio del SavedModel

    # Cargar el modelo
    modelo_cargado = keras.models.load_model('tipo_rostros.h5')

    # Realizar predicciones
    predictions = modelo_cargado.predict(features)
    print (predictions)

    clases_predichas = np.argmax(predictions)
    print(clases_predichas )
    diccionario_clases = {0: "HEART", 1: "OBLONG", 2: "OVAL", 3: "ROUND", 4: "SQUARE"}

    return diccionario_clases[clases_predichas],image


video_capture = cv2.VideoCapture(0)  # Usar '0' para la cámara predeterminada

def gen_frames():
    while True:
        # Leer un cuadro del video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convertir a escala de grises para detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = detector(gray)

        # Para cada rostro detectado, obtener y dibujar los landmarks
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks_np = face_utils.shape_to_np(landmarks)  # Convertir a coordenadas NumPy

            # Dibujar los puntos de referencia en el rostro
            for (x, y) in landmarks_np:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Punto verde

        # Codificar el cuadro como imagen JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Emitir el cuadro como streaming

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



# Iniciar el servidor
if __name__ == '__main__':
    port = 4000
    app.run(port=port)
    print(f'Servidor corriendo en el puerto {port}')

