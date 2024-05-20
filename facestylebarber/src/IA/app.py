import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, jsonify, Response ,stream_with_context,send_file
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
import json
from PIL import Image

app = Flask(__name__)
CORS(app)

shape_predictor_path = "/content/shape_predictor_81_face_landmarks"
model_path = os.path.join(shape_predictor_path, "shape_predictor_81_face_landmarks.dat")
# model = dlib.shape_predictor(model_path)
# detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("/content/shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat")

diccionario_clases = {0: "CORAZÓN", 1: "ALARGADO", 2: "OVALADO", 3: "REDONDO", 4: "CUADRADO"}


detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor(model_path)
modelo_cargado = keras.models.load_model('v3.h5')


# Ruta donde se guardarán las imágenes
upload_dir = 'imagenes'
image_gobla = ''

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

        global image_gobla

        image_gobla = original_filename
        # print(image_gobla)
        try:
            # Procesar la imagen con Pillow (PIL)
            image = Image.open(image_file)
            # image.thumbnail((200, 300))  # Cambiar el tamaño de la imagen
            # Corregir la orientación de la imagen si es necesario
            if hasattr(image, '_getexif'):  # Verificar si la imagen tiene información EXIF
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(0x0112)
                    if orientation is not None:
                        if orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)

            image.save(os.path.join(upload_dir, original_filename))  # Guardar la imagen con el nombre original
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print('imagen guardada')
            features, processed_image = process_image(os.path.join(upload_dir, original_filename))

            # cv2.imshow('Imagen con puntos de referencia', processed_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            processed_image_path = os.path.join(upload_dir,original_filename)
            cv2.imwrite(processed_image_path, processed_image)

            print('Características extraídas:', features)
            print('Imagen procesada con éxito.')
           
            return jsonify({
                'success': True,
                'message': 'Imagen recibida y procesada exitosamente.',
                'processed_image_path': processed_image_path,
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
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_prueba = detector(gray)


    features = []
    for face in faces_prueba:
        # Obtener los puntos de referencia
        landmarks = model(gray, face)
        coords = face_utils.shape_to_np(landmarks)  # Convertir a coordenadas NumPy

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

        print(features)

    #features = np.array(features)
  # Dibujar los puntos sobre la imagen
    for (x, y) in coords:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Punto verde para los landmarks

    # Mostrar la imagen con los puntos de referencia
    # Convertir la imagen procesada a RGB
    # cv2.imshow('Imagen con puntos de referencia', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 
    features = np.array(features)  # Convertir a matriz NumPy
    features = np.expand_dims(features, axis=0)

    # Verificar la forma de features y la entrada esperada por el modelo
    print("Forma de features:", features.shape)

    #forma de exportart 
# Cargar el modelo desde el directorio del SavedModel

    # Cargar el modelo

    # Realizar predicciones
    predictions = modelo_cargado.predict(features)
    print (predictions)

    clases_predichas = np.argmax(predictions)
    print(clases_predichas )

    return diccionario_clases[clases_predichas],image


video_capture = None  
Rostro = 'Conectando'

def gen_frames():
    global video_capture  # Declarar que se usará la variable global
    global Rostro
    
    if video_capture is None:
        video_capture = cv2.VideoCapture(0) 
    while True:
        # Leer un cuadro del video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convertir a escala de grises para detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = []
        # Detectar rostros
        faces = detector(gray)
        if len(faces) == 0:
            # print('no detesto carra')
            Rostro = 'NO DETECTADO'
        else:
            # Para cada rostro detectado, obtener y dibujar los landmarks
            for face in faces:
                landmarks = predictor(gray, face)
                coords = face_utils.shape_to_np(landmarks)  # Convertir a coordenadas NumPy

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

                # print(features)

                # Dibujar los puntos de referencia en el rostro
            for (x, y) in coords:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Punto verde

            features = np.array(features)  # Convertir a matriz NumPy
            features = np.expand_dims(features, axis=0) 

            predictions = modelo_cargado.predict(features)
            # print (predictions)

            clases_predichas = np.argmax(predictions)
            # print(clases_predichas )

            Rostro = diccionario_clases[clases_predichas]
            print(diccionario_clases[clases_predichas])

        

        # Codificar el cuadro como imagen JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # result = {'predictions': predictions.tolist(), 'clases_predichas': clases_predichas}
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #    b'Content-Type: text/plain\r\n\r\n' + Rostro.encode() + b'\r\n')  # Emitir el cuadro como streaming

@app.route('/video_feed')
def video_feed():
    print('inicio de analizador tiempo real')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_face_info')
def get_face_info():
    global Rostro
    # Aquí incluye la lógica para detectar el rostro
    # Supongamos que Rostro contiene los datos del rostro
    print(Rostro)
    return jsonify({"face_info": Rostro})

@app.route('/stop_video')
def stop_video():
    global video_capture
    global Rostro

    if video_capture is not None:
        video_capture.release()  # Liberar la captura de video
        video_capture = None  # Establecer la variable a None para indicar que la captura ha sido detenida
        Rostro = 'NO DETECTADO'
        print('Liberar la captura de video')
    return 'Video detenido exitosamente'

# Ruta para servir la imagen
# import os
# from flask import send_file

@app.route('/image')
def get_image():
    global image_gobla
    # Obtener la ruta absoluta de la imagen
    # image_path = os.path.abspath('1.jpg')
    # image_path = os.path.abspath('C:/Users/juanA/OneDrive/Documentos/GitHub/FaceStyle-Barber/facestylebarber/src/IA/imagenes/1.jpg')
    # print(';;;;;;;')
    print(image_gobla)
    image_path = os.path.abspath(f'C:/Users/juanA/OneDrive/Documentos/GitHub/FaceStyle-Barber/facestylebarber/src/IA/imagenes/{image_gobla}')
    # Envía la imagen como respuesta
    return send_file(image_path, mimetype='image/jpeg')



# Iniciar el servidor
if __name__ == '__main__':
    host = '0.0.0.0' 
    port = 4000
    app.run(host=host, port=port)
    print(f'Servidor corriendo en el puerto {port}')