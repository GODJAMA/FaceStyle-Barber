from flask import Flask, jsonify
import cv2
import dlib
from imutils import face_utils

app = Flask(__name__)

# Cargar el detector de rostros y el predictor de landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/shape_predictor_81_face_landmarks/shape_predictor_81_face_landmarks.dat")

# Definir la función para procesar el video
def process_video():
    # Capturar video desde la webcam
    video_capture = cv2.VideoCapture(0)  # Usa '0' para la cámara predeterminada

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

        # Mostrar el cuadro con los landmarks
        cv2.imshow("Video", frame)

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura de video y cerrar las ventanas
    video_capture.release()
    cv2.destroyAllWindows()

# Ruta para procesar el video y devolver el resultado
@app.route('/process_video', methods=['GET'])
def process_video_route():
    process_video()
    return jsonify({'message': 'Video processing completed'})

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)
