from flask import Flask, Response, request
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Permite CORS para que React pueda hacer peticiones a la API

@app.route('/video_feed', methods=['POST'])
def video_feed():
    # Lee el contenido de la solicitud
    data = request.data
    # Decodifica el contenido del video
    np_arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Codifica el fotograma en formato JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    # Devuelve el fotograma en un formato compatible con streaming
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    print(f'Servidor corriendo en http://{host}:{port}')
    app.run(host=host, port=port)
