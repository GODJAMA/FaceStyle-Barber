from flask import Flask
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
from flask_cors import CORS
from flask import jsonify

app = Flask(__name__)
CORS(app, resources={r"/socket.io/*": {"origins": "*"}})  # Permitir conexiones desde cualquier origen

# Inicializamos SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Almacenar frames de video o transmitir a otro servicio
def process_frame(frame_data):
    # Decodifica el frame recibido
    nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Aquí puedes procesar el frame, guardarlo, o enviarlo a otro servicio
    cv2.imwrite("frame.jpg", frame)  # Ejemplo: guarda el frame como imagen
    

# Evento de recepción de frame desde WebSocket
@socketio.on("video_frame")
def handle_video_frame(data):
    frame_data = data.get("frame")
    if frame_data:
        process_frame(frame_data)

# Ruta para ver si el servidor está funcionando
@app.route("/")
def index():
    return "Video streaming API is running."



if __name__ == "__main__":
    # Configura HTTPS (si tienes certificados SSL)
    socketio.run(app, host="0.0.0.0", port=4000, debug=True)
    # Para pruebas locales sin SSL, usa esto:
    # socketio.run(app, host="0.0.0.0", port=4000)
