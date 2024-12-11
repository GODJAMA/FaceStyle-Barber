import React, { useRef, useEffect, useState, useCallback } from "react";
import io from "socket.io-client";
import { API_BASE_URL } from '../config';
import { data, log } from "@tensorflow/tfjs";

function VideoUploader2() {
  const videoRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [processedImage, setProcessedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [socket, setSocket] = useState(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);


  // Establece la conexión WebSocket solo una vez
  useEffect(() => {
    // Crear la conexión del socket solo una vez cuando se monte el componente
    const newSocket = io(API_BASE_URL, { transports: ["websocket"] });
    setSocket(newSocket);

    // Manejo de eventos de conexión
    newSocket.on("connect", () => {
      console.log("Conexión WebSocket establecida");
    });

    newSocket.on("connect_error", (error) => {
      console.error("Error al conectar:", error);
    });

    // Recibir datos desde el servidor
    newSocket.on("start_prediction_stream", (data) => {
      console.log("Datos recibidos desde el servidor:", data);

      // Verificar si recibimos ambos datos: imagen y predicción
      if (data.frame && data.prediction) {
        setProcessedImage(`data:image/jpeg;base64,${data.frame}`);
        setPrediction(data.prediction);
        console.log("Predicción:", data.prediction);
      } else {
        console.error("No se recibió la imagen o la predicción.");
      }


    });
    console.log("Predicción actualizada:", prediction);
    // Limpiar al desconectar
    return () => {
      newSocket.disconnect();
      console.log("Desconectado del WebSocket");
    };
  }, []);

  // Define stopStreaming con useCallback
  const stopStreaming = useCallback(() => {
    if (streaming) {
      clearInterval(intervalRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      setStreaming(false);
      setProcessedImage(null);
      setPrediction(null); // Limpia la predicción al detener el streaming
    }
  }, [streaming]);

  const startStreaming = () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error("getUserMedia no es compatible en este navegador.");
      alert("Tu navegador no soporta el acceso a la cámara.");
      return;
    }

    if (!streaming && socket) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          streamRef.current = stream;

          intervalRef.current = setInterval(() => {
            if (socket && socket.connected) {
              const canvas = document.createElement("canvas");
              const videoWidth = videoRef.current.videoWidth || 320;
              const videoHeight = videoRef.current.videoHeight || 240;
              canvas.width = videoWidth;
              canvas.height = videoHeight;
              const context = canvas.getContext("2d");
              context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

              const frameData = canvas.toDataURL("image/jpeg", 0.4).split(",")[1];

              // socket.emit("start_prediction_stream", { frame: frameData }, data);
              // Supongamos que `frameData` es la imagen en base64 que vas a enviar

              socket.emit("start_prediction_stream", { frame: frameData }, function(responseData) {
                // Esta función de callback se ejecutará cuando el servidor responda
                console.log("Datos procesados recibidos:", responseData);
                if (responseData !== undefined) {
                  setProcessedImage(responseData.prediccion.frame);
                  setPrediction(responseData.prediccion.prediction);
                } else {
                    console.error("No se recibieron datos del servidor.");
                }
            });
            



            }
          }, 100); // Ajusta la frecuencia de captura
        })
        .catch(error => {
          console.error("Error al acceder a la cámara:", error);
          alert("No se pudo acceder a la cámara. Verifica los permisos.");
        });

      setStreaming(true);
    }
  };


  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  return (
    <div style={{ display: "flex", justifyContent: "center", gap: "20px" }}>
      {/* Cuadro de la cámara */}
      <div style={{ position: "relative", width: "320px", height: "240px" }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          style={{
            width: "100%",
            height: "100%",
            border: "2px solid black",
          }}
        />
        <div style={{ textAlign: "center", marginTop: "10px" }}>Cámara</div>
      </div>

      {/* Cuadro de la imagen procesada */}
      <div style={{ position: "relative", width: "320px", height: "240px" }}>
        {processedImage ? (
          <img
            src={processedImage}
            alt="Imagen procesada"
            style={{
              width: "100%",
              height: "100%",
              border: "2px solid black",
            }}
          />
        ) : (
          <div
            style={{
              width: "100%",
              height: "100%",
              border: "2px solid black",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              color: "gray",
              fontSize: "16px",
            }}
          >
            Esperando imagen procesada...
          </div>
        )}
        <div style={{ textAlign: "center", marginTop: "10px" }}>Procesado</div>
        {prediction && (
          <div
            style={{
              position: "absolute",
              bottom: 10,
              left: 10,
              color: "white",
              backgroundColor: "rgba(0, 0, 0, 0.7)",
              padding: "5px",
              borderRadius: "5px"
            }}
          >
            Predicción: {prediction}
          </div>
        )}
      </div>

      {/* Cuadro de texto para la predicción
      {prediction && (
        <div style={{ width: "320px", marginTop: "20px" }}>
          <textarea
            value={prediction}
            readOnly
            rows={4}
            style={{
              width: "100%",
              height: "100px",
              padding: "10px",
              border: "2px solid black",
              borderRadius: "5px",
              resize: "none",
            }}
          />
        </div>
      )} */}

      {/* Botón de inicio y detención */}
      <div className="button-container" style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
        <button
          onClick={streaming ? stopStreaming : startStreaming}
          className="button"
          style={{
            padding: "10px 20px",
            fontSize: "16px",
            cursor: "pointer",
            borderRadius: "5px",
          }}
        >
          {streaming ? "Detener Transmisión" : "Iniciar Transmisión"}
        </button>
      </div>
    </div>
  );
}

export default VideoUploader2;
