import React, { useState, useEffect } from 'react';
import './VideoUploader.css';
import axios from 'axios';

function VideoUploader() {
  const [showVideo, setShowVideo] = useState(false);
  const [videoSrc, setVideoSrc] = useState('');
  const [faceInfo, setFaceInfo] = useState('');
  const [loading, setLoading] = useState(false); // Estado de carga de la imagen
  const [progress, setProgress] = useState(0); // Estado de progreso de carga

  

  const handleStartAnalysis = () => {
    setShowVideo(true);
    const timestamp = new Date().getTime();
    setVideoSrc(`http://192.168.3.47:4000/video_feed?timestamp=${timestamp}`);
  };

  const handleStopAnalysis = () => {
    setShowVideo(false);

    try {
      axios.get('http://192.168.3.47:4000/stop_video');
    } catch (error) {
      console.error('Error al detener la captura de video:', error);
    }
  };

  const fetchFaceInfo = () => {    
    axios.get('http://192.168.3.47:4000/get_face_info')
      .then(response => {
        setFaceInfo(response.data.face_info);
      })
      .catch(error => {
        console.error('Error al obtener datos del rostro:', error);
      });
  };

  useEffect(() => {
    // Realizar una solicitud para obtener los datos del rostro cada segundo
    // console.log(showVideo);
    if  (showVideo) {
    const interval = setInterval(fetchFaceInfo, 1000);
   
    return () => {
      clearInterval(interval);
    };
  } 
  }, [showVideo]); // Se ejecuta solo una vez al montar el componente


  useEffect(() => {
    setLoading(true); // Establecer el estado de carga a verdadero al iniciar la carga de la imagen
  }, [videoSrc]); // Detectar cambios en la URL de la imagen


  const handleImageLoadStart = () => {
    setProgress(0); // Resetear el progreso de carga al iniciar la carga de la imagen
  };

  const handleImageLoadProgress = (event) => {
    if (event.lengthComputable) {
      const progress = (event.loaded / event.total) * 100;
      setProgress(progress); // Actualizar el progreso de carga mientras se carga la imagen
    }
  };

  const handleImageLoad = () => {
    setLoading(false); // Establecer el estado de carga a falso cuando se complete la carga de la imagen
  };

  return (
    <div>
      <div style={{ marginBottom: '20px' }}></div>
      <div className="header">
        <h1>Punteo de Rostro</h1>
        {!showVideo && (
          <button className="start-button" onClick={handleStartAnalysis}>
            Iniciar análisis de cámara
          </button>
        )}
        {showVideo && (
          <button className="stop-button" onClick={handleStopAnalysis}>
            Detener análisis de cámara
          </button>
        )}
      </div>

     {showVideo && (
        <div className="video-container">
          <div className="image-wrapper">
            {loading && <div className="loading">Conectando...</div>}
            <progress value={progress} max="100" style={{ visibility: loading ? 'visible' : 'hidden' }} />
            <img
              src={videoSrc}
              alt="Video Feed"
              className={`centered-video ${loading ? 'hidden' : ''}`}
              onLoadStart={handleImageLoadStart}
              onProgress={handleImageLoadProgress}
              onLoad={handleImageLoad}
            />
            {!loading && <p>{faceInfo}</p>} {/* Mostrar faceInfo solo cuando la imagen se haya cargado completamente */}
          </div>
        </div>
      )}
        
      
      
    </div>
  );
}

export default VideoUploader;