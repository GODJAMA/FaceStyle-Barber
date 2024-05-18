import React, { useState, useEffect } from 'react';
import './VideoUploader.css';
import axios from 'axios';

function VideoUploader() {
  const [showVideo, setShowVideo] = useState(false);
  const [videoSrc, setVideoSrc] = useState('');
  const [faceInfo, setFaceInfo] = useState('');

  const handleStartAnalysis = () => {
    setShowVideo(true);
    const timestamp = new Date().getTime();
    setVideoSrc(`http://localhost:4000/video_feed?timestamp=${timestamp}`);
  };

  const handleStopAnalysis = () => {
    setShowVideo(false);

    try {
      axios.get('http://localhost:4000/stop_video');
    } catch (error) {
      console.error('Error al detener la captura de video:', error);
    }
  };

  const fetchFaceInfo = () => {    
    axios.get('http://localhost:4000/get_face_info')
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
            <img src={videoSrc} alt="Video Feed" className="centered-video" />
            <p>{faceInfo}</p>
          </div>
        )}
        
      
      
    </div>
  );
}

export default VideoUploader;
