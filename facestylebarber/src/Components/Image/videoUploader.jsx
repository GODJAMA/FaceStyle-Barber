import React, { useState } from 'react';
import './VideoUploader.css';

function VideoUploader() {
  const [showVideo, setShowVideo] = useState(false);

  const handleStartAnalysis = () => {
    setShowVideo(true);
  };

  const handleStopAnalysis = () => {
    setShowVideo(false);
  };

  return (
    <div>
      <div style={{ marginBottom: '20px' }}></div> {/* Espacio en blanco antes del header */}
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
      <div className="video-container">
        {showVideo && (
          <img src="http://localhost:4000/video_feed" alt="Video Feed" className="centered-video" />
        )}
      </div>
    </div>
  );
}

export default VideoUploader;
