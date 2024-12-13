
import React from 'react';
import './VideoUploader.css';

const VideoUploaderView = ({
    showVideo,
    videoSrc,
    loading,
    progress,
    faceInfo,
    currentFace,
    handleStartAnalysis,
    handleStopAnalysis,
    getCorteSugerido,
    suggestion
}) => {

    return (
        <div>
            <div style={{ marginBottom: '20px' }}></div>
            <div className="header">
                <h1>Punteo de Rostro</h1>
                {!showVideo && (
                    <div>
                        <button className="start-button" onClick={handleStartAnalysis}>
                            Iniciar análisis de cámara
                        </button>
                        {currentFace && (
                            <div>
                                <h3>Rostro Detectado:</h3>
                                {suggestion ? (
                                    <>
                                        <h3>{suggestion.title}</h3>
                                        <p>{suggestion.description}</p>
                                    </>
                                ) : (
                                    <p>No hay sugerencia disponible</p>
                                )}
                            </div>
                        )}
                    </div>
                )}
                {showVideo && (
                    <button className="stop-button" onClick={handleStopAnalysis}>
                        Detener análisis de cámara
                    </button>
                )}
            </div>

            {
                showVideo && (
                    <div className="video-container">
                        <div className="image-wrapper">
                            {loading && <div className="loading">Conectando...</div>}
                            <progress value={progress} max="100" style={{ visibility: loading ? 'visible' : 'hidden' }} />
                            <img
                                src={videoSrc}
                                alt="Video Feed"
                                className={`centered-video ${loading ? 'hidden' : ''}`}
                            />
                            {!loading && <p>{faceInfo}</p>}
                        </div>
                    </div>
                )
            }
        </div >
    );
};

export default VideoUploaderView;
