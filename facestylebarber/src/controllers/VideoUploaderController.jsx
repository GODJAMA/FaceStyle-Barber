
import { useState, useEffect } from 'react';
import VideoUploaderModel from '../models/VideoUploaderModel';
import VideoUploaderView from '../view/Components/ViewVideoUploader/VideoUploaderView';
import getFaceTypeSuggestions from '../models/FaceTypeSuggestions';


const VideoUploaderController = () => {
    const [showVideo, setShowVideo] = useState(false);
    const [videoSrc, setVideoSrc] = useState('');
    const [faceInfo, setFaceInfo] = useState('');
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [currentFace, setCurrentFace] = useState('');
    const [suggestion, setSuggestion] = useState(null);

    useEffect(() => {
        if (showVideo) {
            const interval = setInterval(fetchFaceInfo, 1000);
            return () => clearInterval(interval);
        }
    }, [showVideo]);

    const handleStartAnalysis = () => {
        setShowVideo(true);
        const videoSrc = VideoUploaderModel.startVideoAnalysis();
        setVideoSrc(videoSrc);
    };

    const handleStopAnalysis = () => {
        setShowVideo(false);
        VideoUploaderModel.stopVideoAnalysis();
        getCorteSugerido(currentFace);
    };

    const fetchFaceInfo = () => {
        VideoUploaderModel.fetchFaceInfo().then(() => {
            setFaceInfo(VideoUploaderModel.faceInfo);
            if (VideoUploaderModel.faceInfo !== "NO DETECTADO") {
                setCurrentFace(VideoUploaderModel.currentFace);
            }
        });
    };

    const getCorteSugerido = (faceType) => {
        if (!faceType) {  // Verifica si faceType es null o undefined
            return;  // No hace nada si faceType es null
        }
        const faceSuggestion = getFaceTypeSuggestions(faceType);
        if (faceSuggestion) {
            setSuggestion(faceSuggestion); // Si la sugerencia es válida, actualiza el estado
        } else {
            setSuggestion(null); // Si no se encuentra sugerencia, se establece en null
        }
    };

    return (
        <VideoUploaderView
            showVideo={showVideo}
            videoSrc={videoSrc}
            loading={loading}
            progress={progress}
            faceInfo={faceInfo}
            currentFace={currentFace}
            handleStartAnalysis={handleStartAnalysis}
            handleStopAnalysis={handleStopAnalysis}
            getCorteSugerido={getCorteSugerido}
            suggestion={suggestion}
        />
    );
};

export default VideoUploaderController;
