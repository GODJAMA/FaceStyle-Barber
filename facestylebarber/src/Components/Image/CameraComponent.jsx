import React, { useState, useEffect, useRef } from 'react';

const CameraComponent = () => {
  const videoRef = useRef(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const constraints = { video: { facingMode: 'environment' } }; // Solicitar cámara trasera

    const getCameraStream = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError(new Error('getUserMedia is not supported in this browser'));
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setError(err);
      }
    };

    getCameraStream();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  if (error) {
    return <div>Error: {error.message}</div>;
  }

  return (
    <div>
      <h1>Componente de Cámara</h1>
      <video ref={videoRef} autoPlay playsInline style={{ maxWidth: '100%' }} />
    </div>
  );
};

export default CameraComponent;
