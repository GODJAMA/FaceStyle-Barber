import React, { useState, useEffect } from 'react';
import './VideoUploader.css';
import axios from 'axios';
import { API_BASE_URL } from '../config';



function VideoUploader() {
  const [showVideo, setShowVideo] = useState(false);
  const [videoSrc, setVideoSrc] = useState('');
  const [faceInfo, setFaceInfo] = useState('');
  const [loading, setLoading] = useState(false); // Estado de carga de la imagen
  const [progress, setProgress] = useState(0); // Estado de progreso de carga
  const [currentFace, setCurrentFace] = useState('');
  

  const handleStartAnalysis = () => {
    setShowVideo(true);
    const timestamp = new Date().getTime();
    setVideoSrc(`${API_BASE_URL}/video_feed?timestamp=${timestamp}`);
  };

  const handleStopAnalysis = () => {
    setShowVideo(false);

    try {
      axios.get(`${API_BASE_URL}/stop_video`);
    } catch (error) {
      console.error('Error al detener la captura de video:', error);
    }
  };

  const fetchFaceInfo = () => {    
    axios.get(`${API_BASE_URL}/get_face_info`)
      .then(response => {
        setFaceInfo(response.data.face_info);
        if (response.data.face_info !== "NO DETECTADO") {
          setCurrentFace(response.data.face_info); // Guardar el rostro actual solo si es detectado
        }
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
  // const handleShowCurrentFace = () => {
  //   setFaceInfo(currentFace);
  // };

  // Función para obtener las sugerencias de corte según el tipo de rostro
  const getCorteSugerido = (faceType) => {
    console.log(faceType);
    switch (faceType) {
        case "CORAZÓN": // Rostro Corazón
            return (
                <div>
                    <h3>Rostro de Corazón</h3>
                    <p>Su rostro se considera bastante armónico ya que la frente es más ancha que los maxilares o los pómulos, los cuáles suelen marcarse bastante.</p>
                    <p>Los flequillos son los mejores aliados, ya que te permitirán disimular la frente. También los peinados de raya, y los de efecto algo despeinado.</p>
                </div>
            );
        case "ALARGADO": // Rostro Alargado
            return (
                <div>
                    <h3> Alargado</h3>
                    <p>Su rostro es igual que los cuadrados en lo que se refiere a las proporciones, aunque la frente y barbilla serán redondeadas en lugar de ser angulares. Su tipo de rostro es más común de lo que puedes pensar y además del corte que podemos recomendarte te digo que si tienes rostro alargado y te dejas barba te verás muy favorecido</p>
                    <p>Si deseas que tu rostro no se vea tan alargado, deberás llevar un corte en el que se tenga mayor volumen en la parte superior, incluyendo la frente y los lados de la cabeza, manteniendo el cabello de esa zona un poco más largo alrededor de las oreja.</p>
                </div>
            );
        case "OVALADO": // Rostro Ovalado
            return (
                <div>
                    <h3> Ovalado</h3>
                    <p>Su rostro tiene la forma de la cara alargada y redonda. La frente sobresale poco, de la misma manera que la barbilla...</p>
                    <p>Las mejillas dominan su contorno y la barbilla es más corta que la frente. Es un tipo de rostro, junto al cuadrado muy común entre los hombres </p>
                    <p>Su rostro es ideal ya que guarda bastante las proporciones, por lo que cualquier peinado te quedará genial, aunque quizás los más cortos son los que mejor puedan marcar la perfección de tus facciones. Consejo no lleves peinados con mucho flequillo ya que se te verá el rostro más ovalado todavía y de preferencia, elige peinados o cortes en los que el cabello se peine hacia arriba, hacia un lado o hacia atrás, ya que la frente despejada favorece mucho para este tipo de rostro peinados largos.</p>
                </div>
            );
        case "REDONDO": // Rostro Redondo
            return (
                <div>
                    <h3> Redondo</h3>
                    <p>Tal y cómo indica su propia palabra, la cara redonda es la cara que tiene forma completamente redondeada, sin que sobresalgan ni mentón, ni pómulos ni nada. Su rostro, crea una especie de círculo a partir de la anchura que suele tener a la altura de las mejillas, la curva que tiene en la frente y en la barbilla. </p>
                    <p>El mejor corte será un corte largo que permita disimular un poco esa redondez, o concentrar mayor volumen en la parte alta de la cabeza para poder atraer a este punto todas las miradas. Los peinados con raya puede que le favorezcan también o como ves arriba todos aquellos con los que “juegues” en lo que respecta al volumen y hacia arriba de modo que puedas crear el efecto de que las facciones redondeadas quedan algo más disimuladas.</p>
                </div>
            );
        case "CUADRADO": // Rostro Cuadrado
            return (
                <div>
                    <h3> Cuadrado</h3>
                    <p>Su rostro tiene los maxilares muy marcados. Presenta la frente y la barbilla planas. Se asocia con ser el tipo de rostro más masculino de todos. No obstante, le sucede como al triangular, que no es el más común. Dependiendo de si te gustan las facciones marcadas y masculinas puedes elegir cortes que las disimulen o que las marquen más.</p>
                    <p>El corte ideal para su tipo de rostro es poder suavizar un poco sus facciones. Que los ángulos de su cara no se marquen tanto por lo que un corte con volumen en el centro y los lados rapados será sin duda tu corte de cabello perfecto. Los flequillos también pueden quedarte muy bien y como no.</p>
                    <p>Todo lo que sean cortes por encima sus orejas y patillas cortas le van a favorecer de modo que tus facciones no se vean tan cuadradas. Por otro lado, si optas por llevar barba, es una buena idea dejarla un poco larga de modo que la mandíbula se vea más alargada.</p>
                </div>
            );
        default:
            return null;
    }
};

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
              {getCorteSugerido(currentFace)}
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