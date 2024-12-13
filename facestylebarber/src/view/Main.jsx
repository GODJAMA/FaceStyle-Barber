import React, { useState } from 'react';
import './main.css';

// import ImageUploaderView from './viewImagen/ImageUploaderView';
import ImageUploaderController from '../controllers/ImageUploaderController.jsx';
import VideoUploaderController from '../controllers/VideoUploaderController.jsx';

function Main() {
  const [selectedOption, setSelectedOption] = useState(null);

  const handleOptionClick = (option) => {
    setSelectedOption(option);
  };

  return (
    <div>
      <div style={{ marginBottom: '20px' }}></div>
      <div className="container">
        <h1>Selecciona una opción:</h1>
        <div className="options">
          <div className="option" onClick={() => handleOptionClick('foto')}>
            Foto
          </div>
          <div className="option" onClick={() => handleOptionClick('video')}>
            Video
          </div>
        </div>
        
        {selectedOption === 'foto' && <ImageUploaderController />}
        {selectedOption === 'video' && <VideoUploaderController />}

      </div>
    </div>
  );
}

export default Main;
