import React, { useState } from 'react';
import './main.css';
// import ImageUploader from 'Image/ImageUploader';
// import VideoUploader from 'Image/VideoUploader';
import ImageUploader from './Image/ImageUploader';
import VideoUploader from './Image/videoUploader';



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
        <p></p>
        {selectedOption === 'foto' && <ImageUploader />}
        {selectedOption === 'video' && <VideoUploader />}
      </div>
    </div>
  );
}

export default Main;
