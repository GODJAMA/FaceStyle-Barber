import React from 'react';
import { ToastContainer } from 'react-toastify';
import './ImageUploader.css';

const ImageUploaderView = React.memo(({
  fileInputRef,
  handleImageChange,
  handleImageSubmit,
  handleDeleteImage,
  selectedImage,
  imageUploaded,
  faceType,
  imageUrl,
  suggestion
}) => {
    
  return (
    <div>
      <div className="header">
        <h1>Analizador Punteo de Rostro</h1>
        <h3>Subir Imagen</h3>
        
        {/* Formulario para seleccionar una imagen */}
        <div className="custom-input-container">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="input-file"
            capture="environment"
            onChange={handleImageChange}
          />
          
          {/* Botón personalizado */}
          <button className="custom-input-button" onClick={() => fileInputRef.current.click()}>
            Seleccionar Imagen
          </button>

          <button className="button submit-button" onClick={handleImageSubmit}>
            Enviar Imagen
          </button>
        </div>

        {/* Botón para borrar la imagen seleccionada */}
        <p></p>
        {selectedImage && (
          <button className="button delete-button" onClick={handleDeleteImage}>
            Borrar Imagen
          </button>
        )}

        {/* Mostrar la imagen seleccionada */}
        {selectedImage && (
          <div className="image-container">
            <h3>Imagen Seleccionada:</h3>
            <img
              src={URL.createObjectURL(selectedImage)}
              alt="Selected"
              className="image"
            />
          </div>
        )}

        {/* Mostrar el tipo de rostro si la imagen se ha subido correctamente */}
        {imageUploaded && faceType && (
          <div className="container">
            <h1>Su Tipo De Rostro Predecido:</h1>
            <h1>{faceType}</h1>
            {imageUrl && <img className="image" src={imageUrl} alt="Imagen desde la API" />}
            
            {suggestion && (
              <div className="sugerencias-corte">
                <h3>{suggestion.title}</h3>
                <p>{suggestion.description}</p>
              </div>
            )}
          </div>
        )}

        {/* Contenedor para mostrar las notificaciones */}
        <ToastContainer />
      </div>
    </div>
  );
});

export default ImageUploaderView;
