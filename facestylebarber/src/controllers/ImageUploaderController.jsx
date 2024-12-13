import React, { useState, useRef } from 'react';
import { uploadImage, fetchImage } from '../models/ImageUploaderModel'; // Importa las funciones del modelo
import ImageUploaderView from '../view/Components/viewImagen/ImageUploaderView'; // Importa la vista
import {  toast } from 'react-toastify'; // Importa ToastContainer y toast
import 'react-toastify/dist/ReactToastify.css'; // Estilos para el Toast
import getFaceTypeSuggestions from '../models/FaceTypeSuggestions';

const ImageUploaderController = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageUploaded, setImageUploaded] = useState(false);
  const [faceType, setFaceType] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const fileInputRef = useRef(null);
  const [suggestion, setSuggestion] = useState(null);

  // Maneja la selección de imagen
  const handleImageChange = (e) => {
    setSelectedImage(e.target.files[0]);
  };

  // Maneja el envío de la imagen (subir la imagen)
  const handleImageSubmit = async () => {
    if (!selectedImage) {
      toast.error('Por favor, seleccione una imagen'); // Usar ToastContainer para mostrar el error
      return;
    }

    try {
      // Llama a la función uploadImage del modelo para subir la imagen
      const data = await uploadImage(selectedImage);
      
      const proceso =  `../../IA/imagenes/${data.processed_image_path.replace(/\\/g, '/')}`;

      setImageUploaded(true);
      setFaceType(data.face_type); // Asume que el modelo devuelve el tipo de rostro
      setImageUrl(proceso); // Asume que el modelo devuelve la URL de la imagen procesada

      toast.success('¡Imagen subida correctamente!'); // Muestra un mensaje de éxito

      await handleFetchImage(); 
      const faceSuggestion = getFaceTypeSuggestions(data.face_type);
      setSuggestion(faceSuggestion);


    } catch (error) {
      toast.error('Hubo un error al procesar la imagen'); // Muestra un mensaje de error
    }
  };

  // Maneja la eliminación de la imagen seleccionada
  const handleDeleteImage = () => {
    setSelectedImage(null);
    setImageUploaded(false);
    setFaceType(null);
    setImageUrl('');
    toast.info('Imagen eliminada'); // Muestra un mensaje de eliminación
  };

  // Maneja la obtención de la imagen (para mostrarla desde la URL obtenida)
  const handleFetchImage = async () => {
    try {
      const url = await fetchImage(); // Llama a la función fetchImage para obtener la imagen
      
      setImageUrl(url); // Asigna la URL de la imagen obtenida
      toast.success('Imagen obtenida correctamente'); // Muestra un mensaje de éxito
    } catch (error) {
      toast.error('Error al obtener la imagen'); // Muestra un mensaje de error
    }
  };

  return (
    <>
      <ImageUploaderView
        fileInputRef={fileInputRef}
        handleImageChange={handleImageChange}
        handleImageSubmit={handleImageSubmit}
        handleDeleteImage={handleDeleteImage}
        handleFetchImage={handleFetchImage} // Añadimos esta función a la vista
        selectedImage={selectedImage}
        imageUploaded={imageUploaded}
        faceType={faceType}
        imageUrl={imageUrl}
        suggestion={suggestion}
      />
    </>
  );
};

export default ImageUploaderController;
