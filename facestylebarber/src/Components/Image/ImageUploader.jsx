import React, { useState, useRef } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './ImageUploader.css'; // Importar el archivo CSS

function ImageUploader() {
    // Referencia al input de tipo archivo
    const fileInputRef = useRef(null);

    // Estado para almacenar la imagen seleccionada
    const [selectedImage, setSelectedImage] = useState(null);
    // Estado para manejar el estado de la subida de la imagen
    const [imageUploaded, setImageUploaded] = useState(false);

    // Función para manejar el cambio de la imagen seleccionada
    const handleImageChange = (event) => {
        // Obtener la imagen seleccionada del evento
        const image = event.target.files[0];
        // Actualizar el estado con la imagen seleccionada
        setSelectedImage(image);
        // Reiniciar el estado de subida de la imagen
        setImageUploaded(false);
    };

    // Función para manejar el envío de la imagen
    const handleImageSubmit = async () => {
        if (!selectedImage) {
            toast.success('IMAGEN NO SELECCIONADA');
            console.error('No hay ninguna imagen seleccionada.');
            return; // Salir de la función si no hay imagen seleccionada
        }
        try {
            // Crear un objeto FormData para enviar la imagen
            const formData = new FormData();
            formData.append('image', selectedImage);

            // Realizar la solicitud POST a tu API
            const response = await fetch('http://192.168.3.47:4000/upload-image', {
                method: 'POST',
                body: formData
            });

            // Verificar si la solicitud fue exitosa
            if (response.ok) {
                // Mostrar notificación de éxito
                toast.success('¡Imagen subida exitosamente!');
                console.log('Imagen enviada exitosamente a la API.');
                // Establecer el estado de subida de la imagen a verdadero
                setImageUploaded(true);
            } else {
                console.error('Error al enviar la imagen a la API.');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };

    // Función para manejar el borrado de la imagen seleccionada
    const handleDeleteImage = () => {
        setSelectedImage(null); // Borrar la imagen seleccionada
        setImageUploaded(false); // Reiniciar el estado de subida de la imagen
        // Restablecer el valor del input de tipo archivo
        fileInputRef.current.value = null;
        toast.success('IMAGEN BORRADA');
    };

    return (
        <div className="container">
            <h2>Subir Imagen</h2>
            {/* Formulario para seleccionar una imagen */}
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleImageChange}
            />
            {/* Botón para enviar la imagen */}
            <button onClick={handleImageSubmit}>Enviar Imagen</button>
            {/* Botón para borrar la imagen seleccionada */}
            <p></p>
            {selectedImage && (
                <button onClick={handleDeleteImage}>Borrar Imagen</button>
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
            {/* Mostrar mensaje de éxito si la imagen se ha subido correctamente */}
            {imageUploaded && <p>Imagen subida correctamente.</p>}
            {/* Contenedor para mostrar las notificaciones */}
            <ToastContainer />
        </div>
    );
}

export default ImageUploader;
