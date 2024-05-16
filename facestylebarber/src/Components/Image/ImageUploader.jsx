import React, { useState, useRef ,useEffect } from 'react';
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
    // Estado para almacenar el tipo de rostro
    const [faceType, setFaceType] = useState(null);
    // Definir el estado para la ruta de la imagen procesada

    const [processedImagePath, setProcessedImagePath] = useState('');

    useEffect(() => {
        // Solo actualizar la ruta si la imagen ha sido cargada y procesada
        if (imageUploaded && processedImagePath) {
            console.log("Imagen procesada:", processedImagePath);
        }
    }, [imageUploaded, processedImagePath]);


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
            toast.info('IMAGEN NO SELECCIONADA');
            console.error('No hay ninguna imagen seleccionada.');
            return; // Salir de la función si no hay imagen seleccionada
        }
        try {
            // Crear un objeto FormData para enviar la imagen
            const formData = new FormData();
            formData.append('image', selectedImage);

            // Realizar la solicitud POST a tu API
            const response = await fetch('http://127.0.0.1:4000/upload-image', {
                method: 'POST',
                body: formData
            });

            // Verificar si la solicitud fue exitosa
            if (response.ok) {
                // Mostrar notificación de éxito
                const responseData = await response.json();
                toast.success('¡Imagen subida exitosamente!');
                console.log('Imagen enviada exitosamente a la API.');

                console.log(responseData);
                // Establecer el estado de subida de la imagen a verdadero
                setImageUploaded(true);
                setFaceType(responseData.face_type);

                // Construir la ruta completa de la imagen en el cliente
                const proceso =  `../../IA/imagenes/${responseData.processed_image_path.replace(/\\/g, '/')}`;
                setProcessedImagePath(proceso);
                console.log(processedImagePath);
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
        toast.info('IMAGEN BORRADA');
    };

    // Función para verificar la conectividad con el servidor Flask en el puerto 5000
    

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
            
            {/* Mostrar el tipo de rostro si la imagen se ha subido correctamente */}
           
            {/* <img src="../../IA/imagenes/processed_corazon.webp" alt="Corazón procesado"/> */}

            {imageUploaded && faceType && (
                <div>
                    <h1>Tipo de rostro: {faceType}</h1>
                    <img src={processedImagePath}  alt="Imagen procesada" />
                    <img src={`${process.env.PUBLIC_URL}/IA/imagenes/${1}`} alt="Imagen procesada" />
                </div>
            )}

            {/* Contenedor para mostrar las notificaciones */}
            <ToastContainer />
        </div>
    );
}

export default ImageUploader;
