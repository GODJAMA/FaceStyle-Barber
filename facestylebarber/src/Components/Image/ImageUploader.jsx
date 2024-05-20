import React, { useState, useRef ,useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './ImageUploader.css'; // Importar el archivo CSS

// import image from '../../../public/image/1.jpg'


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
            const response = await fetch('http://192.168.3.47:4000/upload-image', {
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


                fetchImage();
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
                        <h3>Rostro Alargado</h3>
                        <p>Su rostro es igual que los cuadrados en lo que se refiere a las proporciones, aunque la frente y barbilla serán redondeadas en lugar de ser angulares. Su tipo de rostro es más común de lo que puedes pensar y además del corte que podemos recomendarte te digo que si tienes rostro alargado y te dejas barba te verás muy favorecido</p>
                        <p>Si deseas que tu rostro no se vea tan alargado, deberás llevar un corte en el que se tenga mayor volumen en la parte superior, incluyendo la frente y los lados de la cabeza, manteniendo el cabello de esa zona un poco más largo alrededor de las oreja.</p>
                    </div>
                );
            case "OVALADO": // Rostro Ovalado
                return (
                    <div>
                        <h3>Rostro Ovalado</h3>
                        <p>Su rostro tiene la forma de la cara alargada y redonda. La frente sobresale poco, de la misma manera que la barbilla...</p>
                        <p>Las mejillas dominan su contorno y la barbilla es más corta que la frente. Es un tipo de rostro, junto al cuadrado muy común entre los hombres </p>
                        <p>Su rostro es ideal ya que guarda bastante las proporciones, por lo que cualquier peinado te quedará genial, aunque quizás los más cortos son los que mejor puedan marcar la perfección de tus facciones. Consejo no lleves peinados con mucho flequillo ya que se te verá el rostro más ovalado todavía y de preferencia, elige peinados o cortes en los que el cabello se peine hacia arriba, hacia un lado o hacia atrás, ya que la frente despejada favorece mucho para este tipo de rostro peinados largos.</p>
                    </div>
                );
            case "REDONDO": // Rostro Redondo
                return (
                    <div>
                        <h3>Rostro Redondo</h3>
                        <p>Tal y cómo indica su propia palabra, la cara redonda es la cara que tiene forma completamente redondeada, sin que sobresalgan ni mentón, ni pómulos ni nada. Su rostro, crea una especie de círculo a partir de la anchura que suele tener a la altura de las mejillas, la curva que tiene en la frente y en la barbilla. </p>
                        <p>El mejor corte será un corte largo que permita disimular un poco esa redondez, o concentrar mayor volumen en la parte alta de la cabeza para poder atraer a este punto todas las miradas. Los peinados con raya puede que le favorezcan también o como ves arriba todos aquellos con los que “juegues” en lo que respecta al volumen y hacia arriba de modo que puedas crear el efecto de que las facciones redondeadas quedan algo más disimuladas.</p>
                    </div>
                );
            case "CUADRADO": // Rostro Cuadrado
                return (
                    <div>
                        <h3>Rostro Cuadrado</h3>
                        <p>Su rostro tiene los maxilares muy marcados. Presenta la frente y la barbilla planas. Se asocia con ser el tipo de rostro más masculino de todos. No obstante, le sucede como al triangular, que no es el más común. Dependiendo de si te gustan las facciones marcadas y masculinas puedes elegir cortes que las disimulen o que las marquen más.</p>
                        <p>El corte ideal para su tipo de rostro es poder suavizar un poco sus facciones. Que los ángulos de su cara no se marquen tanto por lo que un corte con volumen en el centro y los lados rapados será sin duda tu corte de cabello perfecto. Los flequillos también pueden quedarte muy bien y como no.</p>
                        <p>Todo lo que sean cortes por encima sus orejas y patillas cortas le van a favorecer de modo que tus facciones no se vean tan cuadradas. Por otro lado, si optas por llevar barba, es una buena idea dejarla un poco larga de modo que la mandíbula se vea más alargada.</p>
                    </div>
                );
            default:
                return null;
        }
    };

    const [imageUrl, setImageUrl] = useState('');

    const fetchImage = async () => {
        try {
            const response = await fetch('http://192.168.3.47:4000/image'); // Cambia la URL según sea necesario
            if (!response.ok) {
                throw new Error('Error al obtener la imagen');
            }
            const blob = await response.blob();
            setImageUrl(URL.createObjectURL(blob));
        } catch (error) {
            console.error('Error al obtener la imagen:', error);
        }
    };
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
        <button className="custom-input-button" onClick={() => fileInputRef.current.click()}>Seleccionar Imagen</button>

        <button className="button submit-button" onClick={handleImageSubmit}>Enviar Imagen</button>

    </div>
            {/* Botón para borrar la imagen seleccionada */}
            <p></p>
            {selectedImage && (
                <button className="button delete-button" onClick={handleDeleteImage}>Borrar Imagen</button>
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
            </div>

            {imageUploaded && faceType && (
                <div className='container'>
                    <h1>Su Tipo De Rostro Predecido: </h1>
                    <h1>{faceType}</h1>
                    {imageUrl && <img className="image" src={imageUrl} alt="Imagen desde la API" />}
                    {/* <img src={processedImagePath}  alt="Imagen procesada" />
                    <img src={`${process.env.PUBLIC_URL}/IA/imagenes/${1}`} alt="Imagen procesada" /> */}
                    {getCorteSugerido(faceType)}
                </div>
            )}

            {/* Contenedor para mostrar las notificaciones */}
            <ToastContainer />
        
        </div>
    );
}

export default ImageUploader;
