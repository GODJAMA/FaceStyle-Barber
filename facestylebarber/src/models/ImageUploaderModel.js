import { API_BASE_URL } from './config'; // Ajusta la ruta según corresponda

// Función para subir la imagen
export const uploadImage = async (image) => {
  const formData = new FormData();
  formData.append('image', image);

  try {
    const response = await fetch(`${API_BASE_URL}/upload-image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Error al subir la imagen');
    }

    const data = await response.json();
    return data; // Retorna la respuesta de la API (puede contener datos como imageId, etc.)
  } catch (error) {
    throw new Error('Hubo un problema al subir la imagen: ' + error.message);
  }
};
// Función para obtener la imagen
export const fetchImage = async () => {
  try {
    // Realizamos un fetch para obtener la imagen desde la API
    const response = await fetch(`${API_BASE_URL}/image`); // Cambia la URL según sea necesario

    // Verificar si la respuesta fue exitosa
    if (!response.ok) {
      throw new Error('Error al obtener la imagen');
    }

    // Convertir la respuesta en un blob (una representación binaria de la imagen)
    const blob = await response.blob();

    // Crear una URL local para mostrar la imagen
    const imageUrl = URL.createObjectURL(blob);

    // Retornar la URL de la imagen o asignarla a un estado si estás usando React
    return imageUrl; // O usa setImageUrl(imageUrl) si estás en un componente de React

  } catch (error) {
    console.error('Error al obtener la imagen:', error);
  }
};
