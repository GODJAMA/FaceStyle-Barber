const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const upload = multer({ dest: 'uploads/' });
const app = express();

app.use(cors()); // Middleware de CORS

// Ruta donde se guardarán las imágenes
const uploadDir = 'imagenes';

// Función para procesar la imagen con sharp
async function procesarImagen(imagenBuffer) {
  try {
    // Procesar la imagen con sharp
    await sharp(imagenBuffer)
      .resize(200, 300) // Por ejemplo, cambiar el tamaño de la imagen
      .toFile(path.join(__dirname, 'imagenes', 'imagen_procesada.jpg'));
    console.log('Imagen procesada con éxito.');
  } catch (error) {
    console.error('Error al procesar la imagen:', error);
    throw error;
  }
}

// Endpoint para recibir la imagen
app.post('/upload-image', upload.single('image'), async (req, res) => {
  // Obtener la ruta temporal de la imagen
  const tempPath = req.file.path;

  try {
    // Verificar si el directorio de destino existe, si no, crearlo
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    // Leer la imagen desde la carpeta imagenes
    const imagenBuffer = fs.readFileSync(tempPath);

    // Procesar la imagen
    await procesarImagen(imagenBuffer);

    // Devolver una respuesta exitosa al cliente
    res.json({ success: true, message: 'Imagen recibida y procesada exitosamente.' });
  } catch (error) {
    console.error('Error al procesar la imagen:', error);
    // Envía una respuesta de error
    res.status(500).send('Error al procesar la imagen.');
  }
});

// Iniciar el servidor
const PORT = 4000;
app.listen(PORT, () => {
  console.log(`Servidor corriendo en el puerto ${PORT}`);
});
