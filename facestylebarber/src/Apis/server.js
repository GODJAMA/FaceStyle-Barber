const express = require('express');
const multer = require('multer');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const upload = multer({ dest: 'uploads/' });
const app = express();

app.use(cors()); // Middleware de CORS

// Ruta donde se guardarán las imágenes
const uploadDir = 'imagenes';

// Endpoint para recibir la imagen
app.post('/upload-image', upload.single('image'), (req, res) => {
  // Obtener la ruta temporal de la imagen
  const tempPath = req.file.path;
  // Obtener el nombre original de la imagen
  const originalName = req.file.originalname;
  // Definir la ruta donde se guardará la imagen en tu proyecto
  const targetPath = path.join(uploadDir, originalName);

  try {
    // Verificar si el directorio de destino existe, si no, crearlo
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    // Crear un stream de lectura desde el archivo temporal
    const readStream = fs.createReadStream(tempPath);
    // Crear un stream de escritura hacia la ubicación de destino
    const writeStream = fs.createWriteStream(targetPath);

    // Pipe (conectar) el stream de lectura al stream de escritura
    readStream.pipe(writeStream);

    // Cuando el proceso de escritura haya terminado, eliminar el archivo temporal
    writeStream.on('finish', () => {
      fs.unlinkSync(tempPath); // Eliminar el archivo temporal
      console.log('Imagen guardada en:', targetPath);
      // Envía una respuesta de éxito junto con un mensaje JSON
      res.json({ success: true, message: 'Imagen recibida y guardada exitosamente.' });
    });
  } catch (error) {
    console.error('Error al guardar la imagen:', error);
    // Envía una respuesta de error
    res.status(500).send('Error al guardar la imagen.');
  }
});

// Iniciar el servidor
const PORT = 4000;
app.listen(PORT, () => {
  console.log(`Servidor corriendo en el puerto ${PORT}`);
});
