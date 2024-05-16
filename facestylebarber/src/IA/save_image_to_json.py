import base64
import json
import os
# Leer la imagen en bytes
with open("imagenes/1.jpg", "rb") as img_file:
    img_bytes = img_file.read()

# Codificar los bytes de la imagen en base64
img_base64 = base64.b64encode(img_bytes).decode()

# Preparar un diccionario con la imagen codificada en base64
data = {"imagen_base64": img_base64}

# Guardar el diccionario en un archivo JSON
with open("imagen.json", "w") as json_file:
    json.dump(data, json_file)

# Abrir el archivo JSON y cargar su contenido
with open("imagen.json", "r") as json_file:
    json_data = json.load(json_file)
#  Obtener la cadena base64 de la imagen del JSON
img_base64 = json_data["imagen_base64"]

# Decodificar la cadena base64 en bytes
img_bytes = base64.b64decode(img_base64)

# Guardar los bytes de la imagen en un archivo
with open("imagen_recuperada.webp", "wb") as img_file:
    img_file.write(img_bytes)

print("Imagen recuperada guardada como 'imagen_recuperada.webp'")
# Imprimir el contenido del JSON
print(json_data)

import base64
import json
from PIL import Image
from io import BytesIO

# Abrir el archivo JSON y cargar su contenido
with open("imagen.json", "r") as json_file:
    json_data = json.load(json_file)

# Obtener la cadena base64 de la imagen del JSON
img_base64 = json_data["imagen_base64"]

# Decodificar la cadena base64 en bytes
img_bytes = base64.b64decode(img_base64)

# Crear una imagen a partir de los bytes decodificados
image = Image.open(BytesIO(img_bytes))

# Mostrar la imagen
image.show()
