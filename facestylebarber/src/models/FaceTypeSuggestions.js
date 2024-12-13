// src/model/faceTypeSuggestions.js

const suggestions = {
  CORAZÓN: {
    title: 'Rostro de Corazón',
    description: 'Su rostro es armónico, con una frente más amplia y un mentón más estrecho. Le recomendamos peinados que añadan volumen en la parte inferior, como rizos o capas largas.',
  },
  ALARGADO: {
    title: 'Rostro Alargado',
    description: 'El rostro alargado tiene proporciones verticales predominantes. Se recomiendan cortes que reduzcan la sensación de longitud, como flequillos y peinados con volumen lateral.',
  },
  OVALADO: {
    title: 'Rostro Ovalado',
    description: 'El rostro ovalado es versátil y equilibrado. Puede probar una amplia variedad de estilos, como peinados con capas o lisos.',
  },
  REDONDO: {
    title: 'Rostro Redondo',
    description: 'El rostro redondo tiene proporciones similares en ancho y largo. Le sugerimos peinados que alarguen visualmente el rostro, como capas largas o estilos con volumen en la parte superior.',
  },
  CUADRADO: {
    title: 'Rostro Cuadrado',
    description: 'El rostro cuadrado tiene una mandíbula definida y proporciones equilibradas. Le favorecen peinados con ondas suaves y cortes desfilados.',
  },
};

// Función que devuelve las sugerencias para un tipo de rostro
const getFaceTypeSuggestions = (faceType) => {
  const suggestion = suggestions[faceType];
  return suggestion || { title: 'Sin sugerencias', description: 'No se encontraron sugerencias para este tipo de rostro.' };
};

export default getFaceTypeSuggestions;
