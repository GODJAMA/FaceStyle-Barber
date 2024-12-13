

import axios from 'axios';
import { API_BASE_URL } from './config';

const VideoUploaderModel = {
  videoSrc: '',
  faceInfo: '',
  loading: false,
  progress: 0,
  currentFace: '',

  // Método para iniciar la captura del video
  startVideoAnalysis() {
    const timestamp = new Date().getTime();
    this.videoSrc = `${API_BASE_URL}/video_feed?timestamp=${timestamp}`;
    return this.videoSrc;
  },

  // Método para detener la captura del video
  stopVideoAnalysis() {
    axios.get(`${API_BASE_URL}/stop_video`).catch(error => {
      console.error('Error al detener la captura de video:', error);
    });
  },

  // Método para obtener la información del rostro
  fetchFaceInfo() {
    return axios.get(`${API_BASE_URL}/get_face_info`)
      .then(response => {
        this.faceInfo = response.data.face_info;
        if (this.faceInfo !== "NO DETECTADO") {
          this.currentFace = this.faceInfo; // Guardar el rostro actual solo si es detectado
        }
      })
      .catch(error => {
        console.error('Error al obtener datos del rostro:', error);
      });
  },

  // Métodos relacionados con la carga de imágenes
  setLoading(value) {
    this.loading = value;
  },

  setProgress(value) {
    this.progress = value;
  },

  getCurrentFace() {
    return this.currentFace;
  },
};

export default VideoUploaderModel;
