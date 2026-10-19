# Visor de expresiones

Aplicación web para detectar rostros desde la cámara y mostrar cajas, expresiones faciales estimadas y coordenadas en píxeles.

## Uso local

```sh
npm install
npm run dev
```

Abre la URL que imprime Vite. La cámara requiere permiso del navegador y un origen seguro: `localhost` o HTTPS. Al iniciar por primera vez se descargan aproximadamente 67 MB de modelos desde `public/models`; la descarga se conserva en la caché del navegador.

`npm run build` genera los recursos estáticos. Para probar esa compilación usa `npm run preview`. El flujo de GitHub Pages publica automáticamente cada push a `main`.

## Modelos y ejecución

- Detección facial: [YOLOv11n-Face-Detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection), exportación ONNX con licencia Apache-2.0.
- Clasificación de expresiones: exportación ONNX cuantizada de [facial_emotions_image_detection](https://huggingface.co/dima806/facial_emotions_image_detection), licencia Apache-2.0, distribuida en formato ONNX por [Xenova](https://huggingface.co/Xenova/facial_emotions_image_detection).
- ONNX Runtime Web usa WebGPU cuando está disponible y vuelve a WASM si no lo está o si un modelo no es compatible.

Los fotogramas se procesan en el navegador y no se graban ni se envían a la ruta del proxy. Solo se solicitan los archivos de los modelos.

## Alcance

La app detecta rostros; no identifica a las personas. Las etiquetas describen una expresión visible estimada por un clasificador, no determinan cómo se siente alguien. La precisión puede variar con la iluminación, el ángulo y el tamaño del rostro; no se debe usar para decisiones de alto impacto.