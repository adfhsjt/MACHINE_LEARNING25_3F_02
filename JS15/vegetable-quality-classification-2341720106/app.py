import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

try:
    import gradio as gr
    _HAS_GRADIO = True
except Exception:
    _HAS_GRADIO = False

from skimage.color import rgb2hsv, hsv2rgb
from skimage import exposure

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a 3-channel RGB image.")
    img = image.astype(np.float32) / 255.0
    img_tf = tf.image.adjust_contrast(tf.image.adjust_brightness(tf.convert_to_tensor(img), 0.1), 1.3).numpy()
    img_tf = np.clip(img_tf, 0.0, 1.0)

    # Adaptive histogram equalization on V channel 
    try:
        hsv = rgb2hsv(img_tf)
        hsv[:, :, 2] = exposure.equalize_adapthist(hsv[:, :, 2])
        img_tf = hsv2rgb(hsv)
    except Exception:
        img_tf = np.clip(img_tf, 0.0, 1.0)

    # Boost saturation 
    hsv = rgb2hsv(img_tf)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0.0, 1.0)
    return np.clip(hsv2rgb(hsv), 0.0, 1.0).astype(np.float32)

MODEL_PATH = os.environ.get("MODEL_PATH", "model_mobilenetv2_classifier.keras")
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model ({MODEL_PATH}): {e}")
    model = None

class_names = ['utuh', 'tidak_utuh']

def predict_image(image):
    if model is None:
        return "Error: model not loaded."
    img = preprocess_image(image)
    inp = np.expand_dims(img, axis=0)
    preds = model.predict(inp)
    preds = np.array(preds)

    if np.isnan(preds).any() or not np.isfinite(preds).all():
        return "Prediction failed: NaN/inf in model output."

    if preds.ndim == 2 and preds.shape[1] == 2:
        probs = tf.nn.softmax(preds, axis=1).numpy()[0]
        idx = int(np.argmax(probs)); score = float(probs[idx])
    elif (preds.ndim == 2 and preds.shape[1] == 1) or preds.ndim == 1:
        raw = float(np.ravel(preds)[0])
        prob = float(tf.math.sigmoid(raw).numpy())
        idx = 1 if prob > 0.5 else 0; score = prob
    else:
        flat = preds.ravel()
        exp = np.exp(flat - np.max(flat))
        probs = exp / np.sum(exp)
        idx = int(np.argmax(probs)); score = float(probs[idx])

    idx = max(0, min(idx, len(class_names) - 1))
    return f"Prediction: {class_names[idx]} (score: {score:.3f})"

if _HAS_GRADIO and model is not None:
    iface = gr.Interface(fn=predict_image,
                         inputs=gr.Image(type="numpy", label="Upload Image"),
                         outputs=gr.Textbox(label="Prediction"),
                         title="Klasifikasi Keutuhan Sayur")
    iface.launch(share=False)
else:
    from flask import Flask, request, render_template_string
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def home():
        return render_template_string('''
            <div style="text-align:center;padding:50px;">
              <h1>Model Klasifikasi Keutuhan Sayur</h1>
              <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required><br><br>
                <button type="submit">Prediksi</button>
              </form>
            </div>
        ''')

    @app.route('/predict', methods=['POST'])
    def predict_route():
        if model is None:
            return "Error: model not loaded."
        f = request.files['file']
        data = np.frombuffer(f.read(), np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return "Could not decode image."
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return f"<h2 style='text-align:center'>{predict_image(img_rgb)}</h2><center><a href='/'>Kembali</a></center>"

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=7860)