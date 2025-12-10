from flask import Flask, request, jsonify
from flask_cors import CORS

# ML and numerical imports for equipment analytics
import pickle
import numpy as np

# Image and torch imports (for rock/crack analysis)
import io
import torch
from PIL import Image, ImageOps
import base64

# Your custom utility/model imports (these must exist in your project)
from utils.preprocessing import preprocess_input, load_image_from_bytes
from utils.gradcam import generate_gradcam
from models.rock_classifier import get_classifier_model, classify_with_threshold, class_names
from models.crack_segmenter import get_segmenter_model, segment_cracks

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

# Device selection for PyTorch models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load deep learning models (rock/crack)
classifier = get_classifier_model().to(device)
segmenter = get_segmenter_model().to(device)

# Load RandomForest and label encoders for equipment fault prediction
with open('models/model.pkl', 'rb') as f:
    clf, le_equipment, le_location = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return "MineSafety backend is running.", 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    img = request.files['image']
    image_pil = load_image_from_bytes(img.read())
    input_tensor = preprocess_input(image_pil).unsqueeze(0).to(device)

    pred_class, confidence = classify_with_threshold(classifier, input_tensor)
    if pred_class is None:
        return jsonify({
            'result': 'Not a rock type',
            'confidence': confidence,
            'gradcam': None,
            'segmentation_mask': None
        })

    rock_label = class_names[pred_class]
    cam = generate_gradcam(classifier, input_tensor, pred_class).cpu().numpy()
    cam_base64 = encode_image(cam, image_pil)
    mask = segment_cracks(segmenter, image_pil, device)
    mask_base64 = encode_mask(mask)

    return jsonify({
        'result': rock_label,
        'confidence': confidence,
        'gradcam': cam_base64,
        'segmentation_mask': mask_base64
    })

@app.route('/predict_life', methods=['POST'])
def predict_life():
    data = request.get_json()
    try:
        temperature = float(data['temperature'])
        pressure = float(data['pressure'])
        vibration = float(data['vibration'])
        humidity = float(data['humidity'])
        equipment = data['equipment']
        location = data['location']

        equipment_enc = le_equipment.transform([equipment])[0]
        location_enc = le_location.transform([location])[0]
        X = np.array([[temperature, pressure, vibration, humidity, equipment_enc, location_enc]])
        fault_pred = clf.predict(X)[0]
        probability = clf.predict_proba(X)[0][1]
        return jsonify({"fault_pred": int(fault_pred), "probability": float(probability)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Helper functions (GradCAM and segmentation mask encoding)
def encode_image(cam, pil_img):
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap_array = (cam_normalized * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_array).resize((224, 224)).convert('L')
    heatmap_img = ImageOps.colorize(heatmap_img, black="blue", white="red")
    img_resized = pil_img.resize((224, 224)).convert("RGBA")
    heatmap_img = heatmap_img.convert("RGBA")
    blended = Image.blend(img_resized, heatmap_img, alpha=0.4)
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def encode_mask(mask):
    mask_np = (mask.squeeze() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_np).resize((224, 224)).convert("L")
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

if __name__ == "__main__":
    app.run(debug=True)
