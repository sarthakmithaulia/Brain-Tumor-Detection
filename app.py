from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_model.keras')

# Ensure this matches the order of the labels used during training
CLASS_NAMES = ['No Tumor', 'Glioma', 'Meningioma' , 'Pituitary']  # Update if needed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def create_heatmap(img_path, prediction):
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        predicted_class = int(np.argmax(prediction))
        class_name = CLASS_NAMES[predicted_class]
        confidence = float(np.max(prediction) * 100)

        text = f"{class_name} ({confidence:.1f}%)"
        text_color = "red" if class_name != "No Tumor" else "green"

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([(10, 10), (20 + text_width, 20 + text_height)], fill="black")
        draw.text((15, 15), text, fill=text_color, font=font)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])

        # Dynamically identify index of "No Tumor"
        no_tumor_index = CLASS_NAMES.index("No Tumor")
        has_tumor = predicted_class != no_tumor_index
        tumor_type = CLASS_NAMES[predicted_class] if has_tumor else "No Tumor"

        visualization = create_heatmap(filepath, predictions[0])
        os.remove(filepath)

        return jsonify({
            'hasTumor': bool(has_tumor),
            'confidence': confidence,
            'tumorType': tumor_type,
            'visualization': visualization,
            'rawPrediction': [float(x) for x in predictions[0]]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
