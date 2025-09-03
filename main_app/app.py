import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'training'))
sys.path.append(os.path.join(BASE_DIR, 'backend'))

import torch
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from training.preprocess import preprocess_input, le, scaler
from backend.model_loader import load_model


# -------------------------
# Initialize Flask app
# -------------------------
app = Flask(
    __name__,
    template_folder='frontend',      # HTML files
    static_folder='frontend/static'  # CSS, images, JS
)

# -------------------------
# Load model
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, 'backend', 'model', 'best_model.pth')
model, device = load_model(model_path)

# -------------------------
# Update FEATURE_NAMES for 12 features
# -------------------------
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, 'backend', 'model', 'feature_names.json')
with open(FEATURE_NAMES_PATH, 'r') as f:
    FEATURE_NAMES = json.load(f)

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    return render_template('main_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sample_id = data.get('Sample_ID', 'Unknown')
        features = data['features']

        # Ensure the features are in correct order for the model
        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        input_tensor = torch.tensor(scaler.transform(features_df), dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = le.inverse_transform([predicted_class])[0]
        
        return jsonify({"Sample_ID": sample_id, "prediction": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    # Run on port 8001
   app.run(debug=True, host="0.0.0.0", port=8001)