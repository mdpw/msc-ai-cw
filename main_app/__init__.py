from flask import Blueprint, render_template, request, jsonify
import torch
import pandas as pd
from .backend.model_loader import load_model
from .training.preprocess import FEATURE_NAMES, le, scaler
import os

# Blueprint with its own static folder
main_bp = Blueprint(
    'main_app',
    __name__,
    template_folder='frontend',
    static_folder='frontend/static',  # keep app-specific static
    static_url_path='/main_app/static'  # URL path to access static files
)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'backend', 'model', 'best_model.pth')
model, device = load_model(model_path)

@main_bp.route('/')
def home():
    return render_template('main_app.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sample_id = data.get('Sample_ID', 'Unknown')
        features = data['features']

        df = pd.DataFrame([features], columns=FEATURE_NAMES)
        input_tensor = torch.tensor(scaler.transform(df), dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_label = le.inverse_transform([predicted_class])[0]

        return jsonify({"Sample_ID": sample_id, "prediction": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)})
