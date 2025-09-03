from flask import Flask, request, jsonify, render_template
import os
import sys
import pandas as pd
import torch

# Add main_app paths to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(BASE_DIR, 'main_app'))
sys.path.append(os.path.join(BASE_DIR, 'main_app', 'backend'))
sys.path.append(os.path.join(BASE_DIR, 'main_app', 'training'))

from explain_shap import predict_and_explain_shap
from main_app.training.preprocess import FEATURE_NAMES, scaler, le
from main_app.backend.model_loader import load_model

# Initialize Flask app
xai_app = Flask(
    __name__,
    template_folder='frontend',
    static_folder='static'
)

# Load model
model_path = os.path.join(BASE_DIR, 'main_app', 'backend', 'model', 'best_model.pth')
model, device = load_model(model_path)

# Routes
@xai_app.route('/')
def home():
    return render_template('index.html')

@xai_app.route('/predict_shap', methods=['POST'])
def predict_shap():
    try:
        payload = request.get_json(force=True)
        missing = [feat for feat in FEATURE_NAMES if feat not in payload]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        row = {feat: payload[feat] for feat in FEATURE_NAMES}
        df = pd.DataFrame([row])

        result = predict_and_explain_shap(df, top_k=5)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#if __name__ == '__main__':
#    xai_app.run(debug=True, host="0.0.0.0", port=8003)