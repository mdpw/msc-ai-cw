import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Blueprint, render_template, request, jsonify
import pandas as pd
from main_app.backend.model_loader import load_model
from .explain_shap import predict_and_explain_shap
from main_app.training.preprocess import FEATURE_NAMES, scaler, le
import os

xai_bp = Blueprint('xai_app', __name__, template_folder='frontend', static_folder='static')

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../main_app/backend/model/best_model.pth')
model, device = load_model(model_path)

@xai_bp.route('/')
def home():
    return render_template('xai_app.html')

@xai_bp.route('/predict_shap', methods=['POST'])
def predict_shap():
    try:
        payload = request.get_json(force=True)
        missing = [feat for feat in FEATURE_NAMES if feat not in payload]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        df = pd.DataFrame([{feat: payload[feat] for feat in FEATURE_NAMES}])
        result = predict_and_explain_shap(df, top_k=5)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
