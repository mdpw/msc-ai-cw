from flask import Flask, request, jsonify, render_template
import torch
import os
import sys
import json
import pandas as pd
from backend.explain_shap import predict_and_explain_shap
from training.preprocess import FEATURE_NAMES, preprocess_input, le, scaler

# -------------------------
# Add backend and training to sys.path
# -------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

from model_loader import load_model
from model import ANN
from preprocess import preprocess_input, le

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
model_path = os.path.join('backend', 'model', 'best_model.pth')
model, device = load_model(model_path)

# -------------------------
# Update FEATURE_NAMES for 12 features
# -------------------------
FEATURE_NAMES = [
    "Moisture", "Ash", "Volatile_Oil", "Acid_Insoluble_Ash", 
    "Chromium", "Coumarin", "Fiber", "Density", 
    "Oil_Content", "Resin", "Pesticide_Level", "PH_Value"
]

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')

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

@app.route("/predict_shap", methods=["POST"])
def predict_shap():
    try:
        payload = request.get_json(force=True)

        # Make sure all 12 features are present
        missing = [feat for feat in FEATURE_NAMES if feat not in payload]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Convert payload to DataFrame
        row = {feat: payload[feat] for feat in FEATURE_NAMES}
        df = pd.DataFrame([row])

        # Get prediction + SHAP explanation
        result = predict_and_explain_shap(df, top_k=5)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/shap')
def shap_page():
    return render_template('shap_explanation.html')

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    # Run on port 5001
    app.run(debug=True, host="0.0.0.0", port=5001)