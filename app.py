from flask import Flask, request, jsonify, render_template
import torch
import os
import sys
from backend.explain_shap import predict_and_explain_shap
import json
import pandas as pd

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

MODELS_DIR = os.path.join('backend', 'model')
with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

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
        input_tensor = preprocess_input(features).to(device)

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

        # Make sure all features are present
        missing = [feat for feat in FEATURE_NAMES if feat not in payload]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Convert payload to DataFrame
        row = {feat: payload[feat] for feat in FEATURE_NAMES}
        df = pd.DataFrame([row])

        # Get prediction + SHAP explanation
        result = predict_and_explain_shap(df, top_k=5)

        # Return JSON including base64 plot
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/shap')
def shap_page():
    return render_template('shap_explanation.html')

# -------------------------
# Run server
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
