import os
import json
import torch
import shap
import numpy as np
import pandas as pd
from joblib import load
from backend.model import ANN
import matplotlib.pyplot as plt
import io
import base64

# ------------------------------
# Paths and model artifacts
# ------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'model')

scaler = load(os.path.join(MODELS_DIR, 'scaler.joblib'))
label_encoder = load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
with open(os.path.join(MODELS_DIR, 'feature_names.json')) as f:
    FEATURE_NAMES = json.load(f)

MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')

# ------------------------------
# Load model
# ------------------------------
input_size = len(FEATURE_NAMES)
hidden_size = 64
output_size = len(label_encoder.classes_)

model = ANN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ------------------------------
# SHAP Explainer
# ------------------------------
background = torch.zeros((10, input_size))  # fallback background
explainer = shap.DeepExplainer(model, background)

# ------------------------------
# Helper to generate base64 image
# ------------------------------
def shap_plot_to_base64(feature_names, shap_values):
    plt.figure(figsize=(6, 4))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, shap_values, align='center', color='skyblue')
    plt.yticks(y_pos, feature_names)
    plt.xlabel('SHAP value')
    plt.title('Top Feature Contributions')
    plt.gca().invert_yaxis()  # highest contribution on top

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

# ------------------------------
# Main function
# ------------------------------
def predict_and_explain_shap(sample_df: pd.DataFrame, top_k: int = 5):
    # 1. Scale input
    scaled = scaler.transform(sample_df[FEATURE_NAMES])
    x = torch.tensor(scaled, dtype=torch.float32)

    # 2. Model prediction
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # 3. SHAP values
    shap_values_all = explainer.shap_values(x)  # list of arrays for each class
    # For classification, get values for predicted class
    if isinstance(shap_values_all, list):
        shap_vals_for_class = shap_values_all[pred_idx]
    else:
        shap_vals_for_class = shap_values_all  # fallback if single array

    # Flatten to 1D array safely
    shap_vals_for_class = np.array(shap_vals_for_class).flatten().astype(float)

    # 4. Top contributors
    feat_contribs = [
        {
            "feature": f,
            "shap_value": float(v),
            "abs_value": float(abs(v))
        }
        for f, v in zip(FEATURE_NAMES, shap_vals_for_class)
    ]
    feat_contribs_sorted = sorted(feat_contribs, key=lambda d: d["abs_value"], reverse=True)[:top_k]

    # 5. Generate base64 plot
    top_features = [d['feature'] for d in feat_contribs_sorted]
    top_shap_vals = [d['shap_value'] for d in feat_contribs_sorted]
    shap_img_base64 = shap_plot_to_base64(top_features, top_shap_vals)

    # 6. Result
    result = {
        "predicted_label": pred_label,
        "probabilities": {label_encoder.classes_[i]: float(p) for i, p in enumerate(probs)},
        "top_contributors": feat_contribs_sorted,
        "all_contributors": feat_contribs,
        "shap_plot_base64": shap_img_base64
    }
    return result
