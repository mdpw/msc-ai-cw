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

# Load scaler and label encoder
scaler = load(os.path.join(MODELS_DIR, 'scaler.joblib'))
label_encoder = load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))

# Load feature names (12 features)
FEATURE_NAMES = [
    "Moisture", "Ash", "Volatile_Oil", "Acid_Insoluble_Ash",
    "Chromium", "Coumarin", "Fiber", "Density",
    "Oil_Content", "Resin", "Pesticide_Level", "PH_Value"
]

MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')

# ------------------------------
# Load model
# ------------------------------
input_size = len(FEATURE_NAMES)
output_size = len(label_encoder.classes_)
hidden1_size = 128
hidden2_size = 64
dropout_rate = 0.2

model = ANN(
    input_size=input_size,
    hidden1_size=hidden1_size,
    hidden2_size=hidden2_size,
    output_size=output_size,
    dropout_rate=dropout_rate
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ------------------------------
# SHAP Explainer
# ------------------------------
# Using 10 samples of zeros as fallback background
background = torch.zeros((10, input_size), dtype=torch.float32)
explainer = shap.DeepExplainer(model, background)

# ------------------------------
# Helper to generate base64 plot
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
    # 1. Ensure correct order of features and scale
    df_scaled = scaler.transform(sample_df[FEATURE_NAMES])
    x = torch.tensor(df_scaled, dtype=torch.float32)

    # 2. Model prediction
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # 3. SHAP values
    shap_values_all = explainer.shap_values(x)
    shap_vals_for_class = shap_values_all[pred_idx] if isinstance(shap_values_all, list) else shap_values_all
    shap_vals_for_class = np.array(shap_vals_for_class).flatten().astype(float)

    # 4. Top contributors
    feat_contribs = [
        {"feature": f, "shap_value": float(v), "abs_value": float(abs(v))}
        for f, v in zip(FEATURE_NAMES, shap_vals_for_class)
    ]
    feat_contribs_sorted = sorted(feat_contribs, key=lambda d: d["abs_value"], reverse=True)[:top_k]

    # 5. Generate base64 plot
    top_features = [d['feature'] for d in feat_contribs_sorted]
    top_shap_vals = [d['shap_value'] for d in feat_contribs_sorted]
    shap_img_base64 = shap_plot_to_base64(top_features, top_shap_vals)

    # 6. Return result
    result = {
        "predicted_label": pred_label,
        "probabilities": {label_encoder.classes_[i]: float(p) for i, p in enumerate(probs)},
        "top_contributors": feat_contribs_sorted,
        "all_contributors": feat_contribs,
        "shap_plot_base64": shap_img_base64
    }
    return result
