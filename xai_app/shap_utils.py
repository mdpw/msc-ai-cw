import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask
import matplotlib.pyplot as plt
import sys
import os
import io
import base64
import torch
import shap
import numpy as np
import pandas as pd


from joblib import load
# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'main_app', 'backend', 'model')

# ---------------- Import ANN ----------------
from main_app.backend.model import ANN
import json
from io import BytesIO

# ---------------- Artifacts ----------------
scaler = load(os.path.join(MODELS_DIR, 'scaler.joblib'))
label_encoder = load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))

# ---------------- Feature Names ----------------
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.json')
with open(FEATURE_NAMES_PATH, 'r') as f:
    FEATURE_NAMES = json.load(f)

MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')

# ---------------- Load Model ----------------
input_size = len(FEATURE_NAMES)
output_size = len(label_encoder.classes_)
hidden1_size = 128
hidden2_size = 64
dropout_rate = 0.2

model = ANN(input_size, hidden1_size, hidden2_size, output_size, dropout_rate)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()


# ---- Wrapper for PyTorch model so SHAP always gets tensors ----
def model_predict(X):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        return outputs.numpy()  # raw logits

# ---- Convert matplotlib plot to base64 ----
def plot_to_base64():
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img_base64

# ---- Main SHAP function ----
def predict_and_explain_shap(sample_df: pd.DataFrame, top_k: int = 5, global_analysis=False, class_name=None):
    # ---------------- Explainer ----------------
    explainer = shap.Explainer(model_predict, sample_df.values)
    shap_values_all = explainer(sample_df.values)  # Explanation object

    results = {}

    if global_analysis:
        # ---------------- Multi-class handling ----------------
        if shap_values_all.values.ndim == 3:  # (samples, features, classes)
            if class_name:  # user selected a class
                try:
                    class_idx = list(label_encoder.classes_).index(class_name)
                except ValueError:
                    raise ValueError(f"Invalid class name: {class_name}")
                # Select only that class (samples × features)
                values = shap_values_all.values[:, :, class_idx]
                base_values = shap_values_all.base_values[:, class_idx]
                title_suffix = f" - {class_name}"
            else:  # Aggregate across classes
                values = shap_values_all.values.mean(axis=2)
                base_values = shap_values_all.base_values.mean(axis=1)
                title_suffix = " - Aggregate"
        else:  # binary / regression
            values = shap_values_all.values
            base_values = shap_values_all.base_values
            title_suffix = ""

        # ---------------- Summary Plot ----------------
        plt.figure()
        shap.summary_plot(values, sample_df, show=False)
        plt.title("Global SHAP Summary" + title_suffix)
        results["shap_global_summary"] = plot_to_base64()

        # ---------------- Beeswarm Plot ----------------
        plt.figure()
        shap.plots.beeswarm(
            shap.Explanation(
                values=values,
                base_values=base_values,
                data=sample_df.values,
                feature_names=FEATURE_NAMES
            ),
            show=False
        )
        plt.title("Global SHAP Beeswarm" + title_suffix)
        results["shap_global_beeswarm"] = plot_to_base64()

        # ---------------- Dependence Plot (first feature) ----------------
        plt.figure()
        shap.dependence_plot(0, values, sample_df, show=False)
        plt.title("Global SHAP Dependence" + title_suffix)
        results["shap_global_dependence"] = plot_to_base64()

        # ---------------- Force Plot (first sample) ----------------
        plt.figure(figsize=(10, 3))
        shap.plots.force(
            shap.Explanation(
                values=values,
                base_values=base_values,
                data=sample_df.values,
                feature_names=FEATURE_NAMES
            )[0],
            matplotlib=True,
            show=False
        )
        plt.title("Global SHAP Force - First Sample" + title_suffix)
        results["shap_global_force"] = plot_to_base64()

        return results

    else:
        # ---------------- Local SHAP ----------------
        shap_values_plot = shap_values_all

        # Convert to 1D array for single sample
        shap_vals_1d = shap_values_plot.values[0] if shap_values_plot.values.ndim == 2 else np.ravel(shap_values_plot.values)
        feat_contribs = [
            {"feature": f, "shap_value": float(v), "abs_value": float(abs(v))}
            for f, v in zip(FEATURE_NAMES, shap_vals_1d)
        ]
        feat_contribs_sorted = sorted(feat_contribs, key=lambda d: d["abs_value"], reverse=True)[:top_k]

        # ---------------- Bar Plot ----------------
        top_features = [d['feature'] for d in feat_contribs_sorted]
        top_shap_vals = [d['shap_value'] for d in feat_contribs_sorted]

        plt.figure(figsize=(6, 4))
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_shap_vals, align='center', color='skyblue')
        plt.yticks(y_pos, top_features)
        plt.gca().invert_yaxis()
        plt.xlabel('SHAP value')
        plt.title('Top Feature Contributions')
        shap_local_bar = plot_to_base64()

        return {
            "top_contributors": feat_contribs_sorted,
            "all_contributors": feat_contribs,
            "shap_local_bar": shap_local_bar
        }
