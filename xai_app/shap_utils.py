import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, base64, torch, shap, numpy as np, pandas as pd
from joblib import load
from io import BytesIO
import json

# ---------------- Paths ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, 'main_app', 'backend', 'model')

# ---------------- Artifacts ----------------
scaler = load(os.path.join(MODELS_DIR, 'scaler.joblib'))
label_encoder = load(os.path.join(MODELS_DIR, 'label_encoder.joblib'))
with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'r') as f:
    FEATURE_NAMES = json.load(f)

MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')
TRAIN_CSV = os.path.join(MODELS_DIR, 'train_for_shap_original.csv')

# ---------------- Load ANN ----------------
input_size = len(FEATURE_NAMES)
output_size = len(label_encoder.classes_)
from main_app.backend.model import ANN
model = ANN(input_size, 128, 64, output_size, 0.2)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ---------------- Helpers ----------------
def model_predict(X):
    if isinstance(X, pd.DataFrame): 
        X = X.values
    if isinstance(X, np.ndarray): 
        X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad(): 
        return model(X).numpy()

def plot_to_base64():
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img_base64

# ---------------- Main SHAP ----------------
def predict_and_explain_shap(sample_df: pd.DataFrame, top_k=5, global_analysis=False):
    # Load background data for SHAP explainer
    try:
        background = pd.read_csv(TRAIN_CSV)[FEATURE_NAMES].sample(100, random_state=42)
    except Exception as e:
        raise RuntimeError(f"Could not load background dataset: {e}")

    explainer = shap.Explainer(model_predict, background.values)
    shap_values_all = explainer(sample_df.values)

    if global_analysis:
        # Handle multi-class: aggregate across classes
        if shap_values_all.values.ndim == 3:
            values = shap_values_all.values.mean(axis=2)
        else:
            values = shap_values_all.values

        # ---------------- Global Bar Plot ----------------
        plt.figure(figsize=(6,4))
        shap_vals_mean = np.abs(values).mean(axis=0)
        y_pos = np.arange(len(FEATURE_NAMES))
        plt.barh(y_pos, shap_vals_mean, align='center', color='skyblue')
        plt.yticks(y_pos, FEATURE_NAMES)
        plt.gca().invert_yaxis()
        plt.xlabel('Mean |SHAP value|')
        plt.title("Global SHAP Feature Importance")
        shap_global_bar = plot_to_base64()

        # ---------------- Dependence plot for first feature ----------------
        plt.figure()
        shap.dependence_plot(0, values, sample_df, show=False)
        plt.title("Global SHAP Dependence Plot (First Feature)")
        shap_global_dependence = plot_to_base64()

        return {
            "shap_global_bar": shap_global_bar,
            "shap_global_dependence": shap_global_dependence,
            "values": values  # for JS dependence plots
        }

    else:
        # ---------------- Local Prediction ----------------
        sample_array = sample_df.values.astype(np.float32)
        with torch.no_grad():
            out = model(torch.tensor(sample_array))
        predicted_idx = int(torch.argmax(out[0]))
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

        # ---------------- Local SHAP values ----------------
        shap_vals_1d = shap_values_all.values[0] if shap_values_all.values.ndim==2 else shap_values_all.values.ravel()
        feat_contribs = [{"feature": f, "shap_value": float(v), "abs_value": abs(float(v))} for f,v in zip(FEATURE_NAMES, shap_vals_1d)]
        feat_contribs_sorted = sorted(feat_contribs, key=lambda d: d["abs_value"], reverse=True)[:top_k]

        # ---------------- Bar Plot ----------------
        plt.figure(figsize=(6,4))
        top_features = [d['feature'] for d in feat_contribs_sorted]
        top_shap_vals = [d['shap_value'] for d in feat_contribs_sorted]
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_shap_vals, color='skyblue')
        plt.yticks(y_pos, top_features)
        plt.gca().invert_yaxis()
        plt.xlabel('SHAP value')
        plt.title("Top Feature Contributions")
        shap_local_bar = plot_to_base64()

        return {
            "predicted_label": predicted_label,
            "top_contributors": feat_contribs_sorted,
            "all_contributors": feat_contribs,
            "shap_local_bar": shap_local_bar
        }

# ---------------- Dependence Plot ----------------
def dependence_plot(sample_df: pd.DataFrame, values: np.ndarray, feature_index: int):
    plt.figure(figsize=(7,5))
    shap.dependence_plot(feature_index, values, sample_df, show=False)
    plt.title(f"Dependence Plot - {FEATURE_NAMES[feature_index]}")
    return plot_to_base64()
