import sys
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from shap_utils import predict_and_explain_shap, FEATURE_NAMES, plot_to_base64

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

app = Flask(__name__, template_folder='frontend', static_folder='static')

# ---------------- Local SHAP ----------------
@app.route('/predict_shap', methods=['POST'])
def predict_shap():
    try:
        payload = request.get_json(force=True)

        missing = [feat for feat in FEATURE_NAMES if feat not in payload]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        df = pd.DataFrame([{feat: payload[feat] for feat in FEATURE_NAMES}])
        result = predict_and_explain_shap(df, global_analysis=False)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Global SHAP ----------------
@app.route('/global_shap', methods=['GET'])
def global_shap():
    try:
        sample_file = os.path.join(BASE_DIR, '..', 'main_app', 'backend', 'model', 'train_for_shap_original.csv')
        sample_data = pd.read_csv(sample_file)[FEATURE_NAMES].sample(50, random_state=42)

        result = predict_and_explain_shap(sample_data, global_analysis=True)

        # Store SHAP values + sample data in memory for dynamic dependence plots
        app.config["GLOBAL_SHAP_VALUES"] = result["values"]
        app.config["GLOBAL_SAMPLE_DATA"] = sample_data

        return jsonify({
            "shap_global_bar": result["shap_global_bar"],
            "shap_global_dependence": result["shap_global_dependence"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Dependence Plot for selected feature ----------------
@app.route('/dependence_plot', methods=['GET'])
def dependence_plot():
    try:
        feature_index = int(request.args.get("feature_index", -1))
        if "GLOBAL_SHAP_VALUES" not in app.config:
            return jsonify({"error": "Load global SHAP first"}), 400

        shap_values = app.config["GLOBAL_SHAP_VALUES"]
        sample_data = app.config["GLOBAL_SAMPLE_DATA"]

        import matplotlib.pyplot as plt
        plt.figure()
        import shap
        shap.dependence_plot(feature_index, shap_values, sample_data, show=False)
        dep_img = plot_to_base64()

        return jsonify({"shap_global_dependence": dep_img})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Home ----------------
@app.route('/')
def home():
    return render_template('xai_app.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8003)
