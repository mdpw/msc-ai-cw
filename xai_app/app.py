import sys
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from shap_utils import predict_and_explain_shap, FEATURE_NAMES

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

app = Flask(__name__, template_folder='frontend')

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
        # Query param: ?class_name=High / Medium / Low / Aggregate
        class_name = request.args.get("class_name")

        if class_name and class_name.lower() == "aggregate":
            class_name = None  # treat separately

        # Load background training data
        sample_file = os.path.join(BASE_DIR, 'main_app', 'backend', 'model', 'train_for_shap_original.csv')
        sample_data = pd.read_csv(sample_file)[FEATURE_NAMES].sample(50, random_state=42)

        result = predict_and_explain_shap(sample_data, global_analysis=True, class_name=class_name)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# ---------------- Home ----------------
@app.route('/')
def home():
    return render_template('xai_app.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8003)
