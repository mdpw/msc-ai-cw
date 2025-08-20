from flask import Flask, request, jsonify, render_template
import torch
import os
import sys

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


# -------------------------
# Run server
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
