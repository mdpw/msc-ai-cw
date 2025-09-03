from flask import Blueprint, render_template, request, jsonify

genai_bp = Blueprint('genai_app', __name__, template_folder='frontend', static_folder='static')

@genai_bp.route('/')
def home():
    return render_template('genai_app.html')

@genai_bp.route('/predict', methods=['POST'])
def predict_genai():
    # Add your GenAI prediction logic here
    return jsonify({"result": "This is GenAI output"})
