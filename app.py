import sys
import os
from flask import Flask, render_template

# -------------------------
# Add root folder to sys.path
# -------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# Import blueprints
# -------------------------
from main_app import main_bp
from xai_app import xai_bp
from genai_app import genai_bp

# -------------------------
# Initialize main Flask app
# -------------------------
app = Flask(
    __name__,
    template_folder='frontend',  # root templates folder for homepage
    static_folder='static'        # optional static folder for root-level CSS/JS
)

# -------------------------
# Register blueprints
# -------------------------
app.register_blueprint(main_bp, url_prefix='/main_app')
app.register_blueprint(xai_bp, url_prefix='/xai_app')
app.register_blueprint(genai_bp, url_prefix='/genai_app')

# -------------------------
# Root homepage
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------
# Start the server
# -------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
