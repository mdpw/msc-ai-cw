from flask import Flask, render_template

# -------------------------
# Initialize root Flask app
# -------------------------
app = Flask(__name__, template_folder='frontend', static_folder='frontend/static')

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
    # Run root app on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
