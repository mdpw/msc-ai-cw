import os
from flask import Flask, render_template, request
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='frontend', static_folder='static')

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        prompt = request.form["prompt"]
        features = request.form["features"]
        response = analyze_cinnamon_quality(prompt, features)
        return render_template("index.html", response=response)
    return render_template("index.html", response=None)

def analyze_cinnamon_quality(prompt_text, features_text):
    chat = client.chats.create(model="gemini-2.5-flash")
    full_prompt = f"{prompt_text}\n\nFeatures:\n{features_text}"
    response = chat.send_message(full_prompt)
    return response.text

#if __name__ == "__main__":
#    # Run on port 5000
#    app.run(debug=True, host="0.0.0.0", port=8002)