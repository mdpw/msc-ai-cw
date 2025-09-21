import os
from flask import Flask, render_template, request
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='frontend', static_folder='frontend/static')

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.route("/", methods=["GET", "POST"])
def index():
    prompt_val = ""
    features_val = ""
    response = None

    if request.method == "POST":
        prompt_val = request.form["prompt"]
        features_val = request.form["features"]
        response = analyze_cinnamon_quality(prompt_val, features_val)

    return render_template(
        "genai_app.html",
        response=response,
        prompt_val=prompt_val,
        features_val=features_val
    )

def analyze_cinnamon_quality(prompt_text, features_text):
    full_prompt = f"{prompt_text}\n\nFeatures:\n{features_text}"
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        return response.text
    except Exception as e:
        print("Gemini API Error:", e)
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8003)
