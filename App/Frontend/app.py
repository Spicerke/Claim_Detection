from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# URL of your running FastAPI backend
FASTAPI_URL = "http://backend:8000/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        
        if input_text:
            try:
                response = requests.post(FASTAPI_URL, json={"text": input_text})
                response.raise_for_status() 
                result = response.json()
                return render_template("result.html", text=input_text, result=result)
            
            except requests.exceptions.RequestException as e:
                error = "Failed to connect to the backend API. Run the FastAPI server and try again."
                return render_template("index.html", error=error)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)