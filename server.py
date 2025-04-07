from flask import Flask
from main import run_project

app = Flask(__name__)

@app.route("/")
def index():
    run_project()
    return "✅ TLP Project finished successfully!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
