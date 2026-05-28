from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # collect Q1–Q13
    q = [int(request.form[f"Q{i}"]) for i in range(1, 14)]

    age = int(request.form["Age"])
    gender = int(request.form["Gender"])

    features = q + [age, gender]

    prediction = model.predict([np.array(features)])

    result = "Autism Risk" if prediction[0] == 1 else "No Autism Risk"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)