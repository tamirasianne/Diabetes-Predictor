import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
model = pickle.load(open("randomforest_model.pkl", "rb"))

# Features in the exact order used during training
features_list = [
    "Race",
    "GeneralHealth",
    "Sex",
    "Exercise",
    "HeavySmoker",
    "Age",
    "BMI",
    "HeavyAlcohol"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from form in correct order
    input_values = [int(request.form[f]) for f in features_list]

    # Build DataFrame exactly as model expects
    features = pd.DataFrame([input_values], columns=features_list)

    # Predict
    prediction = model.predict(features)[0]
    result = "Yes" if prediction == 1 else "No"

    return render_template("index.html",
                           prediction_text=f"Diabetes Risk Prediction: {result}")


if __name__ == "__main__":
    # This part is for local debugging
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)