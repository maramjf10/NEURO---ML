from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "star_classifier.pkl")
model = joblib.load(MODEL_PATH)

print("===================================")
print("       NEURON AI SYSTEM")
print("===================================")
print("Model loaded successfully!")
print("Available classes:", list(model.classes_))
print("===================================")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data received."}), 400

        ci = float(data.get("ci"))
        absmag = float(data.get("absmag"))

        if not np.isfinite(ci) or not np.isfinite(absmag):
            return jsonify({"error": "Please enter valid finite numbers."}), 400

        star = pd.DataFrame(
            [[ci, absmag]],
            columns=["ci", "absmag"]
        )

        prediction = model.predict(star)[0]
        probabilities = model.predict_proba(star)[0]

        results = {
            str(category): round(float(probability) * 100, 2)
            for category, probability in zip(model.classes_, probabilities)
        }

        return jsonify({
            "prediction": str(prediction),
            "probabilities": results,
            "input": {
                "ci": ci,
                "absmag": absmag
            }
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "Please enter valid numbers for both measurements."
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-info")
def model_info():
    return jsonify({
        "model": "Random Forest Classifier",
        "features": ["ci", "absmag"],
        "categories": [str(c) for c in model.classes_],
        "training_stars": 41926,
        "feature_importance": {
            name: round(float(value) * 100, 2)
            for name, value in zip(
                ["ci", "absmag"],
                model.feature_importances_
            )
        }
    })

@app.route("/api/stars")
def get_stars():
    try:
        data_path = os.path.join("outputs", "cleaned_stars.csv")
        df = pd.read_csv(data_path)
        df = df.dropna(subset=["ci", "absmag", "star_type"])

        stars = []
        for category, group in df.groupby("star_type"):
            sample = group.sample(
                n=min(len(group), 2500),
                random_state=42
            )

            stars.extend(
                sample[["ci", "absmag", "star_type"]]
                .to_dict(orient="records")
            )

        return jsonify({
            "stars": stars,
            "total_stars": len(df)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
