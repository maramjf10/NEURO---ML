import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n===== NEURON STAR CLASSIFIER =====")
print("Starting model training...\n")

stars = pd.read_csv("outputs/cleaned_stars.csv")

X = stars[["ci", "absmag"]]
y = stars["star_type"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training stars:", len(X_train))
print("Testing stars:", len(X_test))

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=3,
    random_state=42
)

print("\nTraining model...")
model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print("Training complete!")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
print("\n===== MODEL PERFORMANCE =====")
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

labels = model.classes_

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    matrix,
    index=labels,
    columns=labels
))

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
joblib.dump(model, "models/star_classifier.pkl")
pd.DataFrame(
    matrix,
    index=labels,
    columns=labels
).to_csv("outputs/confusion_matrix.csv")

results = X_test.copy()
results["actual"] = y_test
results["predicted"] = predictions
results.to_csv("outputs/test_predictions.csv", index=False)

print("\n===== SAVED SUCCESSFULLY =====")
print("Model: models/star_classifier.pkl")
print("Confusion matrix: outputs/confusion_matrix.csv")
print("Predictions: outputs/test_predictions.csv")
print("\nNEURON STAR CLASSIFIER IS READY!")
