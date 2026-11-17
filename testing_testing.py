import joblib
import pandas as pd

print("\n===== NEURON | STAR PREDICTION =====")

model = joblib.load("models/star_classifier.pkl")

print("\nModel loaded successfully!")

ci = float(input("\nEnter Colour Index (B-V): "))
absmag = float(input("Enter Absolute Magnitude: "))

star = pd.DataFrame(
    [[ci, absmag]],
    columns=["ci", "absmag"]
)

prediction = model.predict(star)[0]
probabilities = model.predict_proba(star)[0]

print("\n===== PREDICTION RESULTS =====")
print("Predicted star category:", prediction)
print("\nClassification probabilities:")

for category, probability in zip(
    model.classes_,
    probabilities
):
    print(f"{category}: {probability * 100:.2f}%")
print("\nNEURON PREDICTION COMPLETE!")
