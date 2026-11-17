import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

print("\n===== NEURON | FINAL MODEL =====")
stars = pd.read_csv("outputs/cleaned_stars.csv")

X = stars[["ci", "absmag"]]
y = stars["star_type"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\nTraining final model...")
model.fit(X, y)

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/star_classifier.pkl"
)

print("\n===== FINAL MODEL DETAILS =====")
print("Training stars:", len(X))
print("Features:", list(X.columns))
print("Categories:", list(model.classes_))
print("\nFeature importance:")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(f"{feature}: {importance:.4f}")

print("\nModel saved successfully!")
print("Location: models/star_classifier.pkl")
print("\nNEURON IS READY FOR DEPLOYMENT!")
