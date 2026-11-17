import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

print("\n===== NEURON | MODEL COMPARISON =====")
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

weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

models = {
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=15,
        random_state=42
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ["Gradient Boosting", "Hist Gradient Boosting"]:
        model.fit(
            X_train,
            y_train,
            sample_weight=weights
        )
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy * 100,
        "Balanced Accuracy": balanced * 100,
        "Macro F1": macro_f1 * 100
    })

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print("\nIndividual category performance:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="Macro F1",
    ascending=False
)
print("\n===== FINAL MODEL COMPARISON =====")
print(comparison.to_string(index=False))
comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)
print("\nComparison saved to outputs/model_comparison.csv")
