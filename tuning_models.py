import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

print("\n===== NEURON | MODEL OPTIMISATION =====")

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

searches = {
    "Extra Trees": (
        ExtraTreesClassifier(
            random_state=42,
            n_jobs=1
        ),
        {
            "n_estimators": [200, 300, 400],
            "max_depth": [None, 20, 35, 50],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.8, None],
            "class_weight": ["balanced"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=1
        ),
        {
            "n_estimators": [200, 300, 400],
            "max_depth": [None, 20, 35, 50],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.8, None],
            "class_weight": ["balanced", "balanced_subsample"]
        }
    )
}

results = []
best_models = {}

for name, (model, parameters) in searches.items():
    print(f"\nOptimising {name}...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=8,
        scoring={
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy"
        },
        refit="macro_f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)
    macro = f1_score(
        y_test,
        predictions,
        average="macro"
    )
    print("\nBest parameters:")
    print(search.best_params_)
    print("\nCross-validation Macro F1:")
    print(f"{search.best_score_ * 100:.2f}%")
    print("\nTest performance:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced * 100:.2f}%")
    print(f"Macro F1: {macro * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    results.append({
        "Model": name,
        "CV Macro F1": search.best_score_ * 100,
        "Test Accuracy": accuracy * 100,
        "Test Balanced Accuracy": balanced * 100,
        "Test Macro F1": macro * 100
    })

    best_models[name] = best_model

comparison = pd.DataFrame(results)
comparison = comparison.sort_values(
    by="Test Macro F1",
    ascending=False
)

print("\n===== OPTIMISED MODEL COMPARISON =====")
print(comparison.to_string(index=False))

Path("outputs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

comparison.to_csv(
    "outputs/optimised_comparison.csv",
    index=False
)

winner = comparison.iloc[0]["Model"]

joblib.dump(
    best_models[winner],
    "models/star_classifier_tuned.pkl"
)

print("\nBest model:", winner)
print("Saved: models/star_classifier_tuned.pkl")
print("\nOPTIMISATION COMPLETE!")
