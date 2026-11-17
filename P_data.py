import pandas as pd
import re
from pathlib import Path

print("\n===== NEURON | DATA PREPARATION =====")
stars = pd.read_csv("data/stars-dataset.csv")
print("Original records:", len(stars))

def classify_star(spectral_type):
    if pd.isna(spectral_type):
        return None

    spectral_type = str(spectral_type).strip().upper()
    spectral_type = re.sub(r"\s+", "", spectral_type)

    if spectral_type.startswith("D"):
        return "White dwarf"

    suffixes = ("IAB", "IA", "IB", "III", "II", "IV", "V", "I")
    luminosity = None

    for suffix in suffixes:
        if spectral_type.endswith(suffix):
            luminosity = suffix
            break

    if luminosity is None:
        return None

    previous_text = spectral_type[:-len(luminosity)]
    if re.search(
        r"(?:IAB|IA|IB|III|II|IV|V|I)[/-]$",
        previous_text
    ):
        return None

    if luminosity in ("IA", "IB", "IAB", "I"):
        return "Supergiant"

    if luminosity == "III":
        return "Giant"

    if luminosity == "V":
        return "Main sequence"
    return None

stars["star_type"] = stars["spect"].apply(classify_star)
cleaned = stars[
    ["ci", "absmag", "star_type"]
].copy()

cleaned["ci"] = pd.to_numeric(
    cleaned["ci"],
    errors="coerce"
)

cleaned["absmag"] = pd.to_numeric(
    cleaned["absmag"],
    errors="coerce"
)

cleaned = cleaned.dropna()
cleaned = cleaned[
    cleaned["ci"].map(lambda x: abs(x) != float("inf")) &
    cleaned["absmag"].map(lambda x: abs(x) != float("inf"))
]

Path("outputs").mkdir(exist_ok=True)
cleaned.to_csv(
    "outputs/cleaned_stars.csv",
    index=False
)

print("\n===== CORRECTED DATASET =====")
print("\nOriginal records:", len(stars))
print("Usable labelled stars:", len(cleaned))
print("\nStar categories:")
print(cleaned["star_type"].value_counts())
print("\nCategory percentages:")
print(
    (
        cleaned["star_type"].value_counts(normalize=True) * 100
    ).round(2)
)

print("\nMissing values:")
print(cleaned.isnull().sum())
print("\nSaved: outputs/cleaned_stars.csv")
print("\nCLASSIFICATION CLEANUP COMPLETE!")
