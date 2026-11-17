import pandas as pd
stars = pd.read_csv("data/stars-dataset.csv")
stars = stars.dropna(subset=["spect"]).copy()
stars["spect"] = (
    stars["spect"]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("\n===== NEURON | LABEL AUDIT =====")

subgiants = stars[
    stars["spect"].str.endswith("IV")
]

print("\n1. Classifications ending in IV:")
print("Total:", len(subgiants))
print(subgiants["spect"].value_counts().head(30))

supergiants = stars[
    stars["spect"].str.contains(
        r"(IA|IB|IAB)",
        regex=True
    )
]

print("\n2. Supergiant-like classifications:")
print("Total:", len(supergiants))
print(supergiants["spect"].value_counts().head(30))
white_dwarfs = stars[
    stars["spect"].str.startswith("D")
]

print("\n3. White dwarf classifications:")
print("Total:", len(white_dwarfs))
print(white_dwarfs["spect"].value_counts().head(30))

ambiguous = stars[
    stars["spect"].str.contains(
        r"[/?:+]",
        regex=True
    )
]

print("\n4. Potentially ambiguous classifications:")
print("Total:", len(ambiguous))
print(ambiguous["spect"].value_counts().head(30))
print("\n===== AUDIT COMPLETE =====")
